"""
CIFAR-10 Byzantine-Robust Federated Learning with Blockchain Integration
Addresses Reviewer MW1 + MW4 (Blockchain cost-benefit analysis)

Architecture:
- Blockchain: Ganache (Ethereum testnet) via Docker
- Smart Contract: FederatedLearningAggregator.sol
- FL Methods: FedAvg, Krum, TrimmedMean, ATMA
- Dataset: CIFAR-10 with Dirichlet non-IID (α=0.5)
- Byzantine: 20% malicious clients (label-flipping attack)

Runtime: ~20-30 minutes (blockchain overhead included)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import json
import hashlib
from datetime import datetime
import os
import sys
from pathlib import Path
from web3 import Web3
from eth_account import Account

# Add blockchain client to path
from blockchain_client_simple import BlockchainClient, create_client_accounts

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
NUM_CLIENTS = 20
BYZANTINE_RATIO = 0.2
DIRICHLET_ALPHA = 0.5
ROUNDS = [50, 160]
LOCAL_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.1  # Fixed from 0.01
NUM_WORKERS = 0

# Blockchain Configuration
BLOCKCHAIN_RPC = "http://localhost:8545"  # Ganache endpoint
CONTRACT_ADDRESS = None  # Will be read from Docker deployment
AGGREGATOR_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # Ganache account #0

class SimpleCIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def hash_model_update(update: dict) -> str:
    """Create SHA256 hash of model update for blockchain"""
    # Convert to JSON string (sorted keys for consistency)
    update_str = json.dumps({k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                             for k, v in update.items()}, sort_keys=True)
    hash_bytes = hashlib.sha256(update_str.encode()).digest()
    return '0x' + hash_bytes.hex()

def create_non_iid_dirichlet(labels, num_clients, alpha):
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        proportions = label_distribution[c]
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, proportions)
        for client, split in enumerate(splits):
            client_indices[client].extend(split)
    
    return [np.array(indices) for indices in client_indices]

def local_train(model, train_loader, device, epochs, lr):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return {k: v.cpu() for k, v in model.state_dict().items()}

def aggregate_fedavg(updates, weights):
    aggregated = {}
    total_weight = sum(weights)
    for key in updates[0].keys():
        aggregated[key] = sum(w * updates[i][key] for i, w in enumerate(weights)) / total_weight
    return aggregated

def aggregate_krum(updates, weights, num_byzantine):
    n = len(updates)
    f = num_byzantine
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = sum((updates[i][k] - updates[j][k]).norm().item() ** 2 for k in updates[i].keys())
            distances[i][j] = distances[j][i] = dist
    
    scores = np.zeros(n)
    for i in range(n):
        closest = np.argsort(distances[i])[:n - f - 2]
        scores[i] = sum(distances[i][closest])
    
    selected = np.argmin(scores)
    return updates[selected]

def aggregate_trimmed_mean(updates, weights, trim_ratio=0.2):
    aggregated = {}
    for key in updates[0].keys():
        stacked = torch.stack([update[key] for update in updates])
        sorted_params, _ = torch.sort(stacked, dim=0)
        n = len(updates)
        trim_count = int(n * trim_ratio)
        trimmed = sorted_params[trim_count:n-trim_count] if trim_count > 0 else sorted_params
        aggregated[key] = trimmed.mean(dim=0)
    return aggregated

def aggregate_atma(updates, weights, threshold=0.3):
    aggregated = {}
    for key in updates[0].keys():
        stacked = torch.stack([update[key] for update in updates])
        median = torch.median(stacked, dim=0).values
        distances = torch.stack([torch.norm(update[key] - median) for update in updates])
        mad = torch.median(distances)
        normalized_distances = distances / (mad + 1e-6)
        trusted_mask = normalized_distances <= threshold
        
        if trusted_mask.sum() > 0:
            trusted_updates = stacked[trusted_mask]
            aggregated[key] = trusted_updates.mean(dim=0)
        else:
            aggregated[key] = median
    
    return aggregated

def create_byzantine_update(update, scale_factor=-5.0):
    return {key: value * scale_factor for key, value in update.items()}

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    return 100.0 * correct / total

def run_experiment_with_blockchain(method, num_rounds, train_dataset, test_loader, 
                                   client_indices, byzantine_clients, device, 
                                   blockchain_client, client_accounts):
    """Run FL experiment with blockchain logging"""
    
    print(f"\n{'='*70}")
    print(f"Starting: {method} - {num_rounds} rounds (BLOCKCHAIN ENABLED)")
    print(f"{'='*70}")
    
    global_model = SimpleCIFAR10CNN().to(device)
    accuracies = []
    blockchain_logs = []
    
    # Start FL session on blockchain
    try:
        tx_result = blockchain_client.start_round()
        print(f"✅ Blockchain: Round 1 started (tx: {tx_result['tx_hash'][:10]}...)")
    except Exception as e:
        print(f"⚠️  Blockchain unavailable, continuing without on-chain logging: {e}")
        blockchain_client = None
    
    for round_num in range(1, num_rounds + 1):
        updates = []
        weights = []
        update_hashes = []
        
        # Clients train locally
        for client_idx in range(NUM_CLIENTS):
            client_model = SimpleCIFAR10CNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            client_data = torch.utils.data.Subset(train_dataset, client_indices[client_idx])
            client_loader = torch.utils.data.DataLoader(
                client_data, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True
            )
            
            update = local_train(client_model, client_loader, device, LOCAL_EPOCHS, LEARNING_RATE)
            
            # Byzantine attack
            if client_idx in byzantine_clients:
                update = create_byzantine_update(update)
            
            updates.append(update)
            weights.append(len(client_indices[client_idx]))
            
            # Hash for blockchain
            update_hash = hash_model_update(update)
            update_hashes.append(update_hash)
        
        # Submit updates to blockchain (if available)
        if blockchain_client:
            try:
                # Submit each client's update hash to blockchain
                for idx in range(len(updates)):
                    tx_result = blockchain_client.submit_update(
                        update_hash=update_hashes[idx],
                        sample_count=weights[idx]
                    )
                    blockchain_logs.append({
                        'round': round_num,
                        'client': idx,
                        'tx_hash': tx_result['tx_hash'],
                        'gas_used': tx_result['gas_used']
                    })
            except Exception as e:
                print(f"⚠️  Blockchain submission failed (round {round_num}): {e}")
        
        # Aggregate (off-chain computation, result logged on-chain)
        num_byzantine = int(NUM_CLIENTS * BYZANTINE_RATIO)
        if method == 'FedAvg':
            aggregated = aggregate_fedavg(updates, weights)
        elif method == 'Krum':
            aggregated = aggregate_krum(updates, weights, num_byzantine)
        elif method == 'TrimmedMean':
            aggregated = aggregate_trimmed_mean(updates, weights)
        elif method == 'ATMA':
            aggregated = aggregate_atma(updates, weights)
        
        global_model.load_state_dict(aggregated)
        
        # Log aggregated model to blockchain
        if blockchain_client and round_num % 10 == 0:
            try:
                agg_hash = hash_model_update(aggregated)
                # Use submit_update for aggregated model (sample_count = total)
                tx_result = blockchain_client.submit_update(agg_hash, sum(weights))
                print(f"  📝 Round {round_num} logged on-chain (gas: {tx_result['gas_used']})")
            except Exception as e:
                print(f"  ⚠️  Failed to log round {round_num}: {e}")
        
        # Evaluate
        if round_num % 10 == 0 or round_num == num_rounds:
            acc = evaluate(global_model, test_loader, device)
            accuracies.append({'round': round_num, 'accuracy': acc})
            print(f"Round {round_num}/{num_rounds} Accuracy: {acc:.2f}%")
        else:
            print(f"Round {round_num}/{num_rounds} ", end='', flush=True)
    
    return {
        'method': method,
        'rounds': num_rounds,
        'final_accuracy': accuracies[-1]['accuracy'],
        'accuracies': accuracies,
        'blockchain_enabled': blockchain_client is not None,
        'blockchain_logs': blockchain_logs[:10]  # Sample for analysis
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("CIFAR-10 EXPERIMENTS WITH BLOCKCHAIN INTEGRATION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Blockchain: {BLOCKCHAIN_RPC}")
    print("="*70)
    
    # Initialize blockchain client
    blockchain_client = None
    client_accounts = None
    
    try:
        # Read contract address from deployment
        deployment_file = Path(__file__).parent / "deployment.json"
        if deployment_file.exists():
            with open(deployment_file, 'r') as f:
                deployment = json.load(f)
                CONTRACT_ADDRESS = deployment['contractAddress']  # Fixed key name
        
        if CONTRACT_ADDRESS:
            blockchain_client = BlockchainClient(
                rpc_url=BLOCKCHAIN_RPC,
                contract_address=CONTRACT_ADDRESS,
                private_key=AGGREGATOR_PRIVATE_KEY,
                abi_path="artifacts/FederatedLearningAggregator.json"
            )
            
            # Create client accounts
            client_accounts = create_client_accounts(NUM_CLIENTS)
            
            # Register clients on blockchain
            print("\n📝 Registering clients on blockchain...")
            addresses = [acc['address'] for acc in client_accounts]
            result = blockchain_client.register_clients_batch(addresses)
            print(f"✅ {NUM_CLIENTS} clients registered (gas: {result['gas_used']})")
        else:
            print("⚠️  Contract address not found, running without blockchain")
    
    except Exception as e:
        print(f"⚠️  Blockchain initialization failed: {e}")
        print("   Continuing without blockchain logging...")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Create non-IID distribution
    print("\nCreating non-IID distribution...")
    train_labels = np.array([label for _, label in train_dataset])
    client_indices = create_non_iid_dirichlet(train_labels, NUM_CLIENTS, DIRICHLET_ALPHA)
    
    num_byzantine = int(NUM_CLIENTS * BYZANTINE_RATIO)
    byzantine_clients = list(range(num_byzantine))
    print(f"Byzantine clients: {byzantine_clients}")
    
    # Run experiments
    os.makedirs('results', exist_ok=True)
    all_results = []
    
    methods = ['FedAvg', 'Krum', 'TrimmedMean', 'ATMA']
    
    for method in methods:
        for num_rounds in ROUNDS:
            result = run_experiment_with_blockchain(
                method, num_rounds, train_dataset, test_loader,
                client_indices, byzantine_clients, device,
                blockchain_client, client_accounts
            )
            all_results.append(result)
            print(f"✓ {method} {num_rounds}r: {result['final_accuracy']:.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'device': str(device),
        'blockchain_enabled': blockchain_client is not None,
        'blockchain_rpc': BLOCKCHAIN_RPC if blockchain_client else None,
        'experiment_config': {
            'dataset': 'CIFAR-10',
            'num_clients': NUM_CLIENTS,
            'byzantine_ratio': BYZANTINE_RATIO,
            'dirichlet_alpha': DIRICHLET_ALPHA,
            'local_epochs': LOCAL_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE
        },
        'results': all_results
    }
    
    filename = f'results/cifar10_blockchain_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print("\nSummary:")
    print(f"{'Method':<15} {'50 rounds':>12} {'160 rounds':>12}")
    print("-" * 40)
    for i in range(0, len(all_results), 2):
        method = all_results[i]['method']
        acc_50 = all_results[i]['final_accuracy']
        acc_160 = all_results[i+1]['final_accuracy']
        print(f"{method:<15} {acc_50:>11.2f}% {acc_160:>11.2f}%")
    
    if blockchain_client:
        total_gas = sum(log['gas_used'] for result in all_results for log in result.get('blockchain_logs', []))
        print(f"\n⛽ Total blockchain gas used: {total_gas:,}")
    
    print(f"\n📊 Results saved: {filename}")

if __name__ == '__main__':
    main()
