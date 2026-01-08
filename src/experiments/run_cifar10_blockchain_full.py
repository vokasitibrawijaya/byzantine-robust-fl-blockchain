"""
CIFAR-10 FL with Full Blockchain Integration
Uses smart contract for audit trail and gas cost tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import json
from datetime import datetime
import os
from pathlib import Path
from blockchain_client_working import BlockchainClient

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
NUM_CLIENTS = 20
BYZANTINE_RATIO = 0.2
DIRICHLET_ALPHA = 0.5
ROUNDS = [50, 160]
LOCAL_EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.01  # CRITICAL: 0.1 was too high
NUM_WORKERS = 0

print("=" * 70)
print("CIFAR-10 + BLOCKCHAIN EXPERIMENTS (SMART CONTRACT)")
print("=" * 70)

# Check CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Learning Rate: {LEARNING_RATE}")
print("=" * 70)

# Initialize blockchain
try:
    blockchain = BlockchainClient()
    blockchain_enabled = True
except Exception as e:
    print(f"⚠️  Blockchain not available: {e}")
    print("Running without blockchain integration")
    blockchain_enabled = False

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
    
    return model.state_dict()

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def byzantine_label_flip(update, scale=-5.0):
    """Strong Byzantine attack"""
    attacked = {}
    for k, v in update.items():
        attacked[k] = v * scale
    return attacked

def aggregate_fedavg(updates, weights):
    aggregated = {}
    total_weight = sum(weights)
    
    for key in updates[0].keys():
        aggregated[key] = torch.sum(
            torch.stack([updates[i][key] * weights[i] for i in range(len(updates))]),
            dim=0
        ) / total_weight
    
    return aggregated

def aggregate_krum(updates, weights, num_byzantine):
    n = len(updates)
    m = num_byzantine
    
    # Flatten updates
    flat_updates = []
    for update in updates:
        flat = torch.cat([v.flatten() for v in update.values()])
        flat_updates.append(flat)
    
    # Compute pairwise distances
    scores = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = torch.norm(flat_updates[i] - flat_updates[j]).item()
                distances.append(dist)
        distances.sort()
        score = sum(distances[:n - m - 2])
        scores.append(score)
    
    # Select client with minimum score
    selected_idx = scores.index(min(scores))
    return updates[selected_idx]

def aggregate_trimmed_mean(updates, weights, trim_ratio=0.2):
    aggregated = {}
    num_trim = int(len(updates) * trim_ratio)
    
    for key in updates[0].keys():
        stacked = torch.stack([updates[i][key] for i in range(len(updates))])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[num_trim:-num_trim] if num_trim > 0 else sorted_vals
        aggregated[key] = torch.mean(trimmed, dim=0)
    
    return aggregated

def aggregate_atma(updates, weights, threshold=0.3):
    aggregated = {}
    
    for key in updates[0].keys():
        stacked = torch.stack([updates[i][key] for i in range(len(updates))])
        median = torch.median(stacked, dim=0)[0]
        
        # Compute distances from median
        distances = torch.stack([torch.abs(stacked[i] - median) for i in range(len(updates))])
        mad = torch.median(distances, dim=0)[0]
        
        # Filter updates within threshold
        mask = distances < (threshold * (mad + 1e-6))
        filtered = torch.where(mask, stacked, median.unsqueeze(0).expand_as(stacked))
        aggregated[key] = torch.mean(filtered, dim=0)
    
    return aggregated

# Load CIFAR-10
print("\nCreating non-IID distribution...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

train_labels = np.array([label for _, label in train_dataset])
client_indices = create_non_iid_dirichlet(train_labels, NUM_CLIENTS, DIRICHLET_ALPHA)

# Determine Byzantine clients
num_byzantine = int(NUM_CLIENTS * BYZANTINE_RATIO)
byzantine_clients = list(range(num_byzantine))
print(f"Byzantine clients: {byzantine_clients}")

# Register clients in blockchain
if blockchain_enabled:
    print("\n⛓️  Registering clients in blockchain...")
    gas_register = blockchain.register_clients(NUM_CLIENTS)
    print(f"   Gas used: {gas_register}")

# Test loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

# Experiment configurations
methods = {
    'FedAvg': lambda u, w, _: aggregate_fedavg(u, w),
    'Krum': lambda u, w, _: aggregate_krum(u, w, num_byzantine),
    'TrimmedMean': lambda u, w, _: aggregate_trimmed_mean(u, w),
    'ATMA': lambda u, w, _: aggregate_atma(u, w)
}

results = {}

for method_name, aggregator in methods.items():
    for num_rounds in ROUNDS:
        print("\n" + "=" * 70)
        print(f"{method_name} - {num_rounds} rounds")
        print("=" * 70)
        
        # Initialize model
        global_model = SimpleCIFAR10CNN().to(device)
        
        for round_num in range(1, num_rounds + 1):
            # Blockchain: Start round
            if blockchain_enabled:
                gas_start = blockchain.start_round()
            
            # Local training
            updates = []
            weights = []
            
            for client_id in range(NUM_CLIENTS):
                # Create client data loader
                client_data = torch.utils.data.Subset(train_dataset, client_indices[client_id])
                client_loader = torch.utils.data.DataLoader(
                    client_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
                )
                
                # Local training
                local_model = SimpleCIFAR10CNN().to(device)
                local_model.load_state_dict(global_model.state_dict())
                update = local_train(local_model, client_loader, device, LOCAL_EPOCHS, LEARNING_RATE)
                
                # Byzantine attack
                if client_id in byzantine_clients:
                    update = byzantine_label_flip(update)
                    if blockchain_enabled:
                        model_hash = blockchain.hash_model_update(update)
                        gas_mark = blockchain.mark_byzantine(round_num, client_id, "Label flip attack")
                
                updates.append(update)
                weights.append(len(client_indices[client_id]))
                
                # Blockchain: Submit update
                if blockchain_enabled:
                    model_hash = blockchain.hash_model_update(update)
                    gas_submit = blockchain.submit_update(client_id, model_hash, len(client_indices[client_id]))
            
            # Aggregate
            aggregated_update = aggregator(updates, weights, None)
            global_model.load_state_dict(aggregated_update)
            
            # Blockchain: Complete round
            if blockchain_enabled:
                model_hash = blockchain.hash_model_update(aggregated_update)
                gas_complete = blockchain.complete_round(model_hash)
            
            # Evaluate every 10 rounds
            if round_num % 10 == 0 or round_num == num_rounds:
                accuracy = evaluate(global_model, test_loader, device)
                print(f"Round {round_num}/{num_rounds} Acc: {accuracy:.2f}%", end=" ")
        
        # Final evaluation
        final_accuracy = evaluate(global_model, test_loader, device)
        results[f"{method_name}_{num_rounds}r"] = final_accuracy
        print(f"\n✓ {method_name} {num_rounds}r: {final_accuracy:.2f}%")

# Print summary
print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
print(f"\n{'Method':<20} {'50 rounds':>12} {'160 rounds':>12}")
print("-" * 46)
for method in methods.keys():
    r50 = results.get(f"{method}_50r", 0)
    r160 = results.get(f"{method}_160r", 0)
    print(f"{method:<20} {r50:>11.2f}% {r160:>11.2f}%")

# Gas summary
if blockchain_enabled:
    print("\n" + "=" * 70)
    print("BLOCKCHAIN GAS USAGE")
    print("=" * 70)
    gas_summary = blockchain.get_gas_summary()
    
    for operation, stats in gas_summary.items():
        if operation == 'total_gas':
            print(f"\n{'Total Gas Used:':<30} {stats:,}")
        elif operation == 'estimated_cost_eth':
            print(f"{'Estimated Cost (ETH):':<30} {stats:.6f} ETH")
        elif operation == 'estimated_cost_usd':
            print(f"{'Estimated Cost (USD):':<30} ${stats:.2f}")
        elif stats['count'] > 0:
            print(f"\n{operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {stats['total']:,} gas")
            print(f"  Average: {stats['average']:,.0f} gas")
            print(f"  Min/Max: {stats['min']:,} / {stats['max']:,} gas")

# Save results
os.makedirs('results', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"results/cifar10_blockchain_full_{timestamp}.json"

results_data = {
    'timestamp': timestamp,
    'configuration': {
        'num_clients': NUM_CLIENTS,
        'byzantine_ratio': BYZANTINE_RATIO,
        'dirichlet_alpha': DIRICHLET_ALPHA,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'local_epochs': LOCAL_EPOCHS
    },
    'results': results,
    'blockchain_enabled': blockchain_enabled,
    'gas_summary': gas_summary if blockchain_enabled else None
}

with open(results_file, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\n📊 Results saved: {results_file}")
print("=" * 70)
