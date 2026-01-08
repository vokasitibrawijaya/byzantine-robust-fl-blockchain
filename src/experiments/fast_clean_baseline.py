"""
Fast Clean Baseline Experiment - CPU Optimized
NO Byzantine Attack - Pure federated learning baseline

This script runs FedProx and FedDyn WITHOUT any attack 
to get clean baseline accuracy for calculating Byzantine Degradation %.

Optimized for CPU with smaller model and fewer rounds.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import json
import time
from datetime import datetime
import os

# Force CPU
device = torch.device('cpu')
print(f"Using device: {device}")

# Smaller SimpleCNN optimized for fast training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_cifar10_loaders(num_clients=10, batch_size=64):
    """Create CIFAR-10 dataloaders with Dirichlet distribution"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    # Create non-IID split using Dirichlet(0.5)
    labels = np.array(train_dataset.targets)
    num_classes = 10
    alpha = 0.5
    
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        class_indices = np.where(labels == k)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()
        
        idx = 0
        for c in range(num_clients):
            client_indices[c].extend(class_indices[idx:idx + splits[c]])
            idx += splits[c]
    
    # Create dataloaders
    client_loaders = []
    for indices in client_indices:
        subset = torch.utils.data.Subset(train_dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        client_loaders.append(loader)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    return client_loaders, test_loader


def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total


def fedavg_aggregate(global_model, client_models, weights=None):
    """FedAvg aggregation"""
    if weights is None:
        weights = [1.0 / len(client_models)] * len(client_models)
    
    global_dict = global_model.state_dict()
    
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key])
        for i, client_model in enumerate(client_models):
            global_dict[key] += weights[i] * client_model.state_dict()[key]
    
    global_model.load_state_dict(global_dict)


def train_fedprox(global_model, client_loaders, test_loader, num_rounds=30, local_epochs=2, 
                  lr=0.01, mu=0.01, device='cpu'):
    """FedProx training WITHOUT any Byzantine attack"""
    
    print(f"\n{'='*60}")
    print(f"FedProx (Clean Baseline) - μ={mu}")
    print(f"Rounds: {num_rounds}, Local Epochs: {local_epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    accuracies = []
    start_time = time.time()
    
    for round_idx in range(num_rounds):
        global_model.train()
        global_dict = {k: v.clone() for k, v in global_model.state_dict().items()}
        
        client_models = []
        
        # ALL clients participate (no Byzantine)
        for client_idx, loader in enumerate(client_loaders):
            # Create local model
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_dict)
            optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
            
            # Local training with proximal term
            local_model.train()
            for epoch in range(local_epochs):
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                    
                    # Add proximal term
                    proximal_term = 0.0
                    for name, param in local_model.named_parameters():
                        proximal_term += ((param - global_dict[name].to(device)) ** 2).sum()
                    loss += (mu / 2) * proximal_term
                    
                    loss.backward()
                    optimizer.step()
            
            client_models.append(local_model)
        
        # Simple FedAvg aggregation (no Byzantine defense needed since no attack)
        fedavg_aggregate(global_model, client_models)
        
        # Evaluate
        acc = evaluate(global_model, test_loader, device)
        accuracies.append(acc)
        
        elapsed = time.time() - start_time
        print(f"Round {round_idx+1:3d}/{num_rounds} | Acc: {acc:.2f}% | Time: {elapsed:.1f}s")
    
    return accuracies


def train_feddyn(global_model, client_loaders, test_loader, num_rounds=30, local_epochs=2,
                 lr=0.01, alpha=0.01, device='cpu'):
    """FedDyn training WITHOUT any Byzantine attack"""
    
    print(f"\n{'='*60}")
    print(f"FedDyn (Clean Baseline) - α={alpha}")
    print(f"Rounds: {num_rounds}, Local Epochs: {local_epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    accuracies = []
    start_time = time.time()
    
    num_clients = len(client_loaders)
    
    # Initialize h vectors (gradient tracking)
    h_vectors = []
    for _ in range(num_clients):
        h = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        h_vectors.append(h)
    
    for round_idx in range(num_rounds):
        global_model.train()
        global_dict = {k: v.clone() for k, v in global_model.state_dict().items()}
        
        client_models = []
        
        for client_idx, loader in enumerate(client_loaders):
            # Create local model
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_dict)
            optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
            
            # Local training with dynamic regularization
            local_model.train()
            for epoch in range(local_epochs):
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                    
                    # FedDyn regularization: gradient correction
                    for name, param in local_model.named_parameters():
                        if name in h_vectors[client_idx]:
                            loss += (alpha / 2) * ((param - global_dict[name].to(device)) ** 2).sum()
                            loss -= (h_vectors[client_idx][name].to(device) * param).sum()
                    
                    loss.backward()
                    optimizer.step()
            
            # Update h vector
            for name, param in local_model.named_parameters():
                h_vectors[client_idx][name] = h_vectors[client_idx][name] - alpha * (
                    param.data - global_dict[name]
                )
            
            client_models.append(local_model)
        
        # Simple aggregation
        fedavg_aggregate(global_model, client_models)
        
        # Evaluate
        acc = evaluate(global_model, test_loader, device)
        accuracies.append(acc)
        
        elapsed = time.time() - start_time
        print(f"Round {round_idx+1:3d}/{num_rounds} | Acc: {acc:.2f}% | Time: {elapsed:.1f}s")
    
    return accuracies


def main():
    print("=" * 70)
    print("CLEAN BASELINE EXPERIMENT - NO BYZANTINE ATTACK")
    print("Purpose: Get baseline accuracy for FedProx & FedDyn")
    print("=" * 70)
    
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration - optimized for CPU
    NUM_CLIENTS = 10
    NUM_ROUNDS = 30  # Reduced for faster execution
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 64
    LR = 0.01
    
    # Load data
    print("\nLoading CIFAR-10 data...")
    client_loaders, test_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)
    print(f"Data loaded: {NUM_CLIENTS} clients")
    
    results = {
        'experiment': 'clean_baseline_no_attack',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_clients': NUM_CLIENTS,
            'num_rounds': NUM_ROUNDS,
            'local_epochs': LOCAL_EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'byzantine_fraction': 0.0,
            'attack_type': 'none'
        },
        'methods': {}
    }
    
    # Test FedProx (clean)
    print("\n" + "="*70)
    print("Testing FedProx WITHOUT Byzantine Attack")
    print("="*70)
    
    model_fedprox = SimpleCNN().to(device)
    fedprox_accs = train_fedprox(
        model_fedprox, client_loaders, test_loader,
        num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS,
        lr=LR, mu=0.01, device=device
    )
    
    results['methods']['FedProx'] = {
        'clean_baseline': True,
        'final_accuracy': fedprox_accs[-1],
        'max_accuracy': max(fedprox_accs),
        'accuracies': fedprox_accs,
        'mu': 0.01
    }
    
    print(f"\n>>> FedProx Clean Baseline: {fedprox_accs[-1]:.2f}% (max: {max(fedprox_accs):.2f}%)")
    
    # Test FedDyn (clean)
    print("\n" + "="*70)
    print("Testing FedDyn WITHOUT Byzantine Attack")
    print("="*70)
    
    model_feddyn = SimpleCNN().to(device)
    feddyn_accs = train_feddyn(
        model_feddyn, client_loaders, test_loader,
        num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS,
        lr=LR, alpha=0.01, device=device
    )
    
    results['methods']['FedDyn'] = {
        'clean_baseline': True,
        'final_accuracy': feddyn_accs[-1],
        'max_accuracy': max(feddyn_accs),
        'accuracies': feddyn_accs,
        'alpha': 0.01
    }
    
    print(f"\n>>> FedDyn Clean Baseline: {feddyn_accs[-1]:.2f}% (max: {max(feddyn_accs):.2f}%)")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f'clean_baseline_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("CLEAN BASELINE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"FedProx (No Attack): {fedprox_accs[-1]:.2f}%")
    print(f"FedDyn (No Attack):  {feddyn_accs[-1]:.2f}%")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    main()
