"""
Super Quick Clean Baseline Experiment (10 rounds only)
Purpose: Get CIFAR-10 clean baseline accuracies for Byzantine Degradation % calculation
NO Byzantine attacks - honest clients only
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import json
from datetime import datetime
import copy

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def dirichlet_split(dataset, num_clients, alpha=0.5):
    """Split dataset using Dirichlet distribution for non-IID"""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = 10
    
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()
        
        start = 0
        for i in range(num_clients):
            client_indices[i].extend(class_indices[start:start+splits[i]])
            start += splits[i]
    
    return client_indices

def local_train(model, train_loader, epochs=5, lr=0.01):
    """Local training on client"""
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

def aggregate_fedavg(weights_list):
    """FedAvg aggregation"""
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = torch.stack([w[key].float() for w in weights_list]).mean(dim=0)
    return avg_weights

def aggregate_trimmed_mean(weights_list, trim_ratio=0.2):
    """Trimmed Mean aggregation"""
    avg_weights = {}
    n = len(weights_list)
    trim_count = int(n * trim_ratio)
    
    for key in weights_list[0].keys():
        stacked = torch.stack([w[key].float() for w in weights_list])
        if trim_count > 0 and n > 2 * trim_count:
            sorted_weights, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_weights[trim_count:n-trim_count]
            avg_weights[key] = trimmed.mean(dim=0)
        else:
            avg_weights[key] = stacked.mean(dim=0)
    return avg_weights

def evaluate(model, test_loader):
    """Evaluate model accuracy"""
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

def run_clean_baseline(method_name, aggregate_fn, num_rounds=10, num_clients=20, seed=42):
    """Run clean baseline experiment"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
    
    # Split data using Dirichlet
    client_indices = dirichlet_split(train_dataset, num_clients, alpha=0.5)
    
    # Create client data loaders
    client_loaders = []
    for indices in client_indices:
        subset = torch.utils.data.Subset(train_dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)
        client_loaders.append(loader)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize global model
    global_model = SimpleCNN().to(device)
    
    accuracies = []
    print(f"\n{'='*60}")
    print(f"Running {method_name} (CLEAN BASELINE - NO ATTACKS)")
    print(f"{'='*60}")
    
    for round_num in range(1, num_rounds + 1):
        # All clients train (no Byzantine)
        client_weights = []
        for client_id in range(num_clients):
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())
            weights = local_train(local_model, client_loaders[client_id], epochs=5, lr=0.01)
            client_weights.append(weights)
        
        # Aggregate
        new_weights = aggregate_fn(client_weights)
        global_model.load_state_dict(new_weights)
        
        # Evaluate
        acc = evaluate(global_model, test_loader)
        accuracies.append(acc)
        print(f"  Round {round_num}: {acc:.2f}%")
    
    return {
        "method": method_name,
        "rounds": num_rounds,
        "final_accuracy": accuracies[-1],
        "all_accuracies": accuracies,
        "seed": seed
    }

def main():
    print("=" * 70)
    print("SUPER QUICK CLEAN BASELINE EXPERIMENT (10 rounds)")
    print("CIFAR-10, 20 clients, Dirichlet(0.5), NO Byzantine attacks")
    print("=" * 70)
    
    results = {}
    
    # FedAvg
    results["FedAvg"] = run_clean_baseline("FedAvg", aggregate_fedavg, num_rounds=10)
    
    # Trimmed Mean
    results["TrimmedMean"] = run_clean_baseline("TrimmedMean", 
                                                lambda w: aggregate_trimmed_mean(w, 0.2),
                                                num_rounds=10)
    
    # Save results
    output = {
        "experiment": "clean_baseline_10r",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "device": str(device),
        "config": {
            "dataset": "CIFAR-10",
            "num_clients": 20,
            "byzantine_ratio": 0.0,
            "dirichlet_alpha": 0.5,
            "local_epochs": 5,
            "num_rounds": 10
        },
        "results": results
    }
    
    # Save to results folder
    output_path = "../../results/clean_baseline_10r.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*70}")
    print("CLEAN BASELINE RESULTS SUMMARY")
    print(f"{'='*70}")
    for method, res in results.items():
        print(f"  {method}: {res['final_accuracy']:.2f}% (round 10)")
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
