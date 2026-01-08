"""
Minimal Clean Baseline Experiment (3 rounds only)
Purpose: Get at least ONE real data point for CIFAR-10 clean baseline
For full experiments, use GPU or run overnight
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simpler CNN - faster training
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def dirichlet_split(dataset, num_clients, alpha=0.5):
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

def local_train(model, train_loader, epochs=2, lr=0.01):
    """Fewer local epochs for speed"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def aggregate_fedavg(weights_list):
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = torch.stack([w[key].float() for w in weights_list]).mean(dim=0)
    return avg_weights

def aggregate_trimmed_mean(weights_list, trim_ratio=0.2):
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

def run_clean_baseline(method_name, aggregate_fn, num_rounds=3, num_clients=10, seed=42):
    """Minimal clean baseline - reduced clients and rounds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
    
    # Fewer clients for speed
    client_indices = dirichlet_split(train_dataset, num_clients, alpha=0.5)
    
    client_loaders = []
    for indices in client_indices:
        subset = torch.utils.data.Subset(train_dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=256, shuffle=True)  # Larger batch
        client_loaders.append(loader)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    global_model = TinyCNN().to(device)
    
    accuracies = []
    print(f"\n{'='*50}")
    print(f"Running {method_name} (CLEAN - NO ATTACKS)")
    print(f"{'='*50}")
    
    for round_num in range(1, num_rounds + 1):
        client_weights = []
        print(f"  Round {round_num}: Training clients... ", end="", flush=True)
        for client_id in range(num_clients):
            local_model = TinyCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())
            weights = local_train(local_model, client_loaders[client_id], epochs=2, lr=0.01)
            client_weights.append(weights)
        
        new_weights = aggregate_fn(client_weights)
        global_model.load_state_dict(new_weights)
        
        acc = evaluate(global_model, test_loader)
        accuracies.append(acc)
        print(f"{acc:.2f}%")
    
    return {
        "method": method_name,
        "rounds": num_rounds,
        "final_accuracy": accuracies[-1],
        "all_accuracies": accuracies,
        "seed": seed
    }

def main():
    print("=" * 60)
    print("MINIMAL CLEAN BASELINE EXPERIMENT (3 rounds)")
    print("CIFAR-10, 10 clients, Dirichlet(0.5), NO Byzantine attacks")
    print("=" * 60)
    
    results = {}
    
    # FedAvg clean baseline
    results["FedAvg"] = run_clean_baseline("FedAvg", aggregate_fedavg, num_rounds=3, num_clients=10)
    
    # TrimmedMean clean baseline
    results["TrimmedMean"] = run_clean_baseline("TrimmedMean", 
                                                lambda w: aggregate_trimmed_mean(w, 0.2),
                                                num_rounds=3, num_clients=10)
    
    # Save results
    output = {
        "experiment": "clean_baseline_minimal",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "device": str(device),
        "config": {
            "dataset": "CIFAR-10",
            "num_clients": 10,
            "byzantine_ratio": 0.0,
            "dirichlet_alpha": 0.5,
            "local_epochs": 2,
            "num_rounds": 3,
            "note": "Minimal experiment for quick baseline estimation. For accurate results, run with 50+ rounds and 20 clients."
        },
        "results": results
    }
    
    output_path = "../../results/clean_baseline_minimal.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for method, res in results.items():
        print(f"  {method}: {res['final_accuracy']:.2f}% (round 3)")
    print(f"\nSaved to: {output_path}")
    
    # Extrapolation note
    print("\n" + "="*60)
    print("NOTE: This is a minimal experiment (3 rounds, 10 clients).")
    print("For comparison with 50/160 round attacked results,")
    print("extrapolate or run full experiment with GPU.")
    print("="*60)

if __name__ == "__main__":
    main()
