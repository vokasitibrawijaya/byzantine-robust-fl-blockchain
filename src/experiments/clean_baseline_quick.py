"""
QUICK Clean Baseline Experiment: FedProx and FedDyn WITHOUT Byzantine Attacks
50 rounds only for faster results

Purpose:
- Run FedProx, FedDyn, TrimmedMean, FedAvg WITHOUT any Byzantine attacks
- Calculate: (clean_acc - attacked_acc) / clean_acc = Byzantine Degradation %
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import json
from datetime import datetime
import copy
import os

# ============================================================================
# Configuration - QUICK VERSION (50 rounds, 1 seed)
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

NUM_CLIENTS = 20
DIRICHLET_ALPHA = 0.5
LEARNING_RATE = 0.01
BATCH_SIZE = 128
LOCAL_EPOCHS = 5
NUM_ROUNDS = 50  # Reduced from 160

SEED = 42  # Single seed for quick run
FEDPROX_MU = 0.01
FEDDYN_ALPHA = 0.01
TRIM_RATIO = 0.2

# ============================================================================
# Model Architecture
# ============================================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ============================================================================
# Data Distribution
# ============================================================================

def create_non_iid_dirichlet(labels, num_clients, alpha, seed):
    np.random.seed(seed)
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        proportions = label_distribution[class_idx]
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        client_class_indices = np.split(indices, splits)
        
        for client_id, idx_subset in enumerate(client_class_indices):
            client_indices[client_id].extend(idx_subset.tolist())
    
    return client_indices

# ============================================================================
# Training Functions
# ============================================================================

def train_client_standard(model, data_loader, epochs, lr, device):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def train_client_fedprox(model, data_loader, epochs, lr, device, global_model, mu):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    global_params = {name: param.clone().detach() for name, param in global_model.named_parameters()}
    
    for epoch in range(epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            proximal_term = 0.0
            for name, param in model.named_parameters():
                proximal_term += ((param - global_params[name]) ** 2).sum()
            loss += (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def train_client_feddyn(model, data_loader, epochs, lr, device, global_model, h_i, alpha):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    global_params = {name: param.clone().detach() for name, param in global_model.named_parameters()}
    
    for epoch in range(epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            idx = 0
            for name, param in model.named_parameters():
                param_flat = param.view(-1)
                global_flat = global_params[name].view(-1)
                h_i_slice = h_i[idx:idx + len(param_flat)]
                
                loss -= torch.dot(h_i_slice, param_flat - global_flat)
                loss += (alpha / 2) * ((param_flat - global_flat) ** 2).sum()
                
                idx += len(param_flat)
            
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

# ============================================================================
# Aggregation Functions
# ============================================================================

def fedavg_aggregate(client_weights, client_sizes):
    total_size = sum(client_sizes)
    avg_weights = {}
    
    for key in client_weights[0].keys():
        avg_weights[key] = sum(
            client_weights[i][key] * client_sizes[i] / total_size
            for i in range(len(client_weights))
        )
    
    return avg_weights

def trimmed_mean_aggregate(client_weights, trim_ratio=0.2):
    num_clients = len(client_weights)
    trim_count = int(num_clients * trim_ratio)
    
    if trim_count == 0:
        return fedavg_aggregate(client_weights, [1] * num_clients)
    
    avg_weights = {}
    for key in client_weights[0].keys():
        stacked = torch.stack([w[key].float() for w in client_weights])
        sorted_weights, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_weights[trim_count:-trim_count] if trim_count > 0 else sorted_weights
        avg_weights[key] = trimmed.mean(dim=0)
    
    return avg_weights

# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total

# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(method, seed=SEED, num_rounds=NUM_ROUNDS):
    print(f"\n{'='*60}")
    print(f"Running {method} (CLEAN BASELINE - NO ATTACKS)")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='../../data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='../../data', train=False, download=True, transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    
    train_labels = np.array(train_dataset.targets)
    client_indices = create_non_iid_dirichlet(train_labels, NUM_CLIENTS, DIRICHLET_ALPHA, seed)
    
    global_model = SimpleCNN().to(DEVICE)
    
    if method == 'FedDyn':
        total_params = sum(p.numel() for p in global_model.parameters())
        h_clients = [torch.zeros(total_params, device=DEVICE) for _ in range(NUM_CLIENTS)]
    
    accuracies = []
    
    for round_num in range(1, num_rounds + 1):
        client_weights = []
        client_sizes = []
        
        for client_id in range(NUM_CLIENTS):
            client_subset = torch.utils.data.Subset(train_dataset, client_indices[client_id])
            client_loader = torch.utils.data.DataLoader(
                client_subset, batch_size=BATCH_SIZE, shuffle=True
            )
            
            local_model = copy.deepcopy(global_model)
            
            if method == 'FedAvg':
                weights = train_client_standard(
                    local_model, client_loader, LOCAL_EPOCHS, LEARNING_RATE, DEVICE
                )
            elif method == 'FedProx':
                weights = train_client_fedprox(
                    local_model, client_loader, LOCAL_EPOCHS, LEARNING_RATE, DEVICE,
                    global_model, FEDPROX_MU
                )
            elif method == 'FedDyn':
                weights = train_client_feddyn(
                    local_model, client_loader, LOCAL_EPOCHS, LEARNING_RATE, DEVICE,
                    global_model, h_clients[client_id], FEDDYN_ALPHA
                )
                idx = 0
                for name, param in local_model.named_parameters():
                    param_flat = param.view(-1)
                    global_flat = dict(global_model.named_parameters())[name].view(-1)
                    h_clients[client_id][idx:idx + len(param_flat)] += FEDDYN_ALPHA * (param_flat - global_flat).detach()
                    idx += len(param_flat)
            elif method == 'TrimmedMean':
                weights = train_client_standard(
                    local_model, client_loader, LOCAL_EPOCHS, LEARNING_RATE, DEVICE
                )
            
            client_weights.append(weights)
            client_sizes.append(len(client_indices[client_id]))
        
        if method == 'TrimmedMean':
            new_weights = trimmed_mean_aggregate(client_weights, TRIM_RATIO)
        else:
            new_weights = fedavg_aggregate(client_weights, client_sizes)
        
        global_model.load_state_dict(new_weights)
        
        if round_num % 10 == 0 or round_num == 1:
            accuracy = evaluate(global_model, test_loader, DEVICE)
            accuracies.append({'round': round_num, 'accuracy': accuracy})
            print(f"  Round {round_num}: {accuracy:.2f}%")
    
    final_accuracy = evaluate(global_model, test_loader, DEVICE)
    print(f"Final: {final_accuracy:.2f}%")
    
    return final_accuracy, accuracies

def main():
    print("="*70)
    print("QUICK CLEAN BASELINE EXPERIMENT (50 rounds)")
    print("="*70)
    
    results = {}
    methods = ['FedAvg', 'TrimmedMean', 'FedProx', 'FedDyn']
    
    for method in methods:
        acc, history = run_experiment(method)
        results[method] = {'clean_accuracy': acc, 'history': history}
    
    # Known attacked results
    attacked = {
        'FedAvg': 10.0,
        'TrimmedMean': 66.38,  # 50 rounds attacked
        'FedProx': 10.0,
        'FedDyn': 10.0
    }
    
    print("\n" + "="*70)
    print("BYZANTINE DEGRADATION ANALYSIS (50 rounds)")
    print("="*70)
    print(f"{'Method':<15} {'Clean':<12} {'Attacked':<12} {'Degradation %':<15}")
    print("-" * 55)
    
    for method in methods:
        clean = results[method]['clean_accuracy']
        attack = attacked[method]
        deg = (clean - attack) / clean * 100 if clean > 0 else 0
        results[method]['attacked_accuracy'] = attack
        results[method]['degradation_percent'] = deg
        print(f"{method:<15} {clean:<12.2f} {attack:<12.2f} {deg:<15.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'experiment': 'Clean Baseline (NO Byzantine) - Quick 50 rounds',
        'config': {
            'num_clients': NUM_CLIENTS,
            'num_rounds': NUM_ROUNDS,
            'seed': SEED
        },
        'results': results
    }
    
    results_path = f"../../results/clean_baseline_quick_{timestamp}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\nSaved to: {results_path}")

if __name__ == "__main__":
    main()
