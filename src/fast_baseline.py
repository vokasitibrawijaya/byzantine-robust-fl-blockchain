"""
FAST Clean Baseline - Quick test version
========================================
10 rounds, 3 local epochs untuk test cepat
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
import copy

# Verify GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# ===================== FAST CONFIG =====================
CONFIG = {
    'num_clients': 10,
    'num_rounds': 20,  # FAST: 20 rounds instead of 50
    'local_epochs': 3,  # FAST: 3 epochs instead of 5
    'batch_size': 64,  # FAST: larger batch
    'learning_rate': 0.01,
    'dirichlet_alpha': 0.5,
    'seed': 42,
}

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def dirichlet_split(dataset, num_clients, alpha, seed):
    np.random.seed(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(10):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)
        for i, idx in enumerate(idx_batch):
            client_indices[i].extend(idx.tolist())
    
    return client_indices

def local_train(model, data_loader, epochs, lr, device, proximal_mu=0.0, global_weights=None):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            if proximal_mu > 0 and global_weights is not None:
                proximal_term = sum(((p - global_weights[n].to(device)) ** 2).sum() 
                                   for n, p in model.named_parameters())
                loss += (proximal_mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
    
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

def fedavg_aggregate(client_weights, num_samples):
    total = sum(num_samples)
    return {k: sum(w[k].float() * (n/total) for w, n in zip(client_weights, num_samples))
            for k in client_weights[0].keys()}

def trimmedmean_aggregate(client_weights, trim_ratio=0.2):
    num_trim = max(1, int(len(client_weights) * trim_ratio))
    return {k: torch.sort(torch.stack([w[k].float() for w in client_weights]), dim=0)[0][num_trim:-num_trim].mean(dim=0)
            for k in client_weights[0].keys()}

def run_experiment(attack_mode=False, byzantine_ratio=0.3):
    """Run FL experiment"""
    mode_str = f"ATTACK ({int(byzantine_ratio*100)}% Byzantine)" if attack_mode else "CLEAN (No Attack)"
    print(f"\n{'='*60}")
    print(f"Mode: {mode_str}")
    print(f"{'='*60}")
    
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    train_dataset, test_dataset = load_cifar10()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    client_indices = dirichlet_split(train_dataset, CONFIG['num_clients'], CONFIG['dirichlet_alpha'], CONFIG['seed'])
    
    num_byzantine = int(CONFIG['num_clients'] * byzantine_ratio) if attack_mode else 0
    byzantine_clients = list(range(num_byzantine))
    
    results = {}
    
    for method in ['FedAvg', 'FedProx', 'TrimmedMean']:
        print(f"\n  {method}...")
        
        global_model = SimpleCNN().to(device)
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        
        start_time = time.time()
        
        for round_num in range(1, CONFIG['num_rounds'] + 1):
            client_weights, num_samples = [], []
            
            for cid in range(CONFIG['num_clients']):
                # Prepare data
                if attack_mode and cid in byzantine_clients:
                    # Label flip attack
                    attacked_data = [(train_dataset[i][0], (train_dataset[i][1] + np.random.randint(1,10)) % 10) 
                                    for i in client_indices[cid]]
                    loader = torch.utils.data.DataLoader(attacked_data, batch_size=CONFIG['batch_size'], shuffle=True)
                else:
                    client_data = torch.utils.data.Subset(train_dataset, client_indices[cid])
                    loader = torch.utils.data.DataLoader(client_data, batch_size=CONFIG['batch_size'], shuffle=True)
                
                client_model = SimpleCNN().to(device)
                client_model.load_state_dict(global_model.state_dict())
                
                proximal_mu = 0.01 if method == 'FedProx' else 0.0
                weights = local_train(client_model, loader, CONFIG['local_epochs'], 
                                     CONFIG['learning_rate'], device, proximal_mu,
                                     global_weights if method == 'FedProx' else None)
                
                client_weights.append(weights)
                num_samples.append(len(client_indices[cid]))
            
            # Aggregate
            if method in ['FedAvg', 'FedProx']:
                aggregated = fedavg_aggregate(client_weights, num_samples)
            else:
                aggregated = trimmedmean_aggregate(client_weights)
            
            global_model.load_state_dict(aggregated)
            global_weights = {k: v.cpu().clone() for k, v in aggregated.items()}
            
            if round_num % 5 == 0:
                acc = evaluate(global_model, test_loader, device)
                print(f"    Round {round_num}: {acc:.2f}%")
        
        final_acc = evaluate(global_model, test_loader, device)
        elapsed = time.time() - start_time
        results[method] = {'accuracy': final_acc, 'time': elapsed}
        print(f"    FINAL: {final_acc:.2f}% ({elapsed:.1f}s)")
    
    return results

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("FAST CLEAN BASELINE EXPERIMENT")
    print(f"{'='*60}")
    print(f"Config: {CONFIG}")
    
    # Clean baseline
    clean_results = run_experiment(attack_mode=False)
    
    # Under attack
    attacked_results = run_experiment(attack_mode=True, byzantine_ratio=0.3)
    
    # Calculate degradation
    print(f"\n{'='*60}")
    print("BYZANTINE DEGRADATION ANALYSIS")
    print(f"{'='*60}")
    print(f"\n{'Method':<15} {'Clean':>10} {'Attacked':>10} {'Degradation':>12}")
    print("-" * 50)
    
    for method in clean_results.keys():
        clean = clean_results[method]['accuracy']
        attacked = attacked_results[method]['accuracy']
        deg = (clean - attacked) / clean * 100
        print(f"{method:<15} {clean:>9.2f}% {attacked:>9.2f}% {deg:>11.1f}%")
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 
                               f'fast_baseline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    results = {
        'config': CONFIG,
        'clean': clean_results,
        'attacked': attacked_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    
    print(f"\nSaved to: {output_path}")
