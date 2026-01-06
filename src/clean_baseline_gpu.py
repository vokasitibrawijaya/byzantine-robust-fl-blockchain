"""
Clean Baseline Experiments for Byzantine Degradation %
======================================================
Run FL methods WITHOUT attacks to get clean baseline accuracy.
This enables accurate calculation of: (clean_acc - attacked_acc) / clean_acc

Methods: FedAvg, FedProx, FedDyn, TrimmedMean, ATMA
Dataset: CIFAR-10 with Dirichlet(alpha=0.5) non-IID
GPU: RTX 5060 Ti with PyTorch 2.9.1+cu128
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

# ===================== CONFIGURATION =====================
CONFIG = {
    'num_clients': 20,
    'num_rounds': 50,
    'local_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.01,
    'dirichlet_alpha': 0.5,
    'byzantine_ratio': 0.0,  # NO ATTACKS for clean baseline
    'seed': 42,
}

# ===================== CNN MODEL =====================
class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ===================== DATA LOADING =====================
def load_cifar10():
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def dirichlet_split(dataset, num_clients, alpha, seed):
    """Split dataset using Dirichlet distribution for non-IID"""
    np.random.seed(seed)
    
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = 10
    
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_batch = np.split(idx_k, proportions)
        for i, idx in enumerate(idx_batch):
            client_indices[i].extend(idx.tolist())
    
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    return client_indices

# ===================== FL TRAINING =====================
def local_train(model, data_loader, epochs, lr, device, proximal_mu=0.0, global_weights=None):
    """Local training on client"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # FedProx proximal term
            if proximal_mu > 0 and global_weights is not None:
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    proximal_term += ((param - global_weights[name].to(device)) ** 2).sum()
                loss += (proximal_mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
    
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

# ===================== AGGREGATION METHODS =====================
def fedavg_aggregate(client_weights, num_samples):
    """FedAvg: weighted average by sample count"""
    total_samples = sum(num_samples)
    avg_weights = {}
    
    for key in client_weights[0].keys():
        avg_weights[key] = sum(
            w[key].float() * (n / total_samples) 
            for w, n in zip(client_weights, num_samples)
        )
    
    return avg_weights

def trimmedmean_aggregate(client_weights, trim_ratio=0.2):
    """TrimmedMean: trim top/bottom 20% before averaging"""
    num_trim = max(1, int(len(client_weights) * trim_ratio))
    avg_weights = {}
    
    for key in client_weights[0].keys():
        stacked = torch.stack([w[key].float() for w in client_weights])
        sorted_weights, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_weights[num_trim:-num_trim]
        avg_weights[key] = trimmed.mean(dim=0)
    
    return avg_weights

def krum_aggregate(client_weights, num_byzantine=0):
    """Krum: select weight closest to others"""
    n = len(client_weights)
    f = num_byzantine
    
    # Flatten weights
    flat_weights = []
    for w in client_weights:
        flat = torch.cat([v.float().flatten() for v in w.values()])
        flat_weights.append(flat)
    flat_weights = torch.stack(flat_weights)
    
    # Compute pairwise distances
    distances = torch.cdist(flat_weights.unsqueeze(0), flat_weights.unsqueeze(0)).squeeze()
    
    # Krum score: sum of n-f-2 smallest distances
    scores = []
    for i in range(n):
        dists = distances[i].clone()
        dists[i] = float('inf')  # Exclude self
        k = n - f - 2
        if k > 0:
            smallest = torch.topk(dists, k, largest=False).values
            scores.append(smallest.sum().item())
        else:
            scores.append(0)
    
    best_idx = np.argmin(scores)
    return client_weights[best_idx]

# ===================== MAIN EXPERIMENT =====================
def run_clean_baseline():
    """Run clean baseline experiments (no attacks)"""
    print("\n" + "="*60)
    print("CLEAN BASELINE EXPERIMENTS (No Byzantine Attacks)")
    print("="*60)
    
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_dataset, test_dataset = load_cifar10()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Split data among clients
    print(f"Splitting data among {CONFIG['num_clients']} clients (Dirichlet α={CONFIG['dirichlet_alpha']})...")
    client_indices = dirichlet_split(train_dataset, CONFIG['num_clients'], CONFIG['dirichlet_alpha'], CONFIG['seed'])
    
    # Methods to test
    methods = ['FedAvg', 'FedProx', 'TrimmedMean', 'Krum']
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running {method} (Clean Baseline)...")
        print(f"{'='*50}")
        
        # Initialize global model
        global_model = SimpleCNN().to(device)
        
        # Store initial weights for FedProx
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        
        round_accuracies = []
        start_time = time.time()
        
        for round_num in range(1, CONFIG['num_rounds'] + 1):
            # Collect client updates
            client_weights = []
            num_samples = []
            
            for client_id in range(CONFIG['num_clients']):
                # Create client data loader
                client_data = torch.utils.data.Subset(train_dataset, client_indices[client_id])
                client_loader = torch.utils.data.DataLoader(
                    client_data, batch_size=CONFIG['batch_size'], shuffle=True
                )
                
                # Copy global model to client
                client_model = SimpleCNN().to(device)
                client_model.load_state_dict(global_model.state_dict())
                
                # Local training
                proximal_mu = 0.01 if method == 'FedProx' else 0.0
                weights = local_train(
                    client_model, client_loader, 
                    CONFIG['local_epochs'], CONFIG['learning_rate'],
                    device, proximal_mu, global_weights if method == 'FedProx' else None
                )
                
                client_weights.append(weights)
                num_samples.append(len(client_indices[client_id]))
            
            # Aggregate based on method
            if method == 'FedAvg' or method == 'FedProx':
                aggregated = fedavg_aggregate(client_weights, num_samples)
            elif method == 'TrimmedMean':
                aggregated = trimmedmean_aggregate(client_weights)
            elif method == 'Krum':
                aggregated = krum_aggregate(client_weights)
            
            # Update global model
            global_model.load_state_dict(aggregated)
            global_weights = {k: v.cpu().clone() for k, v in aggregated.items()}
            
            # Evaluate
            accuracy = evaluate(global_model, test_loader, device)
            round_accuracies.append(accuracy)
            
            if round_num % 10 == 0 or round_num == 1:
                elapsed = time.time() - start_time
                print(f"  Round {round_num:3d}/{CONFIG['num_rounds']}: Accuracy = {accuracy:.2f}%  ({elapsed:.1f}s)")
        
        total_time = time.time() - start_time
        final_accuracy = round_accuracies[-1]
        
        results[method] = {
            'final_accuracy': final_accuracy,
            'round_accuracies': round_accuracies,
            'training_time': total_time
        }
        
        print(f"\n  {method} Final: {final_accuracy:.2f}% (Time: {total_time:.1f}s)")
    
    return results

def run_attacked_baseline():
    """Run experiments WITH Byzantine attacks for comparison"""
    print("\n" + "="*60)
    print("ATTACKED EXPERIMENTS (30% Byzantine - Label Flip)")
    print("="*60)
    
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Load data
    train_dataset, test_dataset = load_cifar10()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    client_indices = dirichlet_split(train_dataset, CONFIG['num_clients'], CONFIG['dirichlet_alpha'], CONFIG['seed'])
    
    byzantine_ratio = 0.3
    num_byzantine = int(CONFIG['num_clients'] * byzantine_ratio)
    byzantine_clients = list(range(num_byzantine))  # First clients are Byzantine
    
    methods = ['FedAvg', 'FedProx', 'TrimmedMean', 'Krum']
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running {method} (Under Attack)...")
        print(f"{'='*50}")
        
        global_model = SimpleCNN().to(device)
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        
        round_accuracies = []
        start_time = time.time()
        
        for round_num in range(1, CONFIG['num_rounds'] + 1):
            client_weights = []
            num_samples = []
            
            for client_id in range(CONFIG['num_clients']):
                client_data = torch.utils.data.Subset(train_dataset, client_indices[client_id])
                
                # Byzantine attack: flip labels
                if client_id in byzantine_clients:
                    # Create attacked dataset with flipped labels
                    attacked_data = []
                    for idx in client_indices[client_id]:
                        img, label = train_dataset[idx]
                        # Label flip attack: random wrong label
                        wrong_label = (label + np.random.randint(1, 10)) % 10
                        attacked_data.append((img, wrong_label))
                    client_loader = torch.utils.data.DataLoader(
                        attacked_data, batch_size=CONFIG['batch_size'], shuffle=True
                    )
                else:
                    client_loader = torch.utils.data.DataLoader(
                        client_data, batch_size=CONFIG['batch_size'], shuffle=True
                    )
                
                client_model = SimpleCNN().to(device)
                client_model.load_state_dict(global_model.state_dict())
                
                proximal_mu = 0.01 if method == 'FedProx' else 0.0
                weights = local_train(
                    client_model, client_loader, 
                    CONFIG['local_epochs'], CONFIG['learning_rate'],
                    device, proximal_mu, global_weights if method == 'FedProx' else None
                )
                
                client_weights.append(weights)
                num_samples.append(len(client_indices[client_id]))
            
            # Aggregate
            if method == 'FedAvg' or method == 'FedProx':
                aggregated = fedavg_aggregate(client_weights, num_samples)
            elif method == 'TrimmedMean':
                aggregated = trimmedmean_aggregate(client_weights)
            elif method == 'Krum':
                aggregated = krum_aggregate(client_weights, num_byzantine)
            
            global_model.load_state_dict(aggregated)
            global_weights = {k: v.cpu().clone() for k, v in aggregated.items()}
            
            accuracy = evaluate(global_model, test_loader, device)
            round_accuracies.append(accuracy)
            
            if round_num % 10 == 0 or round_num == 1:
                elapsed = time.time() - start_time
                print(f"  Round {round_num:3d}/{CONFIG['num_rounds']}: Accuracy = {accuracy:.2f}%  ({elapsed:.1f}s)")
        
        total_time = time.time() - start_time
        final_accuracy = round_accuracies[-1]
        
        results[method] = {
            'final_accuracy': final_accuracy,
            'round_accuracies': round_accuracies,
            'training_time': total_time
        }
        
        print(f"\n  {method} Final: {final_accuracy:.2f}% (Time: {total_time:.1f}s)")
    
    return results

def calculate_degradation(clean_results, attacked_results):
    """Calculate Byzantine degradation percentage"""
    print("\n" + "="*60)
    print("BYZANTINE DEGRADATION ANALYSIS")
    print("="*60)
    print(f"\n{'Method':<15} {'Clean':>10} {'Attacked':>10} {'Degradation':>12}")
    print("-" * 50)
    
    degradation = {}
    for method in clean_results.keys():
        clean_acc = clean_results[method]['final_accuracy']
        attacked_acc = attacked_results[method]['final_accuracy']
        deg = (clean_acc - attacked_acc) / clean_acc * 100
        degradation[method] = {
            'clean_accuracy': clean_acc,
            'attacked_accuracy': attacked_acc,
            'degradation_percent': deg
        }
        print(f"{method:<15} {clean_acc:>9.2f}% {attacked_acc:>9.2f}% {deg:>11.1f}%")
    
    return degradation

if __name__ == '__main__':
    print(f"\nExperiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {CONFIG}")
    
    # Run clean baseline
    clean_results = run_clean_baseline()
    
    # Run attacked experiments
    attacked_results = run_attacked_baseline()
    
    # Calculate degradation
    degradation = calculate_degradation(clean_results, attacked_results)
    
    # Save results
    results = {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'clean_baseline': clean_results,
        'under_attack': attacked_results,
        'degradation': degradation
    }
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', f'clean_baseline_gpu_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    # Convert numpy/tensor to serializable
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    # Print LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE FOR PAPER")
    print("="*60)
    print("""
\\begin{table}[h]
\\caption{Byzantine Degradation Percentage (CIFAR-10, 50 Rounds, 30\\% Byzantine)}
\\label{tab:degradation}
\\centering
\\begin{tabular}{lccc}
\\toprule
\\textbf{Method} & \\textbf{Clean Baseline} & \\textbf{Under Attack} & \\textbf{Degradation} \\\\
\\midrule""")
    
    for method in ['FedAvg', 'FedProx', 'TrimmedMean', 'Krum']:
        d = degradation[method]
        print(f"{method} & {d['clean_accuracy']:.2f}\\% & {d['attacked_accuracy']:.2f}\\% & {d['degradation_percent']:.1f}\\% \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\end{table}""")
    
    print(f"\n\nTotal experiment time: Complete!")
