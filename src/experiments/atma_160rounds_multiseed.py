"""
ATMA 160 Rounds Multi-Seed Experiment
======================================
Runs ATMA with 160 training rounds across 3 seeds to get confidence intervals
for the main paper claim about ATMA being state-of-the-art.

Author: Experiment Runner
Date: December 28, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import time

# Configuration
SEEDS = [42, 123, 456]
NUM_ROUNDS = 160
NUM_CLIENTS = 20
BYZANTINE_RATIO = 0.2
NUM_BYZANTINE = int(NUM_CLIENTS * BYZANTINE_RATIO)
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.01
DIRICHLET_ALPHA = 0.5
ATTACK_SCALE = -5.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Create results directory
os.makedirs('results', exist_ok=True)

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dirichlet_split(dataset, num_clients, alpha, seed):
    """Create non-IID split using Dirichlet distribution"""
    np.random.seed(seed)
    
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([y for _, y in dataset])
    
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()
        
        start = 0
        for i, split in enumerate(splits):
            client_indices[i].extend(class_indices[start:start + split].tolist())
            start += split
    
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    return client_indices

def get_data_loaders(seed):
    """Get CIFAR-10 data loaders with Dirichlet non-IID split"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    client_indices = create_dirichlet_split(train_dataset, NUM_CLIENTS, DIRICHLET_ALPHA, seed)
    
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)
    
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    return client_loaders, test_loader

def train_client(model, loader, device):
    """Train a single client"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    for _ in range(LOCAL_EPOCHS):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    
    return {name: param.data.clone() for name, param in model.named_parameters()}

def byzantine_attack(model_params, attack_scale):
    """Apply label-flipping style attack with gradient scaling"""
    attacked_params = {}
    for name, param in model_params.items():
        attacked_params[name] = param * attack_scale
    return attacked_params

def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100.0 * correct / total

def aggregate_atma(global_model, client_updates, byzantine_indices, round_num, history):
    """
    ATMA (Adaptive Trimmed Mean Aggregation) with adaptive threshold
    
    Key features:
    - Adaptive threshold based on historical variance
    - Combines trimming with adaptive detection
    - Better for non-IID and adaptive attacks
    """
    # Get all parameter names
    param_names = list(client_updates[0].keys())
    
    # Compute contribution scores for each client
    scores = []
    for i, update in enumerate(client_updates):
        # Flatten all parameters
        flat = torch.cat([update[name].flatten() for name in param_names])
        scores.append(flat)
    
    scores_tensor = torch.stack(scores)
    
    # Compute pairwise distances
    distances = torch.cdist(scores_tensor.unsqueeze(0), scores_tensor.unsqueeze(0)).squeeze(0)
    
    # Sum of distances for each client (like Krum score)
    distance_sums = distances.sum(dim=1)
    
    # Adaptive threshold based on round number and history
    base_trim = int(NUM_CLIENTS * 0.2)  # Base: trim 20%
    
    # Adaptive component: increase trimming if high variance detected
    if len(history['variance']) > 5:
        recent_var = np.mean(history['variance'][-5:])
        overall_var = np.mean(history['variance']) if history['variance'] else recent_var
        
        if recent_var > 1.5 * overall_var:
            # High variance: more aggressive trimming
            adaptive_trim = min(int(NUM_CLIENTS * 0.3), NUM_CLIENTS // 2 - 1)
        else:
            adaptive_trim = base_trim
    else:
        adaptive_trim = base_trim
    
    # Track variance for next round
    current_var = distance_sums.std().item()
    history['variance'].append(current_var)
    
    # Identify clients to keep (lowest distance sums = most central)
    num_keep = NUM_CLIENTS - adaptive_trim
    _, keep_indices = torch.topk(distance_sums, num_keep, largest=False)
    keep_indices = keep_indices.tolist()
    
    # Aggregate selected clients using mean
    aggregated = {}
    for name in param_names:
        stacked = torch.stack([client_updates[i][name] for i in keep_indices])
        aggregated[name] = stacked.mean(dim=0)
    
    # Update global model
    for name, param in global_model.named_parameters():
        param.data = aggregated[name]
    
    return global_model, history

def run_experiment(seed):
    """Run single ATMA experiment with given seed"""
    print(f"\n{'='*60}")
    print(f"Running ATMA 160 rounds with seed {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    
    # Get data
    client_loaders, test_loader = get_data_loaders(seed)
    
    # Initialize global model
    global_model = SimpleCNN().to(DEVICE)
    
    # Select Byzantine clients
    byzantine_indices = set(np.random.choice(NUM_CLIENTS, NUM_BYZANTINE, replace=False))
    print(f"Byzantine clients: {sorted(byzantine_indices)}")
    
    # History for ATMA adaptive mechanism
    history = {'variance': [], 'accuracy': []}
    
    # Track results
    round_accuracies = []
    
    start_time = time.time()
    
    for round_num in range(NUM_ROUNDS):
        # Collect client updates
        client_updates = []
        
        for client_id in range(NUM_CLIENTS):
            # Clone global model for local training
            local_model = SimpleCNN().to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            
            # Train locally
            update = train_client(local_model, client_loaders[client_id], DEVICE)
            
            # Apply Byzantine attack
            if client_id in byzantine_indices:
                update = byzantine_attack(update, ATTACK_SCALE)
            
            client_updates.append(update)
        
        # Aggregate using ATMA
        global_model, history = aggregate_atma(
            global_model, client_updates, byzantine_indices, round_num, history
        )
        
        # Evaluate every 10 rounds
        if (round_num + 1) % 10 == 0:
            accuracy = evaluate(global_model, test_loader, DEVICE)
            round_accuracies.append({'round': round_num + 1, 'accuracy': accuracy})
            history['accuracy'].append(accuracy)
            
            elapsed = time.time() - start_time
            print(f"Round {round_num + 1}/{NUM_ROUNDS}: Accuracy = {accuracy:.2f}%, "
                  f"Time = {elapsed:.1f}s")
    
    # Final evaluation
    final_accuracy = evaluate(global_model, test_loader, DEVICE)
    total_time = time.time() - start_time
    
    print(f"\nFinal Accuracy: {final_accuracy:.2f}%")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return {
        'seed': seed,
        'final_accuracy': final_accuracy,
        'round_accuracies': round_accuracies,
        'total_time': total_time,
        'byzantine_indices': list(byzantine_indices)
    }

def main():
    """Main function to run all experiments"""
    print("="*70)
    print("ATMA 160 Rounds Multi-Seed Experiment")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Seeds: {SEEDS}")
    print(f"  - Rounds: {NUM_ROUNDS}")
    print(f"  - Clients: {NUM_CLIENTS} ({NUM_BYZANTINE} Byzantine)")
    print(f"  - Dirichlet alpha: {DIRICHLET_ALPHA}")
    print(f"  - Attack scale: {ATTACK_SCALE}")
    print(f"  - Device: {DEVICE}")
    print("="*70)
    
    all_results = {
        'experiment': 'ATMA_160rounds_multiseed',
        'config': {
            'seeds': SEEDS,
            'num_rounds': NUM_ROUNDS,
            'num_clients': NUM_CLIENTS,
            'byzantine_ratio': BYZANTINE_RATIO,
            'dirichlet_alpha': DIRICHLET_ALPHA,
            'attack_scale': ATTACK_SCALE,
            'local_epochs': LOCAL_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        },
        'results': {},
        'summary': {}
    }
    
    total_start = time.time()
    
    for seed in SEEDS:
        result = run_experiment(seed)
        all_results['results'][f'seed_{seed}'] = result
    
    # Compute summary statistics
    accuracies = [all_results['results'][f'seed_{s}']['final_accuracy'] for s in SEEDS]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    all_results['summary'] = {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'confidence_interval_95': 1.96 * std_acc / np.sqrt(len(SEEDS)),
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies),
        'all_accuracies': accuracies,
        'total_experiment_time': time.time() - total_start
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/atma_160rounds_multiseed_{timestamp}.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nATMA 160 Rounds Results:")
    for seed in SEEDS:
        acc = all_results['results'][f'seed_{seed}']['final_accuracy']
        print(f"  Seed {seed}: {acc:.2f}%")
    
    print(f"\nSummary Statistics:")
    print(f"  Mean: {mean_acc:.2f}%")
    print(f"  Std:  {std_acc:.2f}%")
    print(f"  95% CI: ±{all_results['summary']['confidence_interval_95']:.2f}%")
    print(f"  Range: [{min(accuracies):.2f}%, {max(accuracies):.2f}%]")
    print(f"\nResults saved to: {filename}")
    
    return all_results

if __name__ == '__main__':
    results = main()
