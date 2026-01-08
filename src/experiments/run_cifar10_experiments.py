"""
CIFAR-10 Byzantine-Robust Federated Learning Experiments
Addresses MW1 from peer review - expanding beyond MNIST to realistic dataset

Critical Review Point:
- Paper only validates on MNIST (oversimplified)
- Need CIFAR-10 with Dirichlet(0.5) non-IID for realistic heterogeneity
- Test all aggregation methods: FedAvg, Krum, TrimmedMean, ATMA
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime
import os

# Windows PowerShell consoles may default to legacy encodings (e.g., cp1252).
# Ensure prints containing non-ASCII characters don't crash the run.
try:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# CIFAR-10 Configuration (matching MNIST experiments)
NUM_CLIENTS = 20
BYZANTINE_RATIO = 0.2  # 20% Byzantine clients
DIRICHLET_ALPHA = 0.5  # Standard non-IID benchmark
# GPU-optimized: Full rounds for paper
ROUNDS = [50, 160]  # Full validation as requested by reviewer
LOCAL_EPOCHS = 5  # Standard FL setting
BATCH_SIZE = 256  # Larger batch for GPU saturation (RTX 5060 Ti)
LEARNING_RATE = 0.1  # INCREASED: 0.01 caused no learning (10% = random guessing)
NUM_WORKERS = 0  # Disable for Windows compatibility (multiprocessing issues)

class SimpleCIFAR10CNN(nn.Module):
    """CNN for CIFAR-10 matching MNIST architecture complexity"""
    def __init__(self):
        super(SimpleCIFAR10CNN, self).__init__()
        # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)  # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)  # 16x16x64
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_non_iid_dirichlet(train_labels: np.ndarray, 
                               num_clients: int, 
                               alpha: float = 0.5) -> List[np.ndarray]:
    """
    Create non-IID data distribution using Dirichlet distribution
    
    Args:
        train_labels: Training labels
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
               0.5 is standard benchmark (FedProx, FedAvg papers)
    
    Returns:
        List of indices for each client
    """
    num_classes = len(np.unique(train_labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    class_indices = [np.where(train_labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        proportions = label_distribution[class_idx]
        split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        
        for client_idx, client_data in enumerate(np.split(indices, split_points)):
            client_indices[client_idx].extend(client_data.tolist())
    
    # Shuffle each client's data
    for indices in client_indices:
        np.random.shuffle(indices)
    
    return client_indices

def compute_distribution_stats(client_indices: List[np.ndarray], 
                                 labels: np.ndarray) -> Dict:
    """Compute statistics about data distribution heterogeneity"""
    num_classes = len(np.unique(labels))
    class_counts = np.zeros((len(client_indices), num_classes))
    
    for client_idx, indices in enumerate(client_indices):
        for label in labels[indices]:
            class_counts[client_idx, int(label)] += 1
    
    # Jensen-Shannon divergence (measure of distribution heterogeneity)
    # Add epsilon to avoid log(0) and division by zero
    eps = 1e-10
    mean_dist = class_counts.mean(axis=0) / class_counts.mean(axis=0).sum()
    mean_dist = mean_dist + eps
    js_divergences = []
    
    for client_dist in class_counts:
        client_dist = client_dist / client_dist.sum()
        client_dist = client_dist + eps
        m = 0.5 * (client_dist + mean_dist)
        kl1 = np.sum(client_dist * np.log(client_dist / m))
        kl2 = np.sum(mean_dist * np.log(mean_dist / m))
        js_divergences.append(0.5 * (kl1 + kl2))
    
    return {
        'mean_js_divergence': np.mean(js_divergences),
        'std_js_divergence': np.std(js_divergences),
        'min_samples_per_client': min(len(indices) for indices in client_indices),
        'max_samples_per_client': max(len(indices) for indices in client_indices),
        'class_distribution': class_counts.tolist()
    }

def local_train(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader,
                device: torch.device,
                epochs: int = 5,
                lr: float = 0.01) -> Dict[str, torch.Tensor]:
    """Train model locally and return state dict"""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def create_byzantine_update(honest_update: Dict[str, torch.Tensor],
                              attack_type: str = 'label_flip',
                              scale_factor: float = -5.0) -> Dict[str, torch.Tensor]:
    """Create Byzantine malicious update"""
    byzantine_update = {}
    
    if attack_type == 'label_flip':
        # Flip gradients with scaling
        for key, param in honest_update.items():
            byzantine_update[key] = param * scale_factor
    elif attack_type == 'gaussian_noise':
        # Add large Gaussian noise
        for key, param in honest_update.items():
            noise = torch.randn_like(param) * param.std() * 10
            byzantine_update[key] = param + noise
    
    return byzantine_update

def aggregate_fedavg(updates: List[Dict[str, torch.Tensor]],
                     weights: List[float]) -> Dict[str, torch.Tensor]:
    """Standard FedAvg aggregation"""
    aggregated = {}
    total_weight = sum(weights)
    
    for key in updates[0].keys():
        aggregated[key] = sum(
            w * update[key] for w, update in zip(weights, updates)
        ) / total_weight
    
    return aggregated

def aggregate_krum(updates: List[Dict[str, torch.Tensor]],
                   weights: List[float],
                   num_byzantine: int = 4) -> Dict[str, torch.Tensor]:
    """Krum aggregation - select most representative update"""
    n = len(updates)
    f = num_byzantine
    
    # Flatten updates for distance computation
    flattened = []
    for update in updates:
        flat = torch.cat([param.flatten() for param in update.values()])
        flattened.append(flat)
    
    # Compute pairwise distances
    scores = []
    for i, ui in enumerate(flattened):
        distances = [(torch.norm(ui - uj).item(), j) for j, uj in enumerate(flattened) if j != i]
        distances.sort()
        # Sum distances to n-f-2 closest neighbors
        score = sum(d for d, _ in distances[:n-f-2])
        scores.append((score, i))
    
    # Select update with minimum score
    _, selected_idx = min(scores)
    return updates[selected_idx]

def aggregate_trimmed_mean(updates: List[Dict[str, torch.Tensor]],
                            weights: List[float],
                            trim_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
    """TrimmedMean - coordinate-wise trimming"""
    aggregated = {}
    
    for key in updates[0].keys():
        stacked = torch.stack([update[key] for update in updates])
        
        # Sort along client dimension
        sorted_params, _ = torch.sort(stacked, dim=0)
        
        # Trim top and bottom
        n = len(updates)
        trim_count = int(n * trim_ratio)
        if trim_count > 0:
            trimmed = sorted_params[trim_count:-trim_count]
        else:
            trimmed = sorted_params
        
        # Weighted mean of remaining
        aggregated[key] = trimmed.mean(dim=0)
    
    return aggregated

def aggregate_atma(updates: List[Dict[str, torch.Tensor]],
                   weights: List[float],
                   threshold: float = 0.2,
                   adaptation_rate: float = 0.1) -> Tuple[Dict[str, torch.Tensor], float]:
    """ATMA - Adaptive Trimmed Mean Aggregation with dynamic threshold"""
    aggregated = {}
    outlier_counts = []
    
    for key in updates[0].keys():
        stacked = torch.stack([update[key] for update in updates])
        median = torch.median(stacked, dim=0)[0]
        mad = torch.median(torch.abs(stacked - median), dim=0)[0]
        
        # Adaptive threshold based on MAD
        threshold_adapted = threshold * (1 + mad.mean().item())
        
        # Filter outliers
        distances = torch.abs(stacked - median)
        mask = distances <= threshold_adapted * mad
        
        # Count outliers for threshold adaptation
        outlier_counts.append((~mask.any(dim=1)).sum().item())
        
        # Aggregate non-outliers
        filtered = stacked * mask.float()
        counts = mask.sum(dim=0).float().clamp(min=1)
        aggregated[key] = filtered.sum(dim=0) / counts
    
    # Adapt threshold for next round
    outlier_ratio = sum(outlier_counts) / (len(outlier_counts) * len(updates))
    new_threshold = threshold + adaptation_rate * (outlier_ratio - 0.2)
    new_threshold = np.clip(new_threshold, 0.1, 0.3)
    
    return aggregated, new_threshold

def evaluate_model(model: nn.Module,
                   test_loader: torch.utils.data.DataLoader,
                   device: torch.device) -> float:
    """Evaluate model accuracy on test set"""
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

def run_experiment(aggregation_method: str,
                   num_rounds: int,
                   client_indices: List[np.ndarray],
                   train_dataset,
                   test_loader,
                   device: torch.device) -> Dict:
    """Run single experiment configuration"""
    
    print(f"\n{'='*60}")
    print(f"Experiment: {aggregation_method} - {num_rounds} rounds")
    print(f"{'='*60}")
    
    # Initialize global model
    global_model = SimpleCIFAR10CNN().to(device)
    
    # Track metrics
    accuracies = []
    atma_threshold = 0.2  # Initial threshold for ATMA
    
    num_byzantine = int(NUM_CLIENTS * BYZANTINE_RATIO)
    byzantine_clients = list(range(num_byzantine))
    
    for round_num in range(num_rounds):
        print(f"Round {round_num+1}/{num_rounds}", end=' ')
        
        # Client updates
        updates = []
        weights = []
        
        for client_idx in range(NUM_CLIENTS):
            # Create client dataloader
            client_data = torch.utils.data.Subset(train_dataset, client_indices[client_idx])
            client_loader = torch.utils.data.DataLoader(
                client_data, batch_size=BATCH_SIZE, shuffle=True,
                pin_memory=True  # Fast CPU→GPU transfer
            )
            
            # Train locally
            client_model = SimpleCIFAR10CNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            update = local_train(client_model, client_loader, device, 
                                LOCAL_EPOCHS, LEARNING_RATE)
            
            # Byzantine attack
            if client_idx in byzantine_clients:
                update = create_byzantine_update(update, 'label_flip', scale_factor=-5.0)
            
            updates.append(update)
            weights.append(len(client_indices[client_idx]))
        
        # Aggregate
        if aggregation_method == 'FedAvg':
            aggregated = aggregate_fedavg(updates, weights)
        elif aggregation_method == 'Krum':
            aggregated = aggregate_krum(updates, weights, num_byzantine)
        elif aggregation_method == 'TrimmedMean':
            aggregated = aggregate_trimmed_mean(updates, trim_ratio=0.2, weights=weights)
        elif aggregation_method == 'ATMA':
            aggregated, atma_threshold = aggregate_atma(
                updates, weights, threshold=atma_threshold, adaptation_rate=0.1
            )
        
        # Update global model
        global_model.load_state_dict(aggregated)
        
        # Evaluate every 10 rounds
        if (round_num + 1) % 10 == 0:
            accuracy = evaluate_model(global_model, test_loader, device)
            accuracies.append({
                'round': round_num + 1,
                'accuracy': accuracy,
                'atma_threshold': atma_threshold if aggregation_method == 'ATMA' else None
            })
            print(f"Accuracy: {accuracy:.2f}%", end='')
            if aggregation_method == 'ATMA':
                print(f" (threshold: {atma_threshold:.3f})", end='')
        
        print()
    
    # Final evaluation
    final_accuracy = evaluate_model(global_model, test_loader, device)
    
    return {
        'aggregation_method': aggregation_method,
        'num_rounds': num_rounds,
        'final_accuracy': final_accuracy,
        'accuracies_per_round': accuracies,
        'byzantine_ratio': BYZANTINE_RATIO,
        'num_clients': NUM_CLIENTS
    }

def main():
    """Run all CIFAR-10 experiments"""
    
    print("="*70)
    print("CIFAR-10 Byzantine-Robust Federated Learning Experiments")
    print("Addressing Reviewer MW1: Expanding beyond MNIST")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512, shuffle=False,
        pin_memory=True  # Fast CPU→GPU transfer
    )
    
    # Create non-IID distribution
    print(f"\nCreating non-IID distribution (Dirichlet alpha={DIRICHLET_ALPHA})...")
    train_labels = np.array([label for _, label in train_dataset])
    client_indices = create_non_iid_dirichlet(train_labels, NUM_CLIENTS, DIRICHLET_ALPHA)
    
    # Compute distribution statistics
    dist_stats = compute_distribution_stats(client_indices, train_labels)
    print(f"Distribution heterogeneity (JS divergence): {dist_stats['mean_js_divergence']:.4f} ± {dist_stats['std_js_divergence']:.4f}")
    print(f"Samples per client: {dist_stats['min_samples_per_client']} - {dist_stats['max_samples_per_client']}")
    
    # Run experiments
    aggregation_methods = ['FedAvg', 'Krum', 'TrimmedMean', 'ATMA']
    results = []
    
    for method in aggregation_methods:
        for num_rounds in ROUNDS:
            print(f"\n{'='*60}")
            print(f"Starting: {method} - {num_rounds} rounds")
            print(f"{'='*60}")
            result = run_experiment(
                method, num_rounds, client_indices, 
                train_dataset, test_loader, device
            )
            results.append(result)
            
            print(f"\n✓ {method} {num_rounds}r: {result['final_accuracy']:.2f}%")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_file = f'results/cifar10_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    output_data = {
        'experiment_config': {
            'dataset': 'CIFAR-10',
            'num_clients': NUM_CLIENTS,
            'byzantine_ratio': BYZANTINE_RATIO,
            'dirichlet_alpha': DIRICHLET_ALPHA,
            'local_epochs': LOCAL_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'seed': SEED
        },
        'distribution_stats': dist_stats,
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")
    
    # Summary table
    print("\nSummary:")
    print(f"{'Method':<15} {'50 rounds':<12} {'160 rounds':<12}")
    print("-" * 40)
    
    for method in aggregation_methods:
        r50 = next((r['final_accuracy'] for r in results 
                   if r['aggregation_method'] == method and r['num_rounds'] == 50), None)
        r160 = next((r['final_accuracy'] for r in results 
                    if r['aggregation_method'] == method and r['num_rounds'] == 160), None)
        print(f"{method:<15} {r50:>10.2f}%  {r160:>10.2f}%")

if __name__ == '__main__':
    main()
