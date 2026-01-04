"""
Real Federated Learning Training on MNIST (FIXED VERSION)
==========================================================
Uses ACTUAL MNIST dataset from torchvision - NOT synthetic data.
This is the scientifically valid version for IEEE submission.

Demonstrates ATMA-FT with actual model training and convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from datetime import datetime


class SimpleMNISTNet(nn.Module):
    """Lightweight CNN for MNIST classification"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_mnist_data():
    """Load ACTUAL MNIST dataset from torchvision"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset


def partition_data_non_iid(dataset, n_clients: int, alpha: float = 0.5, seed: int = 42):
    """
    Partition dataset using Dirichlet distribution for non-IID data.
    
    Args:
        dataset: PyTorch dataset
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    n_classes = len(np.unique(labels))
    
    # Group indices by label
    label_indices = {i: np.where(labels == i)[0] for i in range(n_classes)}
    
    # Dirichlet distribution for each class
    client_indices = [[] for _ in range(n_clients)]
    
    for c in range(n_classes):
        idx = label_indices[c]
        np.random.shuffle(idx)
        
        # Dirichlet proportions
        proportions = np.random.dirichlet([alpha] * n_clients)
        proportions = proportions / proportions.sum()
        
        # Split indices according to proportions
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)
        split_indices = np.split(idx, proportions[:-1])
        
        for client_id, indices in enumerate(split_indices):
            client_indices[client_id].extend(indices.tolist())
    
    return client_indices


def byzantine_gradient_attack(gradients: List[torch.Tensor], 
                              attack_type: str = 'sign_flip') -> List[torch.Tensor]:
    """
    Apply Byzantine attack to gradients.
    
    Args:
        gradients: List of gradient tensors
        attack_type: 'sign_flip', 'noise', or 'zero'
    """
    if attack_type == 'sign_flip':
        return [-g for g in gradients]
    elif attack_type == 'noise':
        return [g + torch.randn_like(g) * g.std() * 5 for g in gradients]
    elif attack_type == 'zero':
        return [torch.zeros_like(g) for g in gradients]
    return gradients


def aggregate_with_atma(worker_gradients: List[List[torch.Tensor]],
                        worker_ids: List[int],
                        byzantine_ids: set,
                        aggregation_method: str = 'median') -> List[torch.Tensor]:
    """
    Aggregate gradients using ATMA-style robust method.
    
    Args:
        worker_gradients: List of gradient lists (one per worker)
        worker_ids: Worker identifiers
        byzantine_ids: Set of Byzantine worker IDs
        aggregation_method: 'mean', 'median', 'trimmed_mean', 'krum'
    """
    
    if aggregation_method == 'median':
        # Coordinate-wise median (Byzantine-robust)
        aggregated = []
        for param_idx in range(len(worker_gradients[0])):
            stacked = torch.stack([w[param_idx] for w in worker_gradients])
            median_grad = torch.median(stacked, dim=0).values
            aggregated.append(median_grad)
        return aggregated
    
    elif aggregation_method == 'trimmed_mean':
        # Trimmed mean: remove 20% highest and lowest values
        aggregated = []
        trim_ratio = 0.2
        for param_idx in range(len(worker_gradients[0])):
            stacked = torch.stack([w[param_idx] for w in worker_gradients])
            
            # Sort along worker dimension
            sorted_grads, _ = torch.sort(stacked, dim=0)
            
            # Trim
            n_workers = len(worker_gradients)
            trim_count = int(n_workers * trim_ratio)
            if trim_count > 0:
                trimmed = sorted_grads[trim_count:-trim_count]
            else:
                trimmed = sorted_grads
            
            trimmed_mean = trimmed.mean(dim=0)
            aggregated.append(trimmed_mean)
        return aggregated
    
    else:  # 'mean' - standard FedAvg (not robust)
        aggregated = []
        for param_idx in range(len(worker_gradients[0])):
            stacked = torch.stack([w[param_idx] for w in worker_gradients])
            mean_grad = stacked.mean(dim=0)
            aggregated.append(mean_grad)
        return aggregated


def run_federated_training(n_workers: int = 20,
                           byzantine_ratio: float = 0.2,
                           n_rounds: int = 20,
                           local_epochs: int = 1,
                           aggregation_method: str = 'median',
                           seed: int = 42) -> Dict:
    """
    Run federated learning with Byzantine attacks using REAL MNIST data.
    
    Args:
        n_workers: Total number of workers
        byzantine_ratio: Fraction of Byzantine workers
        n_rounds: Number of FL rounds
        local_epochs: Local training epochs per round
        aggregation_method: 'mean', 'median', 'trimmed_mean'
        seed: Random seed
    
    Returns:
        Training results dictionary
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"REAL FL TRAINING - ACTUAL MNIST (FIXED VERSION)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Workers: {n_workers}, Byzantine: {byzantine_ratio:.0%}")
    print(f"Rounds: {n_rounds}, Aggregation: {aggregation_method}")
    
    # Load REAL MNIST data
    print("\n[Loading ACTUAL MNIST dataset...]")
    train_dataset, test_dataset = load_mnist_data()
    print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Partition data (non-IID with Dirichlet)
    print("[Partitioning data (Dirichlet α=0.5 non-IID)...]")
    client_indices = partition_data_non_iid(train_dataset, n_workers, alpha=0.5, seed=seed)
    
    # Create client dataloaders
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        client_loaders.append(loader)
        
    # Test dataloader
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize model
    global_model = SimpleMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Select Byzantine workers
    n_byzantine = int(n_workers * byzantine_ratio)
    byzantine_ids = set(np.random.choice(n_workers, n_byzantine, replace=False))
    print(f"[Byzantine workers: {sorted(byzantine_ids)}]")
    
    history = {
        'rounds': [],
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    for round_num in range(1, n_rounds + 1):
        print(f"\n[Round {round_num}/{n_rounds}]")
        
        # Collect gradients from all workers
        worker_gradients = []
        worker_losses = []
        
        for worker_id in range(n_workers):
            # Create local model copy
            local_model = SimpleMNISTNet().to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
            
            # Local training on REAL MNIST data
            local_model.train()
            total_loss = 0
            n_batches = 0
            
            for epoch in range(local_epochs):
                for images, labels in client_loaders[worker_id]:
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
                    
                    if n_batches >= 10:  # Limit batches per round for speed
                        break
                if n_batches >= 10:
                    break
            
            # Compute gradient as difference from global model
            gradients = []
            for (name, global_param), (_, local_param) in zip(
                global_model.named_parameters(), local_model.named_parameters()
            ):
                grad = global_param.data - local_param.data  # Gradient = old - new (for update)
                gradients.append(grad)
            
            # Byzantine attack
            if worker_id in byzantine_ids:
                gradients = byzantine_gradient_attack(gradients, attack_type='sign_flip')
            
            worker_gradients.append(gradients)
            worker_losses.append(total_loss / max(n_batches, 1))
        
        # Aggregate with ATMA
        aggregated_gradients = aggregate_with_atma(
            worker_gradients,
            list(range(n_workers)),
            byzantine_ids,
            aggregation_method=aggregation_method
        )
        
        # Update global model
        with torch.no_grad():
            for param, agg_grad in zip(global_model.parameters(), aggregated_gradients):
                param.data -= agg_grad  # Apply update (gradient points towards new value)
        
        # Evaluate on test set
        global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = correct / total
        test_loss = test_loss / len(test_loader)
        
        # Record metrics
        history['rounds'].append(round_num)
        history['train_loss'].append(float(np.mean(worker_losses)))
        history['test_loss'].append(float(test_loss))
        history['test_accuracy'].append(float(test_accuracy))
        
        print(f"   Train Loss: {np.mean(worker_losses):.4f}")
        print(f"   Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2%}")
    
    return {
        'n_workers': n_workers,
        'byzantine_ratio': byzantine_ratio,
        'n_rounds': n_rounds,
        'aggregation_method': aggregation_method,
        'device': device,
        'final_test_accuracy': history['test_accuracy'][-1],
        'final_test_loss': history['test_loss'][-1],
        'history': history,
        'converged': history['test_accuracy'][-1] > 0.5,  # Should exceed 50% with real MNIST
        'dataset': 'MNIST (torchvision.datasets.MNIST)',
        'data_type': 'REAL'  # Mark as real data
    }


def run_comparative_experiment(seeds: List[int] = [42, 43, 44], n_rounds: int = 20) -> Dict:
    """
    Compare ATMA aggregation methods across multiple seeds.
    """
    print("=" * 80)
    print("REAL FEDERATED LEARNING - ACTUAL MNIST (FIXED VERSION)")
    print("=" * 80)
    print("This version uses REAL MNIST from torchvision, NOT synthetic data!")
    
    methods = ['mean', 'median', 'trimmed_mean']
    results = {method: [] for method in methods}
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[Seed {seed}]")
        print(f"{'='*60}")
        
        for method in methods:
            print(f"\n  Testing {method.upper()}")
            result = run_federated_training(
                n_workers=20,
                byzantine_ratio=0.2,
                n_rounds=n_rounds,
                aggregation_method=method,
                seed=seed
            )
            results[method].append(result)
    
    # Compute summary statistics
    summary = {}
    for method in methods:
        accuracies = [r['final_test_accuracy'] for r in results[method]]
        losses = [r['final_test_loss'] for r in results[method]]
        summary[method] = {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies, ddof=1)),
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses, ddof=1)),
            'convergence_rate': sum([r['converged'] for r in results[method]]) / len(results[method])
        }
    
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS SEEDS (REAL MNIST)")
    print("=" * 80)
    for method, stats in summary.items():
        print(f"\n{method.upper()}:")
        print(f"  Accuracy: {stats['mean_accuracy']:.2%} ± {stats['std_accuracy']:.2%}")
        print(f"  Loss: {stats['mean_loss']:.4f} ± {stats['std_loss']:.4f}")
        print(f"  Convergence: {stats['convergence_rate']:.0%}")
    
    output = {
        'seeds': seeds,
        'methods': methods,
        'dataset': 'MNIST (torchvision.datasets.MNIST)',
        'data_type': 'REAL',  # Mark as real data
        'n_rounds': n_rounds,
        'detailed_results': results,
        'summary': summary,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'real_fl_training_FIXED.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n[OK] Results saved to: {output_file}")
    print("=" * 80)
    
    return output


if __name__ == "__main__":
    # Run with actual MNIST data
    run_comparative_experiment(seeds=[42, 43, 44], n_rounds=20)
