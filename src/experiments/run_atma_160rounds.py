"""
ATMA 160 Rounds Extended Evaluation
Addresses MW3 from peer review - fair comparison with TrimmedMean

Critical Review Point:
- Paper only tests ATMA at 50 rounds but TrimmedMean at 160 rounds
- This creates unfair comparison (TrimmedMean 93.45% vs ATMA 85.12%)
- Need to evaluate ATMA with extended training to assess true capability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Configuration matching paper
NUM_CLIENTS = 20
BYZANTINE_RATIO = 0.2
NUM_ROUNDS = 160  # Extended to match TrimmedMean
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# ATMA parameters
INITIAL_THRESHOLD = 0.15
ADAPTATION_RATE = 0.1
MIN_THRESHOLD = 0.10
MAX_THRESHOLD = 0.30

class SimpleCNN(nn.Module):
    """Matching paper's CNN architecture"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_non_iid_split(train_dataset, num_clients: int, 
                         shards_per_client: int = 2):
    """Create non-IID data split matching paper methodology"""
    num_shards = num_clients * shards_per_client
    shard_size = len(train_dataset) // num_shards
    
    # Sort by labels
    indices = np.arange(len(train_dataset))
    labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    sorted_indices = indices[np.argsort(labels)]
    
    # Create shards
    shards = [sorted_indices[i*shard_size:(i+1)*shard_size] for i in range(num_shards)]
    np.random.shuffle(shards)
    
    # Assign shards to clients
    client_indices = []
    for i in range(num_clients):
        client_data = np.concatenate(shards[i*shards_per_client:(i+1)*shards_per_client])
        np.random.shuffle(client_data)
        client_indices.append(client_data)
    
    return client_indices

def local_train(model: nn.Module, train_loader, device, epochs=5, lr=0.01):
    """Train model locally"""
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
                            scale_factor: float = -5.0) -> Dict[str, torch.Tensor]:
    """Create Byzantine malicious update (label flip attack)"""
    byzantine_update = {}
    for key, param in honest_update.items():
        byzantine_update[key] = param * scale_factor
    return byzantine_update

def aggregate_atma(updates: List[Dict[str, torch.Tensor]],
                   weights: List[float],
                   threshold: float,
                   adaptation_rate: float) -> Tuple[Dict[str, torch.Tensor], float, Dict]:
    """
    ATMA - Adaptive Trimmed Mean Aggregation
    Returns: (aggregated_model, new_threshold, stats)
    """
    aggregated = {}
    total_outliers = 0
    total_params = 0
    param_stats = []
    
    for key in updates[0].keys():
        # Stack all client updates for this parameter
        stacked = torch.stack([update[key] for update in updates])
        
        # Compute median and MAD for robust statistics
        median = torch.median(stacked, dim=0)[0]
        mad = torch.median(torch.abs(stacked - median), dim=0)[0]
        
        # Adaptive threshold scaled by MAD
        threshold_adapted = threshold * (1 + mad.mean().item())
        
        # Identify outliers
        distances = torch.abs(stacked - median)
        mask = distances <= threshold_adapted * mad.clamp(min=1e-6)
        
        outlier_count = (~mask.any(dim=1)).sum().item()
        total_outliers += outlier_count
        total_params += len(updates)
        
        # Aggregate non-outliers with weights
        filtered = stacked * mask.float()
        counts = mask.sum(dim=0).float().clamp(min=1)
        aggregated[key] = filtered.sum(dim=0) / counts
        
        param_stats.append({
            'param_name': key,
            'outliers_detected': outlier_count,
            'median_mad': mad.mean().item(),
            'threshold_used': threshold_adapted
        })
    
    # Adapt threshold for next round
    outlier_ratio = total_outliers / total_params if total_params > 0 else 0
    
    # Target outlier ratio is Byzantine ratio (0.2)
    # Increase threshold if too many outliers, decrease if too few
    threshold_adjustment = adaptation_rate * (outlier_ratio - BYZANTINE_RATIO)
    new_threshold = threshold - threshold_adjustment  # Subtract because higher outlier ratio = need lower threshold
    new_threshold = np.clip(new_threshold, MIN_THRESHOLD, MAX_THRESHOLD)
    
    stats = {
        'outlier_ratio': outlier_ratio,
        'threshold_adjustment': threshold_adjustment,
        'param_details': param_stats[:3]  # Sample for logging
    }
    
    return aggregated, new_threshold, stats

def evaluate_model(model: nn.Module, test_loader, device) -> float:
    """Evaluate model accuracy"""
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

def run_atma_160rounds():
    """Run ATMA for 160 rounds to compare with TrimmedMean"""
    
    print("="*70)
    print("ATMA 160 Rounds Extended Evaluation")
    print("Addressing Reviewer MW3: Fair comparison with TrimmedMean")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    
    # Create non-IID split
    print("\nCreating non-IID data distribution...")
    client_indices = create_non_iid_split(train_dataset, NUM_CLIENTS, shards_per_client=2)
    
    # Initialize global model
    global_model = SimpleCNN().to(device)
    
    # Track metrics
    results = {
        'rounds': [],
        'accuracies': [],
        'thresholds': [],
        'outlier_ratios': [],
        'threshold_adjustments': []
    }
    
    threshold = INITIAL_THRESHOLD
    num_byzantine = int(NUM_CLIENTS * BYZANTINE_RATIO)
    byzantine_clients = list(range(num_byzantine))
    
    print(f"\nConfiguration:")
    print(f"  Clients: {NUM_CLIENTS} ({num_byzantine} Byzantine)")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Initial threshold: {INITIAL_THRESHOLD}")
    print(f"  Adaptation rate: {ADAPTATION_RATE}\n")
    
    for round_num in range(NUM_ROUNDS):
        # Client training
        updates = []
        weights = []
        
        for client_idx in range(NUM_CLIENTS):
            # Create client dataloader
            client_data = torch.utils.data.Subset(train_dataset, client_indices[client_idx])
            client_loader = torch.utils.data.DataLoader(
                client_data, batch_size=BATCH_SIZE, shuffle=True
            )
            
            # Train locally
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            update = local_train(client_model, client_loader, device, 
                               LOCAL_EPOCHS, LEARNING_RATE)
            
            # Byzantine attack
            if client_idx in byzantine_clients:
                update = create_byzantine_update(update, scale_factor=-5.0)
            
            updates.append(update)
            weights.append(len(client_indices[client_idx]))
        
        # ATMA aggregation
        aggregated, new_threshold, stats = aggregate_atma(
            updates, weights, threshold, ADAPTATION_RATE
        )
        
        # Update global model
        global_model.load_state_dict(aggregated)
        threshold = new_threshold
        
        # Evaluate every 5 rounds
        if (round_num + 1) % 5 == 0 or round_num == 0:
            accuracy = evaluate_model(global_model, test_loader, device)
            
            results['rounds'].append(round_num + 1)
            results['accuracies'].append(accuracy)
            results['thresholds'].append(threshold)
            results['outlier_ratios'].append(stats['outlier_ratio'])
            results['threshold_adjustments'].append(stats['threshold_adjustment'])
            
            print(f"Round {round_num+1:3d}/{NUM_ROUNDS} | "
                  f"Accuracy: {accuracy:6.2f}% | "
                  f"Threshold: {threshold:.3f} | "
                  f"Outliers: {stats['outlier_ratio']:.1%}")
        
        # Progress indicator
        if (round_num + 1) % 20 == 0:
            print("-" * 70)
    
    # Final evaluation
    final_accuracy = evaluate_model(global_model, test_loader, device)
    
    print("\n" + "="*70)
    print(f"FINAL RESULT: {final_accuracy:.2f}%")
    print(f"Comparison with paper:")
    print(f"  - ATMA 50r (paper): 85.12%")
    print(f"  - ATMA 160r (this run): {final_accuracy:.2f}%")
    print(f"  - TrimmedMean 160r (paper): 93.45%")
    print(f"  - Improvement from extended training: {final_accuracy - 85.12:+.2f}%")
    print("="*70)
    
    # Save results
    output_data = {
        'experiment': 'ATMA_160_rounds',
        'config': {
            'num_clients': NUM_CLIENTS,
            'byzantine_ratio': BYZANTINE_RATIO,
            'num_rounds': NUM_ROUNDS,
            'initial_threshold': INITIAL_THRESHOLD,
            'adaptation_rate': ADAPTATION_RATE,
            'learning_rate': LEARNING_RATE
        },
        'final_accuracy': final_accuracy,
        'results': results
    }
    
    output_file = f'results/atma_160rounds_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Plot results
    plot_atma_results(results, final_accuracy)
    
    return results, final_accuracy

def plot_atma_results(results: Dict, final_accuracy: float):
    """Create visualization of ATMA 160-round performance"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy over rounds
    ax1.plot(results['rounds'], results['accuracies'], 'b-', linewidth=2, label='ATMA 160r')
    ax1.axhline(y=85.12, color='orange', linestyle='--', label='ATMA 50r (paper)')
    ax1.axhline(y=93.45, color='green', linestyle='--', label='TrimmedMean 160r (paper)')
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('ATMA Accuracy Evolution (160 Rounds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Threshold adaptation
    ax2.plot(results['rounds'], results['thresholds'], 'r-', linewidth=2)
    ax2.axhline(y=INITIAL_THRESHOLD, color='gray', linestyle='--', label='Initial')
    ax2.axhline(y=MIN_THRESHOLD, color='red', linestyle=':', alpha=0.5, label='Min')
    ax2.axhline(y=MAX_THRESHOLD, color='red', linestyle=':', alpha=0.5, label='Max')
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Threshold Value')
    ax2.set_title('Adaptive Threshold Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Outlier detection rate
    ax3.plot(results['rounds'], [r * 100 for r in results['outlier_ratios']], 'purple', linewidth=2)
    ax3.axhline(y=BYZANTINE_RATIO * 100, color='red', linestyle='--', label=f'Byzantine ratio ({BYZANTINE_RATIO:.0%})')
    ax3.set_xlabel('Training Round')
    ax3.set_ylabel('Outliers Detected (%)')
    ax3.set_title('ATMA Outlier Detection Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparison bar chart
    methods = ['ATMA 50r\n(paper)', 'ATMA 160r\n(this run)', 'TrimmedMean 160r\n(paper)']
    accuracies = [85.12, final_accuracy, 93.45]
    colors = ['orange', 'blue', 'green']
    
    bars = ax4.bar(methods, accuracies, color=colors, alpha=0.7)
    ax4.set_ylabel('Final Test Accuracy (%)')
    ax4.set_title('Method Comparison')
    ax4.set_ylim([80, 95])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/atma_160rounds_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('visualizations/atma_160rounds_analysis.png', dpi=300, bbox_inches='tight')
    print("Plots saved to visualizations/atma_160rounds_analysis.pdf")
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    results, final_accuracy = run_atma_160rounds()
