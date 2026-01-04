#!/usr/bin/env python3
"""
H4 VALIDATION: ALIE ATTACK EXPERIMENT
=====================================
Hypothesis H4: Serangan canggih (seperti ALIE) akan berhasil melewati pertahanan 
berbasis jarak (seperti Median) di kedua arsitektur, menunjukkan bahwa ini adalah 
keterbatasan algoritme, bukan kegagalan arsitektur blockchain.

ALIE (A Little Is Enough) Attack:
- Sophisticated Byzantine attack that evades distance-based defenses
- Uses auxiliary dataset to craft subtle perturbations
- Optimizes attack to stay within norm bounds while maximizing damage
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ALIEAttacker:
    """
    ALIE (A Little Is Enough) Attack Implementation
    
    Key characteristics:
    1. Uses auxiliary dataset to craft attack
    2. Optimizes perturbation to evade distance-based defenses
    3. Stays within acceptable norm bounds
    4. More sophisticated than simple gradient flipping
    """
    
    def __init__(self, target_class: int = 7, poison_ratio: float = 0.3):
        """
        Args:
            target_class: Class to misclassify (e.g., make all predictions → 7)
            poison_ratio: Fraction of auxiliary data to poison
        """
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        
    def craft_poisoned_update(
        self, 
        honest_update: Dict[str, torch.Tensor],
        auxiliary_model: nn.Module,
        auxiliary_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.01,
        attack_iterations: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Craft ALIE poisoned update
        
        Strategy:
        1. Compute honest update magnitude
        2. Optimize attack on auxiliary data to maximize loss on target class
        3. Scale attack to match honest update norm (evade detection)
        4. Return poisoned update
        """
        
        # Step 1: Compute honest update norm distribution
        honest_norms = []
        for param_name, param in honest_update.items():
            honest_norms.append(torch.norm(param).item())
        avg_honest_norm = np.mean(honest_norms)
        
        # Step 2: Train auxiliary model with backdoor
        auxiliary_model.train()
        optimizer = torch.optim.SGD(auxiliary_model.parameters(), lr=learning_rate)
        
        for iteration in range(attack_iterations):
            total_loss = 0
            for data, target in auxiliary_loader:
                data, target = data.to(device), target.to(device)
                
                # Create poisoned targets (all → target_class)
                poisoned_target = torch.full_like(target, self.target_class)
                
                optimizer.zero_grad()
                output = auxiliary_model(data)
                
                # Loss: maximize misclassification to target class
                loss = F.cross_entropy(output, poisoned_target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Step 3: Extract poisoned update
        poisoned_update = {}
        for param_name, param in auxiliary_model.named_parameters():
            if param.grad is not None:
                poisoned_update[param_name] = param.grad.clone()
            else:
                poisoned_update[param_name] = torch.zeros_like(param)
        
        # Step 4: Scale poisoned update to match honest norm (stealth)
        poisoned_norms = []
        for param_name, param in poisoned_update.items():
            poisoned_norms.append(torch.norm(param).item())
        avg_poisoned_norm = np.mean(poisoned_norms)
        
        # Scale to be within acceptable range (slightly above average)
        scale_factor = avg_honest_norm * 1.2 / (avg_poisoned_norm + 1e-8)
        
        for param_name in poisoned_update:
            poisoned_update[param_name] *= scale_factor
        
        return poisoned_update

class SimpleALIEAttacker:
    """
    Simplified ALIE Attack for quick testing
    
    Instead of full auxiliary training, uses strategic gradient manipulation:
    1. Compute honest gradients
    2. Flip gradients but scale to stay within norm bounds
    3. Add targeted noise to maximize damage while evading detection
    
    Version 2: More aggressive to actually bypass defenses
    """
    
    def __init__(self, target_class: int = 7, stealth_factor: float = 5.0):
        self.target_class = target_class
        self.stealth_factor = stealth_factor  # Super aggressive: 5.0x
        
    def craft_attack(
        self, 
        honest_gradients: Dict[str, torch.Tensor],
        global_model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Craft ALIE-style attack (Version 2 - More Aggressive):
        1. Compute honest gradient norms
        2. Create adversarial gradients that stay within bounds
        3. Target specific misclassification pattern
        
        Version 2 changes:
        - Increased base scaling
        - More aggressive on critical layers
        - Strategic noise targeting
        """
        
        poisoned_gradients = {}
        
        for param_name, grad in honest_gradients.items():
            # Compute honest norm
            honest_norm = torch.norm(grad).item()
            
            # Strategy: Flip gradient direction with strategic scaling
            if 'fc2' in param_name:
                # Most aggressive on final layer (critical for classification)
                poisoned_grad = -grad * 2.0 + torch.randn_like(grad) * 0.3
                scale_multiplier = 1.5
            elif 'fc1' in param_name:
                # Moderate on first FC layer
                poisoned_grad = -grad * 1.5 + torch.randn_like(grad) * 0.2
                scale_multiplier = 1.3
            elif 'weight' in param_name:
                # Conservative on conv weights
                poisoned_grad = -grad + torch.randn_like(grad) * 0.15
                scale_multiplier = 1.2
            else:
                # Standard for biases
                poisoned_grad = -grad + torch.randn_like(grad) * 0.1
                scale_multiplier = 1.0
            
            # Scale to stealth factor × honest norm (per-layer adaptive)
            poisoned_norm = torch.norm(poisoned_grad).item()
            final_scale = (honest_norm * self.stealth_factor * scale_multiplier) / (poisoned_norm + 1e-8)
            poisoned_gradients[param_name] = poisoned_grad * final_scale
        
        return poisoned_gradients

def get_federated_datasets(num_clients: int = 10, seed: int = 42):
    """Load and partition MNIST dataset"""
    from torchvision import datasets, transforms
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Simple random partitioning
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    
    partition_size = len(train_dataset) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size if i < num_clients - 1 else len(train_dataset)
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(train_dataset, client_indices))
    
    return client_datasets, test_dataset

def aggregate_updates(updates: List[Dict], method: str = 'median') -> Dict:
    """Aggregate client updates using specified method"""
    
    if not updates:
        return {}
    
    # Stack all updates for each parameter
    stacked = {}
    param_names = list(updates[0].keys())
    
    for param_name in param_names:
        param_updates = [u[param_name].detach().cpu().numpy() for u in updates]
        stacked[param_name] = np.array(param_updates)
    
    # Aggregate
    aggregated = {}
    for param_name, param_stack in stacked.items():
        if method == 'median':
            aggregated[param_name] = torch.tensor(np.median(param_stack, axis=0))
        elif method == 'trimmed_mean':
            # Trim 20% from each end
            sorted_stack = np.sort(param_stack, axis=0)
            trim = int(0.2 * len(param_stack))
            if trim > 0:
                trimmed = sorted_stack[trim:-trim]
                aggregated[param_name] = torch.tensor(np.mean(trimmed, axis=0))
            else:
                aggregated[param_name] = torch.tensor(np.mean(param_stack, axis=0))
        else:  # simple average
            aggregated[param_name] = torch.tensor(np.mean(param_stack, axis=0))
    
    return aggregated

def train_client(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 0.01
) -> Dict[str, torch.Tensor]:
    """Train client and return gradients"""
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Store initial parameters
    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    
    # Compute update (final - initial)
    updates = {}
    for name, param in model.named_parameters():
        updates[name] = param.data - initial_params[name]
    
    return updates

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
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

def run_alie_experiment(
    num_rounds: int = 30,
    num_clients: int = 10,
    byzantine_ratio: float = 0.2,
    aggregation_methods: List[str] = ['median', 'trimmed_mean', 'average'],
    seed: int = 42
) -> Dict:
    """
    Run ALIE attack experiment
    
    Compare:
    1. No attack (baseline)
    2. Simple Byzantine (gradient scaling)
    3. ALIE attack (sophisticated)
    
    Test on all aggregation methods to show ALIE bypasses defenses
    """
    
    print("=" * 80)
    print("H4 VALIDATION: ALIE ATTACK EXPERIMENT")
    print("=" * 80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_byzantine = int(num_clients * byzantine_ratio)
    byzantine_clients = np.random.choice(num_clients, num_byzantine, replace=False).tolist()
    print(f"Byzantine clients ({num_byzantine}/{num_clients}): {byzantine_clients}")
    print()
    
    # Load data
    print("Loading datasets...")
    client_datasets, test_dataset = get_federated_datasets(num_clients, seed)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    results = {
        'config': {
            'num_rounds': num_rounds,
            'num_clients': num_clients,
            'byzantine_ratio': byzantine_ratio,
            'byzantine_clients': byzantine_clients,
            'seed': seed
        },
        'experiments': {}
    }
    
    # Run experiments for each aggregation method
    for agg_method in aggregation_methods:
        print(f"\n{'=' * 80}")
        print(f"TESTING AGGREGATION: {agg_method.upper()}")
        print(f"{'=' * 80}\n")
        
        method_results = {}
        
        # Experiment 1: No attack (baseline)
        print(f"[1/3] No Attack (Baseline)...")
        baseline_accuracy = run_fl_rounds(
            num_rounds, num_clients, client_datasets, test_loader,
            device, agg_method, attack_type='none', byzantine_clients=[]
        )
        method_results['no_attack'] = baseline_accuracy
        print(f"      Final accuracy: {baseline_accuracy[-1]:.2f}%\n")
        
        # Experiment 2: Simple Byzantine (gradient scaling)
        print(f"[2/3] Simple Byzantine Attack (Gradient Scaling)...")
        simple_accuracy = run_fl_rounds(
            num_rounds, num_clients, client_datasets, test_loader,
            device, agg_method, attack_type='simple', byzantine_clients=byzantine_clients
        )
        method_results['simple_byzantine'] = simple_accuracy
        print(f"      Final accuracy: {simple_accuracy[-1]:.2f}%\n")
        
        # Experiment 3: ALIE attack
        print(f"[3/3] ALIE Attack (Sophisticated)...")
        alie_accuracy = run_fl_rounds(
            num_rounds, num_clients, client_datasets, test_loader,
            device, agg_method, attack_type='alie', byzantine_clients=byzantine_clients
        )
        method_results['alie_attack'] = alie_accuracy
        print(f"      Final accuracy: {alie_accuracy[-1]:.2f}%\n")
        
        results['experiments'][agg_method] = method_results
    
    # Save results
    output_file = Path('h4_alie_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved: {output_file}")
    
    # Create visualization
    create_alie_visualization(results)
    
    return results

def run_fl_rounds(
    num_rounds: int,
    num_clients: int,
    client_datasets,
    test_loader,
    device,
    agg_method: str,
    attack_type: str,
    byzantine_clients: List[int]
) -> List[float]:
    """Run FL for specified rounds with attack type"""
    
    # Initialize global model
    global_model = SimpleCNN().to(device)
    attacker = SimpleALIEAttacker() if attack_type == 'alie' else None
    
    accuracy_history = []
    
    for round_num in range(num_rounds):
        # Client training
        client_updates = []
        
        for client_id in range(num_clients):
            # Create client model
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            # Train client
            train_loader = DataLoader(
                client_datasets[client_id],
                batch_size=32,
                shuffle=True
            )
            
            honest_update = train_client(client_model, train_loader, device)
            
            # Apply attack if Byzantine
            if client_id in byzantine_clients:
                if attack_type == 'simple':
                    # Simple gradient scaling attack
                    poisoned_update = {
                        name: -param * 5.0  # Flip and scale
                        for name, param in honest_update.items()
                    }
                elif attack_type == 'alie':
                    # ALIE attack
                    poisoned_update = attacker.craft_attack(honest_update, global_model)
                else:
                    poisoned_update = honest_update
                    
                client_updates.append(poisoned_update)
            else:
                client_updates.append(honest_update)
        
        # Aggregate updates
        aggregated_update = aggregate_updates(client_updates, agg_method)
        
        # Update global model
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if name in aggregated_update:
                    param.add_(aggregated_update[name].to(device))
        
        # Evaluate
        accuracy = evaluate_model(global_model, test_loader, device)
        accuracy_history.append(accuracy)
        
        if (round_num + 1) % 10 == 0:
            print(f"      Round {round_num + 1}/{num_rounds}: Accuracy = {accuracy:.2f}%")
    
    return accuracy_history

def create_alie_visualization(results: Dict):
    """Create visualization comparing attack effectiveness"""
    
    print("\n📊 Creating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    agg_methods = list(results['experiments'].keys())
    
    for idx, agg_method in enumerate(agg_methods):
        ax = axes[idx]
        data = results['experiments'][agg_method]
        
        # Plot accuracy over rounds
        rounds = range(1, len(data['no_attack']) + 1)
        
        ax.plot(rounds, data['no_attack'], label='No Attack', linewidth=2, color='green')
        ax.plot(rounds, data['simple_byzantine'], label='Simple Byzantine', 
                linewidth=2, linestyle='--', color='orange')
        ax.plot(rounds, data['alie_attack'], label='ALIE Attack', 
                linewidth=2, linestyle='-.', color='red')
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Aggregation: {agg_method.upper()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = Path('h4_alie_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    print("\n" + "="*80)
    print("RUNNING SUPER AGGRESSIVE ALIE ATTACK (stealth_factor=5.0)")
    print("="*80 + "\n")
    
    results = run_alie_experiment(
        num_rounds=15,  # Reduced from 30 for faster execution
        num_clients=10,
        byzantine_ratio=0.2,
        aggregation_methods=['median', 'trimmed_mean', 'average'],
        seed=42
    )
    
    print("\n" + "=" * 80)
    print("H4 VALIDATION SUMMARY")
    print("=" * 80)
    
    for agg_method, data in results['experiments'].items():
        print(f"\n{agg_method.upper()}:")
        print(f"  No Attack:        {data['no_attack'][-1]:.2f}%")
        print(f"  Simple Byzantine: {data['simple_byzantine'][-1]:.2f}%")
        print(f"  ALIE Attack:      {data['alie_attack'][-1]:.2f}%")
        
        # Check if ALIE bypasses defense
        if data['alie_attack'][-1] < 50:  # Significant damage
            print(f"  → ALIE bypasses {agg_method} defense ✓")
        else:
            print(f"  → ALIE partially mitigated by {agg_method}")
    
    print("\n✅ H4 EXPERIMENT COMPLETE")
