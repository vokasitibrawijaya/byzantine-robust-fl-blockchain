#!/usr/bin/env python3
"""
H2 Provenance Detection - RERUN WITH VALID METHODOLOGY

IMPORTANT: This script re-runs the H2 experiment with:
1. Clear ground truth labels (actual attack rounds, independent of detection)
2. Proper ROC calculation using sklearn
3. Transparent logging of all intermediate steps

The goal is to answer: "Can blockchain-based provenance detect Byzantine attacks?"
"""

import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def simulate_federated_round(round_num: int, 
                              n_clients: int = 20,
                              byzantine_ratio: float = 0.2,
                              attack_active: bool = False) -> dict:
    """
    Simulate one round of federated learning with potential Byzantine attack.
    
    Returns:
        Dictionary with round metrics that would be logged to blockchain
    """
    n_byzantine = int(n_clients * byzantine_ratio)
    n_honest = n_clients - n_byzantine
    
    # Honest clients produce updates with realistic variance
    honest_norms = np.random.normal(1.0, 0.2, n_honest)
    
    if attack_active:
        # Byzantine clients use ADAPTIVE attacks that try to evade detection
        # This makes detection genuinely challenging
        attack_type = np.random.choice(['stealthy', 'subtle', 'moderate', 'obvious'], 
                                        p=[0.3, 0.3, 0.25, 0.15])
        if attack_type == 'stealthy':
            # Nearly indistinguishable from honest - within 1 sigma
            byzantine_norms = np.random.normal(1.1, 0.2, n_byzantine)
        elif attack_type == 'subtle':
            # Slightly elevated - 1-2 sigma
            byzantine_norms = np.random.normal(1.3, 0.25, n_byzantine)
        elif attack_type == 'moderate':
            # Clearly elevated - 2-3 sigma
            byzantine_norms = np.random.normal(1.8, 0.3, n_byzantine)
        else:
            # Obviously malicious - >3 sigma
            byzantine_norms = np.random.normal(2.5, 0.4, n_byzantine)
    else:
        # Even Byzantine clients behave normally when not attacking
        byzantine_norms = np.random.normal(1.0, 0.2, n_byzantine)
    
    all_norms = np.concatenate([honest_norms, byzantine_norms])
    
    # Aggregation statistics (what gets logged)
    return {
        'round': round_num,
        'mean_norm': float(np.mean(all_norms)),
        'std_norm': float(np.std(all_norms)),
        'max_norm': float(np.max(all_norms)),
        'n_clients': n_clients,
        'attack_active': attack_active  # Ground truth (in real system, this is unknown)
    }


def compute_anomaly_score(round_data: dict, baseline_mean: float, baseline_std: float) -> float:
    """
    Compute anomaly score for a round based on deviation from baseline.
    
    This is what the blockchain-based detection system would compute.
    Score > threshold indicates potential attack.
    """
    if baseline_std < 1e-6:
        baseline_std = 0.1  # Prevent division by zero
    
    # Z-score based on mean norm deviation
    z_mean = (round_data['mean_norm'] - baseline_mean) / baseline_std
    
    # Also consider std deviation (attacks increase variance)
    z_std = (round_data['std_norm'] - baseline_std) / (baseline_std * 0.5)
    
    # Combined anomaly score
    anomaly_score = np.sqrt(z_mean**2 + z_std**2)
    
    return float(anomaly_score)


def run_h2_experiment(n_rounds: int = 50,
                       n_clients: int = 20,
                       byzantine_ratio: float = 0.2,
                       attack_probability: float = 0.3,
                       blockchain_enabled: bool = True,
                       seed: int = 42) -> dict:
    """
    Run H2 provenance detection experiment.
    
    Args:
        n_rounds: Number of FL rounds to simulate
        n_clients: Total number of clients
        byzantine_ratio: Fraction of Byzantine clients
        attack_probability: Probability Byzantine clients attack each round
        blockchain_enabled: If True, logs are immutable; if False, attacker can tamper
        seed: Random seed for reproducibility
    
    Returns:
        Complete experiment results including ROC data
    """
    np.random.seed(seed)
    
    # Phase 1: Collect baseline (first 10 rounds, no attacks)
    baseline_rounds = []
    for r in range(10):
        round_data = simulate_federated_round(r, n_clients, byzantine_ratio, attack_active=False)
        baseline_rounds.append(round_data)
    
    baseline_mean = np.mean([r['mean_norm'] for r in baseline_rounds])
    baseline_std = np.std([r['mean_norm'] for r in baseline_rounds])
    
    print(f"Baseline established: mean={baseline_mean:.3f}, std={baseline_std:.3f}")
    
    # Phase 2: Main experiment
    all_rounds = []
    ground_truth = []  # True if attack occurred
    anomaly_scores = []
    
    for r in range(n_rounds):
        # Decide if attack happens this round (independent of detection)
        attack_this_round = np.random.random() < attack_probability
        
        # Simulate round
        round_data = simulate_federated_round(r, n_clients, byzantine_ratio, 
                                               attack_active=attack_this_round)
        
        # If centralized (no blockchain), attacker can tamper logs to hide attack
        if not blockchain_enabled and attack_this_round:
            # Tamper the logged data to look normal
            round_data['mean_norm'] = baseline_mean + np.random.normal(0, baseline_std * 0.5)
            round_data['std_norm'] = baseline_std * np.random.uniform(0.8, 1.2)
        
        all_rounds.append(round_data)
        ground_truth.append(attack_this_round)
        
        # Compute anomaly score (detection)
        score = compute_anomaly_score(round_data, baseline_mean, baseline_std)
        anomaly_scores.append(score)
    
    # Convert to numpy arrays
    ground_truth = np.array(ground_truth, dtype=int)
    anomaly_scores = np.array(anomaly_scores)
    
    # Compute ROC using sklearn
    if ground_truth.sum() > 0 and ground_truth.sum() < len(ground_truth):
        fpr, tpr, thresholds = roc_curve(ground_truth, anomaly_scores)
        auc = roc_auc_score(ground_truth, anomaly_scores)
    else:
        # Edge case: all same class
        fpr, tpr, thresholds = [0, 1], [0, 1], [0]
        auc = 0.5
    
    # Find optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
    # Confusion matrix at optimal threshold
    predictions = (anomaly_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    
    results = {
        'experiment_info': {
            'n_rounds': n_rounds,
            'n_clients': n_clients,
            'byzantine_ratio': byzantine_ratio,
            'attack_probability': attack_probability,
            'blockchain_enabled': blockchain_enabled,
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        },
        'baseline': {
            'mean': float(baseline_mean),
            'std': float(baseline_std)
        },
        'ground_truth_summary': {
            'total_rounds': int(len(ground_truth)),
            'attack_rounds': int(ground_truth.sum()),
            'clean_rounds': int((ground_truth == 0).sum())
        },
        'detection_results': {
            'auc': float(auc),
            'optimal_threshold': float(best_threshold),
            'confusion_matrix': {
                'TP': int(tp),
                'FP': int(fp),
                'TN': int(tn),
                'FN': int(fn)
            },
            'metrics': {
                'TPR': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'FPR': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                'Precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'F1': float(2*tp / (2*tp + fp + fn)) if (2*tp + fp + fn) > 0 else 0.0,
                'Youden_J': float(j_scores[best_idx])
            }
        },
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        },
        'raw_data': {
            'anomaly_scores': anomaly_scores.tolist(),
            'ground_truth': ground_truth.tolist()
        }
    }
    
    return results


def generate_roc_figure(blockchain_results: dict, 
                        centralized_results: dict,
                        output_dir: str):
    """Generate ROC curve figure comparing blockchain vs centralized."""
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Blockchain ROC
    bc_fpr = blockchain_results['roc_curve']['fpr']
    bc_tpr = blockchain_results['roc_curve']['tpr']
    bc_auc = blockchain_results['detection_results']['auc']
    ax.plot(bc_fpr, bc_tpr, 'b-', linewidth=2, 
            label=f'Blockchain (AUC={bc_auc:.3f})')
    
    # Centralized ROC
    ct_fpr = centralized_results['roc_curve']['fpr']
    ct_tpr = centralized_results['roc_curve']['tpr']
    ct_auc = centralized_results['detection_results']['auc']
    ax.plot(ct_fpr, ct_tpr, 'r--', linewidth=2, 
            label=f'Centralized (AUC={ct_auc:.3f})')
    
    # Optimal point for blockchain
    bc_metrics = blockchain_results['detection_results']['metrics']
    ax.plot(bc_metrics['FPR'], bc_metrics['TPR'], 'bo', markersize=10,
            label=f'Optimal (TPR={bc_metrics["TPR"]:.1%}, FPR={bc_metrics["FPR"]:.1%})')
    
    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.5)')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('H2: Provenance Corruption Detection\n(Blockchain vs Centralized)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/h2_roc_curve.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}/h2_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {output_dir}/h2_roc_curve.pdf")


def main():
    print("=" * 70)
    print("H2 PROVENANCE DETECTION - VALID RERUN")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Random Seed: {SEED}")
    print()
    
    # Get project paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    results_dir = project_root / 'results'
    viz_dir = project_root / 'ACCESS_latex_template_20240429' / 'visualizations'
    
    # Run experiments
    print("[1/4] Running blockchain-enabled experiment...")
    blockchain_results = run_h2_experiment(
        n_rounds=50,
        n_clients=20,
        byzantine_ratio=0.2,
        attack_probability=0.3,
        blockchain_enabled=True,
        seed=SEED
    )
    
    print(f"      Blockchain AUC: {blockchain_results['detection_results']['auc']:.4f}")
    print(f"      Attack rounds: {blockchain_results['ground_truth_summary']['attack_rounds']}/{blockchain_results['ground_truth_summary']['total_rounds']}")
    
    print("\n[2/4] Running centralized experiment...")
    centralized_results = run_h2_experiment(
        n_rounds=50,
        n_clients=20,
        byzantine_ratio=0.2,
        attack_probability=0.3,
        blockchain_enabled=False,
        seed=SEED
    )
    
    print(f"      Centralized AUC: {centralized_results['detection_results']['auc']:.4f}")
    
    print("\n[3/4] Generating ROC figure...")
    generate_roc_figure(blockchain_results, centralized_results, str(viz_dir))
    generate_roc_figure(blockchain_results, centralized_results, 
                        str(project_root / 'visualizations'))
    
    print("\n[4/4] Saving results...")
    combined_results = {
        'blockchain': blockchain_results,
        'centralized': centralized_results,
        'comparison': {
            'auc_difference': blockchain_results['detection_results']['auc'] - centralized_results['detection_results']['auc'],
            'blockchain_advantage': blockchain_results['detection_results']['auc'] > centralized_results['detection_results']['auc']
        }
    }
    
    output_path = results_dir / 'h2_valid_rerun_results.json'
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    print(f"      Saved: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    bc = blockchain_results['detection_results']
    ct = centralized_results['detection_results']
    
    print("\n                    Blockchain    Centralized")
    print("-" * 50)
    print(f"AUC:                {bc['auc']:.4f}         {ct['auc']:.4f}")
    print(f"TPR (Recall):       {bc['metrics']['TPR']:.1%}          {ct['metrics']['TPR']:.1%}")
    print(f"FPR:                {bc['metrics']['FPR']:.1%}           {ct['metrics']['FPR']:.1%}")
    print(f"Precision:          {bc['metrics']['Precision']:.1%}          {ct['metrics']['Precision']:.1%}")
    print(f"F1 Score:           {bc['metrics']['F1']:.4f}         {ct['metrics']['F1']:.4f}")
    print(f"Youden's J:         {bc['metrics']['Youden_J']:.4f}         {ct['metrics']['Youden_J']:.4f}")
    
    print("\nConfusion Matrix (Blockchain):")
    cm = bc['confusion_matrix']
    print(f"  TP={cm['TP']}, FP={cm['FP']}, TN={cm['TN']}, FN={cm['FN']}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if bc['auc'] > 0.7:
        print("✅ Blockchain-based detection shows GOOD discriminative power")
    elif bc['auc'] > 0.5:
        print("⚠️ Blockchain-based detection shows MODERATE discriminative power")
    else:
        print("❌ Blockchain-based detection shows POOR discriminative power")
    
    if bc['auc'] > ct['auc'] + 0.1:
        print("✅ Blockchain significantly outperforms centralized logging")
    elif bc['auc'] > ct['auc']:
        print("⚠️ Blockchain marginally better than centralized logging")
    else:
        print("❌ No clear advantage of blockchain over centralized logging")
    
    print("=" * 70)
    
    return combined_results


if __name__ == '__main__':
    results = main()
