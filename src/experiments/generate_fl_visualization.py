"""
Generate visualization for Real FL Training FIXED results
Uses actual MNIST data results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_file = Path(__file__).parent.parent.parent / 'results' / 'real_fl_training_FIXED.json'

with open(results_file, 'r') as f:
    data = json.load(f)

# Extract data for plotting
methods = ['mean', 'median', 'trimmed_mean']
method_labels = ['Mean', 'Median', 'TrimmedMean']
colors = ['#e74c3c', '#3498db', '#27ae60']

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Accuracy over rounds (seed 42)
ax1 = axes[0]
for i, method in enumerate(methods):
    result = data['detailed_results'][method][0]  # First seed
    rounds = result['history']['rounds']
    accuracies = [acc * 100 for acc in result['history']['test_accuracy']]
    ax1.plot(rounds, accuracies, label=method_labels[i], color=colors[i], linewidth=2, marker='o', markersize=4)

ax1.set_xlabel('Training Round', fontsize=12)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
ax1.set_title('Real FL Training on MNIST (Seed 42)', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 80])
ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')

# Right plot: Final accuracy comparison with error bars
ax2 = axes[1]
summary = data['summary']

x_pos = np.arange(len(methods))
means = [summary[m]['mean_accuracy'] * 100 for m in methods]
stds = [summary[m]['std_accuracy'] * 100 for m in methods]

bars = ax2.bar(x_pos, means, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax2.annotate(f'{mean:.1f}% ± {std:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height + std + 1),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(method_labels, fontsize=12)
ax2.set_ylabel('Final Test Accuracy (%)', fontsize=12)
ax2.set_title('Final Accuracy Comparison (3 Seeds)', fontsize=14)
ax2.set_ylim([0, 85])
ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
ax2.text(2.5, 12, 'Random Baseline (10%)', fontsize=9, color='gray')

plt.tight_layout()

# Save figure
output_dir = Path(__file__).parent.parent.parent / 'ACCESS_latex_template_20240429' / 'visualizations'
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / 'fl_convergence.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fl_convergence.png', dpi=150, bbox_inches='tight')

print(f"[OK] Saved visualizations to {output_dir}")
print(f"\nSummary:")
for method in methods:
    s = summary[method]
    print(f"  {method.upper()}: {s['mean_accuracy']*100:.2f}% ± {s['std_accuracy']*100:.2f}%")
