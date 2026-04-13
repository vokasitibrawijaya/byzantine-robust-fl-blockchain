"""
Regenerate Figures 1 and 2 with improved readability for ETASR revision.
Addresses Reviewer A: "Figures 1-2 not readable"
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
base = Path(__file__).parent.parent
results_dir = base.parent / 'results'
out_dir = base / 'paper_1kolom'
viz_dir = out_dir / 'visualizations'

# ============================================================
# FIGURE 1: Convergence Comparison (CIFAR-10, all algorithms)
# ============================================================
print("[*] Generating Figure 1: Convergence Comparison...")

cifar = json.load(open(results_dir / 'cifar10_blockchain_simple_20251226_232614.json'))

# Extract per-method convergence data (160 rounds)
methods_160 = {}
methods_50 = {}
for r in cifar['results']:
    key = r['method']
    rounds_list = [a['round'] for a in r['accuracies']]
    acc_list = [a['accuracy'] for a in r['accuracies']]
    if r['rounds'] == 160:
        methods_160[key] = (rounds_list, acc_list)
    elif r['rounds'] == 50:
        methods_50[key] = (rounds_list, acc_list)

fig, ax = plt.subplots(figsize=(8, 5.5))

# Plot styles - distinct colors and markers
styles = {
    'TrimmedMean': {'color': '#2ca02c', 'marker': 's', 'ls': '-', 'lw': 2.5, 'ms': 7},
    'ATMA':        {'color': '#1f77b4', 'marker': '^', 'ls': '-', 'lw': 2.5, 'ms': 7},
    'Krum':        {'color': '#ff7f0e', 'marker': 'D', 'ls': '--', 'lw': 2.0, 'ms': 6},
    'FedAvg':      {'color': '#d62728', 'marker': 'o', 'ls': ':', 'lw': 2.0, 'ms': 6},
}

# Plot 160-round data for each method
for method in ['TrimmedMean', 'ATMA', 'Krum', 'FedAvg']:
    if method in methods_160:
        rds, acc = methods_160[method]
        s = styles[method]
        label = f"{method} (final: {acc[-1]:.1f}%)"
        ax.plot(rds, acc, label=label, color=s['color'], marker=s['marker'],
                linestyle=s['ls'], linewidth=s['lw'], markersize=s['ms'],
                markeredgecolor='white', markeredgewidth=0.8)

# Add random baseline line
ax.axhline(y=10, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)
ax.text(165, 11.5, 'Random Baseline (10%)', fontsize=9, color='gray', ha='right')

ax.set_xlabel('Training Round', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('CIFAR-10 Convergence Under 20% Byzantine Attack\n(Dirichlet α=0.5, Label-Flip Scale=-5.0)',
             fontsize=13, fontweight='bold')
ax.legend(loc='center right', fontsize=11, framealpha=0.95, edgecolor='gray')
ax.set_xlim([5, 165])
ax.set_ylim([0, 75])
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.set_xticks([10, 30, 50, 70, 90, 110, 130, 150])

plt.tight_layout()
plt.savefig(out_dir / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'convergence_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  -> Saved convergence_comparison.png/pdf (300 DPI)")

# ============================================================
# FIGURE 2: ROC Curve (improved readability)
# ============================================================
print("[*] Generating Figure 2: ROC Curve (improved)...")

h2 = json.load(open(results_dir / 'h2_valid_rerun_results.json'))

fig, ax = plt.subplots(figsize=(6.5, 5.5))

# Blockchain ROC data (from h2_valid_rerun_results.json)
fpr_bc = h2['blockchain']['roc_curve']['fpr']
tpr_bc = h2['blockchain']['roc_curve']['tpr']
auc_bc = h2['blockchain']['detection_results']['auc']

# Centralized ROC data
fpr_ct = h2['centralized']['roc_curve']['fpr']
tpr_ct = h2['centralized']['roc_curve']['tpr']
from sklearn.metrics import auc as compute_auc
auc_ct = compute_auc(fpr_ct, tpr_ct)

# Plot Blockchain ROC
ax.plot(fpr_bc, tpr_bc, color='#1f77b4', linewidth=3.0, 
        label=f'Blockchain (AUC = {auc_bc:.3f})', zorder=5)

# Centralized baseline
ax.plot(fpr_ct, tpr_ct, color='#d62728', linewidth=2.5, linestyle='--',
        label=f'Centralized / Mutable (AUC = {auc_ct:.3f})', zorder=4)

# Random classifier
ax.plot([0, 1], [0, 1], color='gray', linewidth=1.2, linestyle=':', 
        alpha=0.7, label='Random (AUC = 0.5)', zorder=3)

# Optimal operating point
opt_tpr = h2['blockchain']['detection_results']['metrics']['TPR']
opt_fpr = h2['blockchain']['detection_results']['metrics']['FPR']
ax.scatter([opt_fpr], [opt_tpr], s=150, color='#1f77b4', zorder=6, edgecolor='white', linewidth=2)
ax.annotate(f'Optimal Point\n(FPR={opt_fpr:.2f}, TPR={opt_tpr:.2f})', 
            xy=(opt_fpr, opt_tpr), xytext=(0.15, 0.65),
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax.set_title('Provenance Verification ROC Curve\n(Blockchain vs. Centralized System)', 
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10.5, framealpha=0.95, edgecolor='gray')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.25, linewidth=0.5)

plt.tight_layout()
plt.savefig(viz_dir / 'h2_roc_curve.png', dpi=300, bbox_inches='tight')
plt.savefig(viz_dir / 'h2_roc_curve.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  -> Saved h2_roc_curve.png/pdf (300 DPI)")

print("\n[DONE] All figures regenerated at 300 DPI with improved readability.")
