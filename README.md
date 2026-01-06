# Byzantine-Robust Federated Learning with Blockchain

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://doi.org/10.1109/ACCESS.2026.XXXXXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://python.org)

## Paper Supplementary Materials

This repository contains the experimental code, results, and artifacts for:

**"Byzantine-Robust Federated Learning with Adaptive Aggregation and Blockchain: Empirical Validation and Resolution of the Transparency Paradox"**

*Submitted to IEEE Access (Major Revision)*

---

## 📁 Repository Structure

```
├── src/                              # Source code
│   ├── aggregation/                  # Aggregation algorithms (ATMA, TrimmedMean, Krum, etc.)
│   ├── experiments/                  # Main experiment scripts
│   │   ├── h2_valid_rerun.py        # H2: ROC/AUC provenance detection
│   │   ├── real_fl_training_FIXED.py # Real FL training on MNIST
│   │   ├── run_cifar10_blockchain_simple.py  # CIFAR-10 experiments
│   │   └── ...
│   ├── blockchain/                   # Smart contracts & blockchain integration
│   └── utils/                        # Utility functions
├── results/                          # Experiment results (JSON)
│   ├── h2_valid_rerun_results.json   # Table II, Figure 2
│   ├── real_fl_training_FIXED.json   # Table VI
│   ├── cifar10_blockchain_simple_*.json  # Table XII
│   ├── multiseed_comparison_*.json   # Table XIII
│   └── blockchain_cost_benefit_analysis.json  # Table XIV
├── visualizations/                   # Generated figures
├── ACCESS_latex_template_20240429/   # Paper LaTeX source
├── ARTIFACT_MANIFEST.md              # Detailed artifact documentation
└── MAJOR_REVISION_RESPONSE.md        # Revision changelog
```

---

## 📊 Key Results Summary

### CIFAR-10 Validation (Dirichlet α=0.5, 20% Byzantine)

| Method | 50 Rounds | 160 Rounds | Status |
|--------|-----------|------------|--------|
| **TrimmedMean** | 66.38% | **67.92%** | Best |
| ATMA | 64.38% | 65.78% | Competitive |
| Krum | 36.71% | 43.41% | Moderate |
| FedAvg | 10.0% | 10.0% | Collapsed |
| FedProx | 10.0% | 10.0% | Not Byzantine-robust |
| FedDyn | 10.0% | 10.0% | Not Byzantine-robust |

### Provenance Detection (H2)

| Metric | Blockchain | Centralized |
|--------|------------|-------------|
| **AUC** | **0.957** | 0.000 |
| TPR | 81.0% | 0% |
| FPR | 0.0% | 0% |

### Real FL on MNIST (20 rounds, seeds 42-44)

| Method | Accuracy |
|--------|----------|
| TrimmedMean | 73.00% ± 3.39% |
| Median | 71.48% ± 4.52% |
| Mean | 66.95% ± 3.55% |

---

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/vokasitibrawijaya/byzantine-robust-fl-blockchain.git
cd byzantine-robust-fl-blockchain

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy scikit-learn web3 matplotlib
```

---

## 🚀 Running Experiments

### H2: Provenance Detection (ROC/AUC)
```bash
python src/experiments/h2_valid_rerun.py
# Output: results/h2_valid_rerun_results.json
```

### Real FL Training on MNIST
```bash
python src/experiments/real_fl_training_FIXED.py
# Output: results/real_fl_training_FIXED.json
```

### CIFAR-10 with Byzantine Attacks
```bash
python src/experiments/run_cifar10_blockchain_simple.py
# Output: results/cifar10_blockchain_simple_*.json
```

### Multi-seed Confidence Intervals
```bash
python src/experiments/quick_multiseed_fedprox_feddyn.py
# Output: results/multiseed_comparison_*.json
```

---

## 📋 Experiment Configuration

| Experiment | Dataset | Attack | Byzantine | Rounds | Seeds |
|------------|---------|--------|-----------|--------|-------|
| Table II | MNIST | Tamper | 30% prob | 50 | 42 |
| Table VI | MNIST | Sign-flip | 4/20 | 20 | 42,43,44 |
| Table XII | CIFAR-10 | Label-flip(λ=-5) | 4/20 | 50-160 | 42 |
| Table XIII | CIFAR-10 | Label-flip(λ=-5) | 4/20 | 50 | 42,43,44 |

**Note:** Table XIII uses reduced hyperparameters (epochs=3, batch=256) for rapid multi-seed evaluation, explaining the 34.62% vs 66.38% accuracy difference from Table XII.

---

## 📄 Citation

```bibtex
@article{author2026byzantine,
  title={Byzantine-Robust Federated Learning with Adaptive Aggregation and Blockchain},
  author={[Author Names Withheld for Blind Review]},
  journal={IEEE Access},
  year={2026},
  note={Under Review}
}
```

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Related Resources

- Paper PDF: [paper/federated_learning_paper.pdf](paper/federated_learning_paper.pdf)
- Aggregation algorithms: [src/aggregation/](src/aggregation/)
- Experiment scripts: [src/experiments/](src/experiments/)
- All results: [results/](results/)

---

## 📊 Latest Results (January 2026)

### Byzantine Degradation Analysis (CIFAR-10, 30% Byzantine Attack)

| Method | Clean Baseline | Under Attack | Degradation |
|--------|----------------|--------------|-------------|
| FedAvg | 59.13% | 54.99% | **7.0%** |
| FedProx | 60.07% | 56.67% | **5.7%** |
| TrimmedMean | 56.68% | 53.19% | **6.2%** |

*Experimental setup: NVIDIA RTX 5060 Ti GPU, PyTorch 2.9.1+cu128, seed=42*

---

*Last Updated: January 6, 2026*
