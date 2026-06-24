# Byzantine-Robust Federated Learning with Blockchain Auditability

This repository contains the reproducible experiment used for manuscript
revision R2:

**"Byzantine-Robust Federated Learning with Adaptive Aggregation and
Blockchain: A Reproducible Empirical Evaluation"**

## Final R2 Experiment

All compared methods use one configuration:

| Parameter | Value |
|---|---|
| Dataset | MNIST |
| Model | CNN, 105,866 parameters |
| Clients | 20, full participation |
| Byzantine clients | IDs 0-3 (20%) |
| Partition | Dirichlet alpha=0.5 |
| Attack | Cyclic label flip plus 5x model-delta scaling |
| Rounds | 20 |
| Local work | 3 mini-batch SGD steps/client/round |
| Seeds | 42, 43, 44 |

Final test accuracy:

| Method | Mean | Sample SD | 95% CI |
|---|---:|---:|---:|
| FedAvg (clean) | 82.49% | 1.29 | [79.28, 85.69] |
| FedAvg (attack) | 22.36% | 2.81 | [15.39, 29.33] |
| Krum | 60.01% | 7.37 | [41.71, 78.32] |
| TrimmedMean | 78.23% | 1.51 | [74.47, 81.99] |
| ATMA | 78.15% | 1.80 | [73.68, 82.62] |

ATMA and TrimmedMean have statistically indistinguishable final accuracy
(`p=0.706`). ATMA's contribution in this experiment is automatic trim-ratio
adaptation and anomaly flagging, not higher final accuracy.

## Blockchain Evidence

The ATMA seed-42 run deploys `FLAudit.sol` to Ganache chain 1337 and records:

- 400 client-update hashes
- 20 round summaries
- 420 successful post-deployment transactions
- 24,415,241 measured gas including deployment
- successful contract read-back verification for every stored update

These are local EVM audit measurements. They are not mainnet cost, Layer-2,
latency, decentralization, or production-readiness claims.

## Reproduce

Start Ganache:

```powershell
ganache --server.host 127.0.0.1 --server.port 8545 `
  --wallet.deterministic --wallet.totalAccounts 25 `
  --chain.chainId 1337 --logging.quiet
```

Compile the contract:

```powershell
npx --yes solc@0.8.19 --bin --abi `
  src/blockchain/contracts/FLAudit.sol `
  -o artifacts/blockchain/fl_audit
```

Run the experiment:

```powershell
python -m src.experiments.unified_mnist_blockchain_experiment `
  --rounds 20 --max-batches 3 --seeds 42 43 44 `
  --methods clean_fedavg fedavg krum trimmed_mean atma `
  --blockchain --output results/unified_mnist_actual.json
```

Regenerate statistics and figures:

```powershell
python -m src.experiments.analyze_unified_revision
```

Run aggregation tests:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Main Artifacts

- `results/unified_mnist_actual.json`: raw final runs and transaction receipts
- `results/unified_mnist_analysis.json`: calculated statistics
- `src/aggregation/robust_aggregation.py`: corrected Krum and ATMA
- `src/blockchain/contracts/FLAudit.sol`: audit contract
- `visualizations/revision_actual/`: generated figures and LaTeX tables
- `revision_ETASR_APRIL2026/18579_REVISION_R2_ETASR_FINAL.docx`: submission file
- `revision_ETASR_APRIL2026/18579_REVISION_R2_EVIDENCE_BASED.pdf`: review PDF

Files and results from earlier development iterations remain for provenance but
are not used as evidence in revision R2.
