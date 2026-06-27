# Byzantine-Robust Federated Learning with Blockchain Auditability

This repository contains the reproducible experiment used for manuscript
revision R4:

**"Byzantine-Robust Federated Learning with Adaptive Aggregation and
Blockchain: Empirical Validation of ATMA and Resolution of the Transparency
Paradox"**

## Final R4 Experiment

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
| Equal-client mean (attack) | 31.94% | 7.06 | [14.41, 49.48] |
| Krum | 60.01% | 7.37 | [41.71, 78.32] |
| TrimmedMean | 78.23% | 1.51 | [74.47, 81.99] |
| MAD-ATMA | 78.22% | 1.98 | [73.29, 83.14] |

No statistically significant final-accuracy difference was detected between
MAD-ATMA and TrimmedMean (`p=0.967`, three paired seeds). MAD-ATMA uses
distance-to-median MAD scores and adapts its trim ratio inside a prespecified
5-25% safety interval. Its contribution in this experiment is adaptation and
anomaly flagging, not higher final accuracy.

The R4 manuscript cites 20 references in order of first appearance. Eight added
journal references are from 2021-2025 and cover trustworthy FL, Byzantine-robust
secure aggregation, decentralized FL, and blockchain-enabled FL.

Canonical FedAvg is weighted by full client partition size, while the robust
rules are client-symmetric. Because each client performs the same three
mini-batches per round, an attacked equal-client mean was run as a sensitivity
baseline. It improves the undefended result by 9.58 percentage points
(`p=0.070`) but remains below all three robust methods.

## Transparency Paradox Experiment

For every seed, the blind and ledger-informed conditions use the same MNIST
partition, CNN, four Byzantine clients, MAD-ATMA settings, attack
initialization, local batches, rounds, and fresh Ganache audit contract. The
only experimental difference is access to the previous round's committed
Byzantine-client flags:

- blind attacker: fixed attack scale 5.0 and no ledger reads
- ledger-informed attacker: 228 flag queries, implemented as 456 contract view
  calls including existence checks, and a prespecified feedback controller
  that adapts the attack scale inside [1.0, 5.0]

Measured across seeds 42, 43, and 44:

| Condition | Final accuracy | Detection recall | Undetected | Mean scale |
|---|---:|---:|---:|---:|
| Blind | 78.22% +/- 1.98 | 100.0% | 0.0% | 5.00 |
| Ledger-informed | 76.59% +/- 1.65 | 47.5% | 52.5% | 1.63 |

Ledger information reduces final accuracy by 1.62 percentage points
(`p=0.0145`) and leaves 126 of 240 Byzantine client-round updates undetected.
Across six contracts, all 2,400 client commitments and 120 round summaries pass
read-back verification. The manuscript therefore treats the Transparency
Paradox as a bounded tradeoff: immediate public flags aid evasion, while
write-once logging preserves evidence.

## Blockchain Evidence

The MAD-ATMA seed-42 run deploys the write-once `FLAudit.sol` contract to
Ganache chain 1337 and records:

- 400 client-update hashes
- 20 round summaries
- 420 successful post-deployment transactions
- 34,647,901 measured gas including deployment
- successful read-back of every client record and complete round summary
- overwrite guards for existing client records and finalized rounds

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

Run the equal-client sensitivity baseline:

```powershell
python -m src.experiments.unified_mnist_blockchain_experiment `
  --rounds 20 --max-batches 3 --seeds 42 43 44 `
  --methods fedavg_equal `
  --output results/unified_mnist_equal_weight_sensitivity.json
```

Run the controlled Transparency Paradox experiment:

```powershell
python -m src.experiments.transparency_paradox_experiment `
  --rounds 20 --max-batches 3 --seeds 42 43 44 `
  --output results/transparency_paradox_actual.json
```

Regenerate statistics and figures:

```powershell
python -m src.experiments.analyze_unified_revision
python -m src.experiments.analyze_transparency_paradox
```

Run aggregation tests:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Main Artifacts

- `results/unified_mnist_actual.json`: raw final runs and transaction receipts
- `results/unified_mnist_equal_weight_sensitivity.json`: equal-client baseline
- `results/unified_mnist_analysis.json`: calculated statistics
- `results/transparency_paradox_actual.json`: blind/informed raw runs and chain
  evidence
- `results/transparency_paradox_analysis.json`: transparency statistics
- `18579_R4_REPRODUCIBILITY_SUPPLEMENT.zip`: exact R4 supplementary archive
- `src/aggregation/robust_aggregation.py`: corrected Krum and MAD-ATMA
- `src/experiments/transparency_paradox_experiment.py`: controlled ledger
  feedback experiment
- `src/blockchain/contracts/FLAudit.sol`: audit contract
- `visualizations/revision_actual/`: generated figures and LaTeX tables
- `revision_ETASR_APRIL2026/18579_REVISION_R4_ETASR_FINAL.docx`: submission file
- `revision_ETASR_APRIL2026/18579_REVISION_R4_EVIDENCE_BASED.pdf`: review PDF

Files and results from earlier development iterations are retained only in Git
history. The `main` branch is intentionally limited to the R4 submission
artifacts and reproducibility files.
