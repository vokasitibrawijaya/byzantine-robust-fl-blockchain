# Revision R2 Evidence Audit

## Submission Artifact

Upload this single Word file:

`revision_ETASR_APRIL2026/18579_REVISION_R2_ETASR_FINAL.docx`

It contains reviewer comments and answers first, followed by the revised
manuscript. The document uses Letter page size, Times New Roman, one-column
response/front matter, and a two-column manuscript body.

## Claim-to-Evidence Map

| Manuscript claim | Evidence |
|---|---|
| Clean FedAvg 82.49% | `results/unified_mnist_actual.json`, `summary.clean_fedavg` |
| Attacked FedAvg 22.36% | `summary.fedavg` |
| Krum 60.01% | `summary.krum` |
| TrimmedMean 78.23% | `summary.trimmed_mean` |
| ATMA 78.15% | `summary.atma` |
| ATMA vs. TrimmedMean `p=0.706` | `results/unified_mnist_analysis.json`, `paired_tests.atma_minus_trimmed_mean` |
| ATMA precision/recall/F1 | `atma_detection` in the analysis JSON |
| Krum selected no Byzantine client | `krum_selection` in the analysis JSON |
| 420 successful audit transactions | `blockchain` in the analysis JSON |
| 24,415,241 measured gas | `blockchain.total_gas_used_including_deployment` |
| Contract read-back passed | `blockchain.all_readback_verifications_passed` |

## Blockchain Identifiers

- Chain ID: `1337`
- Contract: `0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab`
- Deployment transaction:
  `fbc2f910da9de1012c0ec4e46fbdb519c848fb542600d6e788d85ba452c8bef7`
- Deployment block: `1`
- Final block: `421`

## Removed Unsupported Claims

Revision R2 does not claim:

- Transparency Paradox resolution
- FLARE experimental validation
- CIFAR-10 results from legacy scripts
- FedProx/FedDyn superiority or collapse
- live Layer-2 deployment
- mainnet dollar costs
- decentralized consensus validation
- production readiness
- 1,000-client FL scalability

## Verification Performed

- Robust aggregation unit tests: 3 passed.
- LaTeX compilation: successful, no overfull boxes or unresolved references.
- PDF: Letter page size, five pages.
- Word XML:
  - three sections: one column, one column, two columns
  - Letter page size
  - 59 native Office Math objects
  - no literal LaTeX commands
  - Times New Roman, no Aptos
  - two 300-DPI result figures constrained to column width
