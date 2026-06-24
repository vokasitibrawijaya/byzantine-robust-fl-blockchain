# Revision R3 Evidence Audit

## Submission Artifact

Upload this single Word file:

`revision_ETASR_APRIL2026/18579_REVISION_R3_ETASR_FINAL.docx`

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
| MAD-ATMA 78.22% | `summary.atma` |
| MAD-ATMA vs. TrimmedMean `p=0.967` | `results/unified_mnist_analysis.json`, `paired_tests.atma_minus_trimmed_mean` |
| MAD-ATMA precision/recall/F1 | `atma_detection` in the analysis JSON |
| Krum selected no Byzantine client | `krum_selection` in the analysis JSON |
| 420 successful audit transactions | `blockchain` in the analysis JSON |
| 34,647,901 measured gas | `blockchain.total_gas_used_including_deployment` |
| Client and summary read-back passed | `blockchain.all_readback_verifications_passed` |

## Blockchain Identifiers

- Chain ID: `1337`
- Contract: `0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab`
- Deployment transaction:
  `fb94cfc4d18b60d70bad58f3ddc979537c2fd6147e710c7a3464ed40b1016372`
- Deployment block: `1`
- Final block: `421`

## Removed Unsupported Claims

Revision R3 does not claim:

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

- Robust aggregation unit tests: 4 passed.
- Contract overwrite checks: client and round duplicate writes revert.
- LaTeX compilation: successful, no overfull boxes or unresolved references.
- PDF: Letter page size, six total pages including the reviewer response.
- Word XML:
  - three sections: one column, one column, two columns
  - Letter page size
  - native ETASR title, author, affiliation, heading, body, caption, and reference styles
  - six separate author/affiliation blocks
  - 71 native Office Math objects
  - no literal LaTeX commands
  - no footnotes
  - Times New Roman, no Aptos
  - two 300-DPI result figures constrained to column width
