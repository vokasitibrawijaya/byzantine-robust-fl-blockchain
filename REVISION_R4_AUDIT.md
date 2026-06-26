# Revision R4 Evidence Audit

## Submission Artifact

Upload:

`revision_ETASR_APRIL2026/18579_REVISION_R4_ETASR_FINAL.docx`

The matching review PDF is:

`revision_ETASR_APRIL2026/18579_REVISION_R4_EVIDENCE_BASED.pdf`

The exact code-and-results supplement is:

`18579_R4_REPRODUCIBILITY_SUPPLEMENT.zip`

Current artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `revision_ETASR_APRIL2026/18579_REVISION_R4_ETASR_FINAL.docx` | `BCA77A52C0566398516C4541BA560E39D81CA4EC2352D43130A72944B666EE76` |
| `revision_ETASR_APRIL2026/18579_REVISION_R4_EVIDENCE_BASED.pdf` | `18CA68738860312B87B769E99E08B065C86BAC3726AC44CE26270A8E81CE4CBB` |

## R4 Corrections

- Added the current ETASR declarations:
  - Declaration of Competing Interests
  - Acknowledgment
  - Data Availability
  - AI Use and Declaration of Generative AI Use
- Added and explicitly distinguished Nishimoto et al.'s FedATM (2023).
- Restricted the novelty statement to the implemented MAD detector-to-trim
  mapping and measured blockchain audit integration.
- Explained that canonical FedAvg is partition-size weighted while the robust
  aggregators are client-symmetric.
- Added a separately labeled attacked equal-client mean sensitivity baseline.
- Restored the original manuscript title while explicitly limiting empirical
  validation claims to the executed MAD-ATMA variant and bounded Transparency
  Paradox protocol.
- Added the missing explicit MAD definition and the implementation meaning of
  the epsilon term in the robust-score equation.
- Made the FedAvg denominator explicit as `sum_{j=1}^{N} n_j`.
- Numbered all native Word display equations in the final DOCX.
- Added a controlled blind-versus-ledger-informed attacker experiment in which
  training, attack initialization, and blockchain logging remain identical.
- Expanded the bibliography to 20 cited references.
- Added eight relevant 2021-2025 open-access journal references.
- Ordered references exactly by first appearance in the manuscript.
- Updated Rachmad Andri Atmoko to affiliations 1 and 2 and marked him as the
  corresponding author.
- Updated the Acknowledgment to thank the Electrical Engineering Department,
  Faculty of Engineering, Universitas Brawijaya, and the Laboratory of Internet
  of Things and Human Centered Design, Faculty of Vocational Studies,
  Universitas Brawijaya, for computational resources and supercomputer support.
- Removed the otherwise blank "REVISED MANUSCRIPT FOLLOWS" PDF page.

## Weighting Sensitivity Evidence

Source:

`results/unified_mnist_equal_weight_sensitivity.json`

| Quantity | Value |
|---|---:|
| Seed accuracies | 38.34%, 24.37%, 33.12% |
| Mean accuracy | 31.94% |
| Sample SD | 7.06 |
| 95% CI | [14.41%, 49.48%] |
| Gain over sample-weighted attacked FedAvg | 9.58 percentage points |
| Paired p-value vs. sample-weighted attacked FedAvg | 0.070 |

Krum, TrimmedMean, and MAD-ATMA remain above the equal-client mean by 28.07,
46.29, and 46.27 percentage points, respectively. Thus, weighting changes the
reported effect size but not the ordering of the evaluated defenses.

## Transparency Paradox Evidence

Sources:

- `results/transparency_paradox_actual.json`
- `results/transparency_paradox_analysis.json`

| Quantity | Blind | Ledger-informed |
|---|---:|---:|
| Final accuracy | 78.22% +/- 1.98 | 76.59% +/- 1.65 |
| Byzantine detection recall | 100.0% | 47.5% |
| Undetected Byzantine updates | 0/240 | 126/240 |
| Mean attack scale | 5.00 | 1.63 |

The informed-minus-blind final-accuracy difference is -1.62 percentage points
(`p=0.0145`; seedwise differences -1.36, -2.01, and -1.50). The informed
condition performs 228 prior-flag queries, implemented as 456 contract view
calls including existence checks. Six fresh contracts preserve 2,400 client
commitments and 120 round summaries in 2,520 successful post-deployment
transactions; all read-back checks pass.

The manuscript calls this a bounded empirical resolution, not a universal
claim: public flags materially aid evasion, but the attacker cannot erase or
overwrite the resulting audit evidence.

## Verification

- Equal-client sensitivity run completed for seeds 42, 43, and 44.
- Transparency Paradox run completed for blind and ledger-informed conditions
  for seeds 42, 43, and 44.
- Robust aggregation and controller tests: 6 passed.
- LaTeX: successful, no overfull boxes or unresolved citations/references.
- PDF: US Letter, 9 pages including the one-page reviewer response.
- Word:
  - native ETASR styles
  - three sections with one-column response/front matter and two-column body
  - 90 native Office Math objects
  - 12 numbered display equations
  - three embedded figures and five tables
  - no literal LaTeX commands
  - numbered in-text citations and supplementary filename preserved
  - 20 references with sequential first appearance from [1] to [20]
  - affiliations 1 and 2 assigned to Rachmad Andri Atmoko
  - Rachmad Andri Atmoko marked as corresponding author
  - declarations, FedATM citation, and sensitivity analysis present
