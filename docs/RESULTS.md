# Results so far (v0.1)

All numbers are from **single runs (one seed)**. They were produced on Kaggle (2x T4) with the single-file script this
repository was refactored from; the refactored code has been tested on CPU only (see the README, "Status").
Accuracy = accuracy on pairs with an expert preference (see `docs/DEVIATIONS.md` for why this metric).

## Main model: ViT-B, GELU heads, no head dropout, 10 epochs, all training data

Test set = the Kaggle `Labeled_data_test.csv` (2,086 pairs, 1,761 with a preference).

| | Paper | This repo |
|---|---|---|
| Overall | 82.51 | **86.83** |
| SNR | 80.81 | 87.11 |
| Contrast | 66.55 | 80.81 |
| Resolution | 82.43 | 84.32 |
| Artifacts | 82.42 | 86.58 |

Checkpoint selected on validation (best Overall 88.04, within these 10 epochs), so the validation-test gap (about
1.2 points) is the usual selection optimism. The paper trains for 100 epochs; longer schedules have not been run. Caveats: the metric is inferred, the split is Kaggle's (one subject in both CSVs, no shared scan
or image file), and the paper's own pipeline differs in several ways (`docs/DEVIATIONS.md`).

## Baselines on the same test split (`scripts/baselines.py`)

| Method | Paper | This repo |
|---|---|---|
| PSNR | 67.01 | 67.01 |
| SSIM | 73.71 | 73.88 (own Gaussian-window implementation) |
| LPIPS, DISTS, ILNIQE, LIQE, NIQE, BRISQUE | see paper Table 2 | run via `pyiqa`; reported to match the paper, **exact numbers to be added** |
| Q-Align, Compare2Score | 72.29, 72.00 | not run (very large models) |
| SNR, CNR | 56.22, 50.43 | not run; the script uses own simple definitions (the paper gives no ROI definition) |

## Metadata-only baseline (`scripts/sanity_checks.py`)

Logistic regression on the degradation tags in the two filenames only (no pixels): **73.42%** on the 1,761 non-tied
test pairs. So part of what the benchmark measures is *which* degradations were applied; MPQA is 13 points above this
simple baseline. A stronger metadata baseline (tag combinations, parameters) has not been run yet.

## Leakage check

Zero shared original scans, degraded image files or identical pairs between train, validation and test. One subject id
appears in both the train and test CSVs.

## Head-variant smoke test (`scripts/smoke_test.py`; 6 epochs, 25% of the data, validation accuracy, last epoch)

| Variant | SNR | Contrast | Resolution | Artifacts | mean of 4 | Overall |
|---|---|---|---|---|---|---|
| relu, dropout 0.5 (authors' setup) | 42.5 | 10.8 | 13.9 | 19.5 | **21.7** (collapsed) | 73.2 |
| gelu, dropout 0.5 | 67.4 | 80.0 | 45.8 | 59.3 | 63.1 | 61.7 |
| relu, dropout 0 | 80.0 | 66.6 | 86.1 | 71.9 | 76.1 | 82.3 |
| **gelu, dropout 0** | 86.7 | 75.9 | 82.0 | 85.4 | **82.5** | 86.1 |

Mean per-dimension accuracy by epoch for the control: 27.4, 45.0, 36.6, 12.7, 1.3, 21.7 (collapse); for gelu/dropout 0:
83.9, 77.8, 79.4, 81.1, 84.8, 82.5.
