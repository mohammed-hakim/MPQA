# MPQA reproduction

A clean, tested re-implementation of **MPQA**, the multi-dimensional perceptual quality assessment model for MRI from

> Huang et al., *Multi-dimensional perceptual quality assessment for magnetic resonance images*,
> Health Information Science and Systems 13:65 (2025). https://doi.org/10.1007/s13755-025-00382-2
> (authors' code: https://github.com/bymianbao/MPQA, dataset: MRIQP, ~10.5k expert-annotated preference pairs)

Given two degraded MRI slices, the model predicts which one looks better overall and along four dimensions (SNR,
contrast, resolution, artifacts). **Version 0.1 covers the core model and its diagnostics; the paper's other
experiments are on the roadmap below.**

## What v0.1 found

* On the Kaggle test split the model reaches **86.8% overall accuracy** (paper: 82.5%) after only 10 epochs, with all
  four dimensions above the paper's numbers. Single seed; see `docs/RESULTS.md` for the caveats.
* PSNR and SSIM reproduce the paper's baseline numbers (67.01 vs 67.01, 73.88 vs 73.71), which supports the accuracy
  metric used here (accuracy on pairs with an expert preference; the paper does not define it, so this is inferred).
* The authors' head settings (ReLU, dropout 0.5) **collapse the four per-dimension scores** while the overall score
  still looks fine. Removing the head dropout fixes it (`docs/DEVIATIONS.md`, `scripts/smoke_test.py`).
* A baseline that sees **only the degradation tags in the filenames** already reaches 73.4%, so part of the benchmark
  measures *which* degradations were applied (`scripts/sanity_checks.py`).

## Quickstart

**On Kaggle:** open `notebooks/kaggle_run.ipynb` (upload it, or open it from the cloned repo), edit the first code
cell and run the cells top to bottom. It covers setup, sanity checks, the smoke test, training with resume, evaluation,
diagnostics, training curves and the baselines. See `docs/KAGGLE.md`.

**Anywhere else:**

```bash
git clone https://github.com/<your-user>/mpqa-repro.git && cd mpqa-repro
pip install -r requirements.txt
python main.py train    --set data_dir=/path/to/LabelMRI4 --set out_dir=runs/main
python main.py evaluate --ckpt runs/main/best.pth --set data_dir=/path/to/LabelMRI4
python main.py --help   # all commands: train, evaluate, smoke, baselines, diagnose, sanity, clean-test
```

`data_dir` must contain the two CSVs (`Labeled_data_train.csv`, `Labeled_data_test.csv`) and the `noisedImages/` folder
(plus `originalImages/` for the full-reference baselines). Kaggle instructions: `docs/KAGGLE.md`.

## Configuration

All hyperparameters are in `configs/default.yaml`. Override any of them on the command line, and typos are errors:

```bash
python main.py train --set epochs=100 --set act=relu --set head_drop=0.5 --set batch_size=4
```

Training resumes automatically from `<out_dir>/last.pth`; `--set resume_from=PATH` continues from another run.

## Commands

`python main.py <command>` runs the matching script in `scripts/` (you can also run the scripts directly).

| Command (script) | What it does |
|---|---|
| `train` (`scripts/train.py`) | train with per-epoch validation, then test the best checkpoint |
| `evaluate` (`scripts/evaluate.py`) | evaluate `best.pth` / `last.pth` on the test or validation split |
| `baselines` (`scripts/baselines.py`) | the paper's Table 2 baselines (PSNR, SSIM, LPIPS, DISTS, ILNIQE, LIQE, NIQE, BRISQUE, SNR, CNR) via `pyiqa` |
| `smoke` (`scripts/smoke_test.py`) | 2x2 head-variant test (activation x dropout); runs two variants in parallel on 2 GPUs |
| `diagnose` (`scripts/diagnose.py`) | per-dimension collapse check of a trained checkpoint (fp16 vs fp32) |
| `sanity` (`scripts/sanity_checks.py`) | leakage between splits and the metadata-only baseline (CPU only) |
| `clean-test` (`scripts/clean_test.py`) | write a test CSV without subjects that also appear in the train CSV |

## Layout

```
main.py                  single entry point: python main.py <command>
notebooks/kaggle_run.ipynb   run the whole pipeline on Kaggle, cell by cell
configs/default.yaml     every hyperparameter
mpqa/
  config.py              YAML + key=value overrides
  labels.py              CSV labels, degradation tags, subject-level split (no torch)
  data.py                PairDataset, loaders
  model.py               backbone + two heads, overall score
  losses.py              ZLPR, ranking term, joint loss (paper Eqs. 4-8)
  metrics.py             accuracy on preferred / all pairs
  trainer.py             fit / evaluate / checkpoint / resume / multi-GPU
  analysis.py            leakage report, metadata-only baseline
  baselines.py           SNR / CNR and the accuracy table
scripts/                 command-line entry points (above)
tests/                   33 tests, CPU only
docs/                    DEVIATIONS.md, RESULTS.md, KAGGLE.md
```

## Status and tests

* `pytest` runs 33 tests in about two minutes on a CPU, with no dataset and no downloads: the paper's formulas (ZLPR, the
  three branches of the ranking loss), label parsing, the subject-level split, configuration, and an end-to-end
  train / checkpoint / resume / evaluate run with a tiny ViT on fake images.
* Every script, `main.py` and the Kaggle notebook were run end to end on fake data (the baselines with a stand-in for `pyiqa`).
* **GPU paths are not covered by the tests** (fp16 autocast, `DataParallel`, the 2-GPU smoke-test launcher). The
  results in `docs/RESULTS.md` come from the single-file script this repository was refactored from, on Kaggle 2x T4.
  Re-running `scripts/train.py` on a GPU to confirm the same behaviour is the first item on the roadmap.
* Tested with Python 3.11, torch 2.14, timm 1.0, pandas 3.1, scikit-learn 1.9, scikit-image 0.26 (CPU build).

## Known limitations

* Single seed, no confidence intervals yet.
* The model rarely predicts "equal": accuracy on all pairs including ties is only about 48% (the paper's metric ignores ties).
* SNR / CNR in `scripts/baselines.py` use my own simple definitions because the paper gives no region definition.
* One subject appears in both the train and test CSVs (no shared scan or image file); `drop_shared_subjects=true` removes it.
* The dataset-generation code (HCP data and the seven degradation operations) is out of scope; the released MRIQP is used.

## Roadmap

1. Re-run the full pipeline on GPU from this repository and add the exact numbers to `docs/RESULTS.md`.
2. Multiple seeds (mean +- std) and the full-data head ablation.
3. The paper's other tables: encoder comparison (Table 3), training-data size (4), lambda sweep (5), loss ablation (6).
4. Dataset statistics (Fig. 2), case study (Fig. 4), a stronger metadata-only baseline, handling of "equal" pairs.
