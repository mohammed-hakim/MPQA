# MPQA reproduction (v1)

Re-implementation of **MPQA**, a multi-dimensional perceptual quality model for MRI:

> Huang et al., *Multi-dimensional perceptual quality assessment for magnetic resonance images*,
> Health Information Science and Systems 13:65 (2025). https://doi.org/10.1007/s13755-025-00382-2

Given two degraded MRI slices, it predicts which one looks better overall and for SNR, contrast, resolution and artifacts.

## Result

ViT-B encoder, 10 epochs, all training data, one run (accuracy on pairs with an expert preference).

| | Paper | v1 |
|---|---|---|
| Overall, test | 82.51% | 88.04% |
| SNR / Contrast / Resolution / Artifacts, test | 80.81 / 66.55 / 82.43 / 82.42 | 87.11 / 80.81 / 84.32 / 86.58 |

Notes:
* The accuracy metric is inferred, not defined in the paper: it matches the paper's PSNR baseline (67.01) exactly.
* The authors' head settings (ReLU, dropout 0.5) collapsed the four per-dimension scores in our runs. v1 uses GELU and no head dropout. See `docs/DEVIATIONS.md`.
* Single seed. The GPU code paths were not re-run after the refactor (the tests run on CPU).

## Run

```bash
pip install -r requirements.txt
python main.py train    --set data_dir=/path/to/LabelMRI4 --set out_dir=runs/main
python main.py evaluate --ckpt runs/main/best.pth --set data_dir=/path/to/LabelMRI4
```

`data_dir` needs the two CSVs and the `noisedImages/` folder. Settings are in `configs/default.yaml` and can be overridden with `--set key=value`. Training resumes from `<out_dir>/last.pth`.

On Kaggle, use `notebooks/kaggle_run.ipynb` (see `docs/KAGGLE.md`).

Other commands: `smoke` (head-variant test), `baselines` (PSNR, SSIM, LPIPS, ...), `diagnose`, `sanity`, `clean-test`. Run `python main.py --help`.

## Tests

`pytest` (CPU, about a minute, no dataset needed).

## Next

The paper's other experiments (encoder comparison, data size, lambda sweep, loss ablation), several seeds.
