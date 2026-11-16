"""Helpers for the baseline benchmark: MRI-specific SNR / CNR and the accuracy table."""
import numpy as np
import pandas as pd

# Paper Table 2, "Overall accuracy (%)"
PAPER_TABLE2 = {"psnr": 67.01, "ssim": 73.71, "lpips": 78.19, "dists": 76.55, "ilniqe": 68.43, "liqe": 73.71,
                "niqe": 62.01, "brisque": 71.49, "qalign": 72.29, "compare2score": 72.00, "snr": 56.22, "cnr": 50.43}
FULL_REFERENCE = {"psnr", "ssim", "lpips", "dists"}                 # need the clean original image
LOWER_IS_BETTER = {"lpips", "dists", "ilniqe", "niqe", "brisque"}   # their score is flipped before comparing


def snr_cnr(gray):
    """gray: HxW float array in [0,1]. The paper gives no ROI definition, so these are simple automatic versions:
    noise = std of the four image corners (background); foreground = Otsu split of the intensities.
    SNR = mean(foreground) / noise;  CNR = |mean(brightest class) - mean(middle class)| / noise (3-class Otsu).
    Needs scikit-image. Returns (nan, nan) if the image cannot be split into 3 classes."""
    from skimage.filters import threshold_multiotsu
    h, w = gray.shape
    ph, pw = max(2, h // 10), max(2, w // 10)
    bg = np.concatenate([gray[:ph, :pw].ravel(), gray[:ph, -pw:].ravel(),
                         gray[-ph:, :pw].ravel(), gray[-ph:, -pw:].ravel()])
    noise = bg.std() + 1e-3
    try:
        t1, t2 = threshold_multiotsu(gray, classes=3)
        hi, mid = gray[gray > t2], gray[(gray > t1) & (gray <= t2)]
        return float(gray[gray > t1].mean() / noise), float(abs(hi.mean() - mid.mean()) / noise)
    except Exception:
        return float("nan"), float("nan")


def accuracy_table(diffs, names, y, paper=PAPER_TABLE2, mpqa=None, tol=1.5):
    """diffs[name] = per-pair (score_img1 - score_img2), higher = img1 better; y in {-1,0,1}.
    Accuracy on pairs with an expert preference; a NaN or an exact tie counts as wrong."""
    pref = y != 0
    rows = []
    for n in names:
        d = np.asarray(diffs[n], dtype=float)
        acc = (np.sign(d[pref]) == y[pref]).mean() * 100
        rows.append(dict(metric=n, ours=round(acc, 2), paper=paper.get(n, np.nan),
                         delta=round(acc - paper[n], 2) if n in paper else np.nan,
                         match=("yes" if abs(acc - paper[n]) <= tol else "NO") if n in paper else "",
                         failed_pairs=int(np.isnan(d[pref]).sum())))
    if mpqa is not None:
        rows.append(dict(metric="MPQA (this run)", ours=mpqa, paper=82.51, delta=round(mpqa - 82.51, 2),
                         match="", failed_pairs=0))
    return pd.DataFrame(rows)
