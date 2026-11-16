import numpy as np
import pytest

pytest.importorskip("skimage")

from mpqa.baselines import accuracy_table, snr_cnr  # noqa: E402


def _head(rng, sd):
    yy, xx = np.mgrid[:128, :128]
    r = np.hypot(yy - 64, xx - 64)
    img = np.zeros((128, 128)); img[r < 45] = 0.55; img[r < 25] = 0.85
    return np.clip(img + rng.normal(0, sd, img.shape), 0, 1)


def test_snr_cnr_fall_as_noise_rises():
    rng = np.random.default_rng(0)
    vals = [snr_cnr(_head(rng, sd)) for sd in (0.01, 0.05, 0.15)]
    assert vals[0][0] > vals[1][0] > vals[2][0] and vals[0][1] > vals[1][1] > vals[2][1]


def test_accuracy_table_logic():
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.choice([-1.0, 0.0, 1.0], n, p=[.42, .16, .42])
    nan_scores = y + rng.normal(0, 1.2, n); nan_scores[:50] = np.nan
    t = accuracy_table({"psnr": y + rng.normal(0, 1.2, n), "ssim": rng.normal(0, 1, n),
                        "lpips": -y + rng.normal(0, 0.5, n), "niqe": nan_scores},
                       ["psnr", "ssim", "lpips", "niqe"], y, mpqa=86.83).set_index("metric")
    assert t.loc["psnr", "ours"] > 75                       # informative metric
    assert 45 < t.loc["ssim", "ours"] < 58                  # chance
    assert t.loc["lpips", "ours"] < 10                      # inverted direction is obvious
    assert t.loc["niqe", "failed_pairs"] > 0                # NaNs are counted, as wrong
    assert t.loc["MPQA (this run)", "delta"] == pytest.approx(4.32)
