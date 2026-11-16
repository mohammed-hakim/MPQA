#!/usr/bin/env python
"""Benchmark the paper's Table 2 baselines on the test pairs and compare with the paper.

Same scoring rule as MPQA: on pairs with an expert preference, the image with the better score is the prediction.
Full-reference metrics (PSNR, SSIM, LPIPS, DISTS) compare each degraded image with its clean original; no-reference
ones (ILNIQE, LIQE, NIQE, BRISQUE) score the image alone; SNR / CNR are simple definitions of my own (see
mpqa/baselines.py). Needs: pip install pyiqa scikit-image   (and internet for the weights).

    python scripts/baselines.py --n-pairs 100            # quick trial
    python scripts/baselines.py --mpqa-score 86.83       # full run, adds the MPQA row
    python scripts/baselines.py --heavy                  # also Q-Align / Compare2Score (very large; untested on a T4)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpqa.baselines import FULL_REFERENCE, LOWER_IS_BETTER, PAPER_TABLE2, accuracy_table, snr_cnr  # noqa: E402
from mpqa.cli import add_config_args, cfg_from_args  # noqa: E402
from mpqa.data import load_frames  # noqa: E402
from mpqa.labels import IMG1_COL, IMG2_COL, ORIG_COL, signed_labels  # noqa: E402


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    parser.add_argument("--n-pairs", type=int, default=None, help="score only N random test pairs (quick trial)")
    parser.add_argument("--size", type=int, default=None, help="resize every image to SIZE x SIZE first (default: native)")
    parser.add_argument("--heavy", action="store_true", help="also try Q-Align and Compare2Score")
    parser.add_argument("--mpqa-score", type=float, default=None, help="your MPQA test accuracy, to add a row")
    parser.add_argument("--tol", type=float, default=1.5, help="|ours - paper| within this many points counts as a match")
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    import pyiqa
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test = load_frames(cfg, verbose=False)
    if args.n_pairs:
        test = test.sample(min(args.n_pairs, len(test)), random_state=0).reset_index(drop=True)
    y = signed_labels(test[7].values)
    print(f"device: {dev} | test pairs: {len(test)} ({int((y != 0).sum())} with a preference)")

    names = ["psnr", "ssim", "lpips", "dists", "ilniqe", "liqe", "niqe", "brisque"] + (["qalign", "compare2score"] if args.heavy else [])
    metrics, errors = {}, {}
    for n in names:
        try:
            metrics[n] = pyiqa.create_metric(n, device=dev)
            print(f"loaded {n}")
        except Exception as e:
            errors[n] = f"could not load: {type(e).__name__}: {str(e)[:150]}"
            print(f"SKIP {n}: {errors[n]}")

    def load_rgb(path):
        im = Image.open(path).convert("RGB")
        if args.size:
            im = im.resize((args.size, args.size), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]

    cache, need_ref = {}, any(n in FULL_REFERENCE for n in metrics)

    def score_image(orig, deg):
        if (orig, deg) in cache:
            return cache[(orig, deg)]
        dist = load_rgb(Path(cfg.data_dir) / cfg.img_dir / deg)
        ref = load_rgb(Path(cfg.data_dir) / cfg.orig_dir / orig) if need_ref else None
        s = {}
        with torch.no_grad():
            for n, m in metrics.items():
                try:
                    v = m(dist.to(dev), ref.to(dev)) if n in FULL_REFERENCE else m(dist.to(dev))
                    v = float(v.item() if hasattr(v, "item") else v)
                    s[n] = -v if n in LOWER_IS_BETTER else v
                except Exception as e:
                    s[n] = float("nan")
                    errors.setdefault(n, f"{type(e).__name__}: {str(e)[:150]}")
        s["snr"], s["cnr"] = snr_cnr(dist[0].mean(0).numpy())
        cache[(orig, deg)] = s
        return s

    t0 = time.time()
    diffs = {n: [] for n in list(metrics) + ["snr", "cnr"]}
    for r in tqdm(test.values, desc="scoring pairs"):
        a = score_image(str(r[ORIG_COL]), str(r[IMG1_COL]))
        b = score_image(str(r[ORIG_COL]), str(r[IMG2_COL]))
        for n in diffs:
            diffs[n].append(a[n] - b[n])

    order = [n for n in PAPER_TABLE2 if n in diffs]
    table = accuracy_table(diffs, order, y, mpqa=args.mpqa_score, tol=args.tol)
    print(f"\nscoring took {(time.time() - t0) / 60:.1f} min  (accuracy on pairs with an expert preference, %)\n")
    print(table.to_string(index=False))
    missing = [n for n in PAPER_TABLE2 if n not in diffs]
    if missing:
        print("\nnot run:", missing)
    for k, v in errors.items():
        print(f"problem with {k}: {v}")
    print("\nNote: SNR / CNR use simple definitions of my own (the paper gives no ROI definition), so they are approximate.")


if __name__ == "__main__":
    main()
