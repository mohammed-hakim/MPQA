#!/usr/bin/env python
"""Per-dimension collapse check for a trained checkpoint.
Do the SNR / Contrast / Resolution / Artifacts scores actually differ between the two images of a pair, or have they
collapsed to (almost) the same value? Also compares fp16 and fp32 inference.

    python scripts/diagnose.py --ckpt runs/default/best.pth --n 500
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpqa.cli import add_config_args, cfg_from_args  # noqa: E402
from mpqa.data import load_frames, make_loader  # noqa: E402
from mpqa.labels import DIMS  # noqa: E402
from mpqa.model import overall_vector  # noqa: E402
from mpqa.trainer import build_model, get_device, load_weights  # noqa: E402


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    parser.add_argument("--ckpt", default=None, help="default: <out_dir>/best.pth")
    parser.add_argument("--n", type=int, default=0, help="use N random test pairs (0 = all)")
    parser.add_argument("--modes", default="fp16,fp32", help="comma-separated: fp16, fp32")
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    dev = get_device()
    print("device:", dev, flush=True)
    core, model = build_model(cfg, dev, pretrained=False)
    load_weights(core, args.ckpt or str(Path(cfg.out_dir) / "best.pth"), dev)
    model.eval()
    _, _, test = load_frames(cfg, verbose=False)
    if args.n:
        test = test.sample(min(args.n, len(test)), random_state=0).reset_index(drop=True)
    loader = make_loader(test, cfg, False, batch_size=cfg.batch_size * 2)

    @torch.no_grad()
    def collect(fp16):
        D, Y, W = [], [], []
        for batch in tqdm(loader, desc="fp16" if fp16 else "fp32"):
            img1, img2, _, _, y = batch
            with torch.autocast(dev.type, dtype=torch.float16, enabled=fp16 and dev.type == "cuda"):
                (s1, w1), (s2, w2) = model(img1.to(dev)), model(img2.to(dev))
            s1, w1, s2, w2 = s1.float(), w1.float(), s2.float(), w2.float()
            D.append((overall_vector(s1, w1) - overall_vector(s2, w2)).cpu()); Y.append(y)
            W.append(torch.softmax(w1, 1).cpu())
        return torch.cat(D).numpy(), torch.cat(Y).numpy(), torch.cat(W).numpy()

    for tag in args.modes.split(","):
        d, y, w = collect(tag == "fp16")
        pref = y != 0
        rows = []
        for i, name in enumerate(DIMS):
            di, yi, pi = d[:, i], y[:, i], pref[:, i]
            nz = pi & (di != 0)
            rows.append(dict(dim=name,
                             tied_in_labels_pct=round(100 * (~pi).mean(), 1),
                             diff_std=round(float(di.std()), 4),
                             pct_diff_exactly_0=round(100 * (di == 0).mean(), 1),
                             pct_abs_diff_below_eps=round(100 * (np.abs(di) <= cfg.eps).mean(), 1),
                             acc_non_tied=round(100 * (np.sign(di[pi]) == yi[pi]).mean(), 2) if pi.any() else np.nan,
                             direction_acc_when_diff_nonzero=round(100 * (np.sign(di[nz]) == yi[nz]).mean(), 2) if nz.any() else np.nan))
        print(f"\n===== {tag} inference, {len(y)} test pairs =====")
        print(pd.DataFrame(rows).to_string(index=False))
        print("mean softmax weight per dimension (SNR, Contrast, Resolution, Artifacts):", w.mean(0).round(3))


if __name__ == "__main__":
    main()
