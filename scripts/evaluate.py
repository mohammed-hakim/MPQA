#!/usr/bin/env python
"""Evaluate a saved checkpoint (best.pth or last.pth) on the test or validation split.

    python scripts/evaluate.py --ckpt runs/default/best.pth
    python scripts/evaluate.py --ckpt runs/default/best.pth --set drop_shared_subjects=true
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpqa.cli import add_config_args, cfg_from_args  # noqa: E402
from mpqa.analysis import leakage_report  # noqa: E402
from mpqa.data import load_frames  # noqa: E402
from mpqa.trainer import evaluate_checkpoint, print_report  # noqa: E402


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    parser.add_argument("--ckpt", default=None, help="checkpoint path (default: <out_dir>/best.pth)")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    train, val, test = load_frames(cfg)
    eval_df = test if args.split == "test" else val
    overlap = leakage_report(train, eval_df)
    print(
        f"train/{args.split} overlap: shared subjects={overlap['shared_subjects']}, "
        f"shared scans={overlap['shared_scans']}, shared degraded files={overlap['shared_files']}, "
        f"evaluation pairs touching a train file={overlap['pairs_touching_a_file']:.1f}%, "
        f"identical pairs={overlap['identical_pairs']}"
    )
    ckpt = args.ckpt or str(Path(cfg.out_dir) / "best.pth")
    _, acc = evaluate_checkpoint(cfg, ckpt, df=eval_df)
    print_report(acc, title=args.split.upper())


if __name__ == "__main__":
    main()
