#!/usr/bin/env python
"""Train MPQA (validation every epoch), then evaluate the best checkpoint on the test set.

    python scripts/train.py --set data_dir=/path/to/LabelMRI4 --set epochs=40
    python scripts/train.py --set epochs=60            # same out_dir: resumes from last.pth automatically
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpqa.cli import add_config_args, cfg_from_args  # noqa: E402
from mpqa.trainer import evaluate_checkpoint, fit, print_report  # noqa: E402


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    parser.add_argument("--no-test", action="store_true", help="skip the final test evaluation")
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    fit(cfg)
    best = Path(cfg.out_dir) / "best.pth"
    if cfg.save_checkpoints and best.exists() and not args.no_test:
        _, acc, acc_all = evaluate_checkpoint(cfg, best)
        print_report(acc, acc_all, title="TEST")


if __name__ == "__main__":
    main()
