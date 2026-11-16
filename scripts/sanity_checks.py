#!/usr/bin/env python
"""CPU-only sanity checks (no images loaded):
 1. leakage between train / val / test (shared scans, subjects, degraded files, identical pairs);
 2. a metadata-only baseline: predict the Overall preference from the degradation TAGS in the two filenames,
    with no pixels. A high score means the labels are largely explained by WHICH degradations were applied.
needs scikit-learn.     python scripts/sanity_checks.py --set data_dir=/path/to/LabelMRI4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpqa.analysis import leakage_report, metadata_baseline  # noqa: E402
from mpqa.cli import add_config_args, cfg_from_args  # noqa: E402
from mpqa.data import read_csvs  # noqa: E402
from mpqa.labels import subject_split  # noqa: E402


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)).parse_args()
    cfg = cfg_from_args(args)
    full, test = read_csvs(cfg)
    train, val = subject_split(full, cfg.val_frac, cfg.seed)

    print("=== 1. leakage check (all zeros / ~0% is what we want) ===")
    for (na, a), (nb, b) in [(("train", train), ("val", val)), (("train", train), ("test", test)), (("val", val), ("test", test))]:
        r = leakage_report(a, b)
        print(f"{na:>5} vs {nb:<5} | shared scans {r['shared_scans']:5d} | shared subjects {r['shared_subjects']:4d} | "
              f"shared degraded files {r['shared_files']:5d} | {nb} pairs touching a {na} file {r['pairs_touching_a_file']:5.1f}% | "
              f"identical pairs {r['identical_pairs']}")

    print("\n=== 2. metadata-only baseline (no pixels): filename degradation tags -> Overall preference ===")
    acc, coef, n = metadata_baseline(full, test)
    print(f"test accuracy on {n} non-tied pairs: {acc:.2f}%   (compare with the MPQA test number)")
    print("learned weight per tag (negative = images carrying this degradation tend to be judged worse):", coef)


if __name__ == "__main__":
    main()
