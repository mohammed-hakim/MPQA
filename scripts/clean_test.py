#!/usr/bin/env python
"""Write a copy of the test CSV without subjects that also appear in the train CSV.

    python scripts/clean_test.py --out test_clean.csv
    python scripts/evaluate.py --ckpt runs/default/best.pth --set test_csv=/abs/path/test_clean.csv
(or simply  --set drop_shared_subjects=true  on train.py / evaluate.py, which does the same in memory)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpqa.cli import add_config_args, cfg_from_args  # noqa: E402
from mpqa.data import read_csvs  # noqa: E402
from mpqa.labels import ORIG_COL, drop_subjects, shared_subjects, subject_of  # noqa: E402


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    parser.add_argument("--out", default="test_clean.csv")
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    full, test = read_csvs(cfg)
    print(f"unique subjects: train-csv {subject_of(full).nunique()}, test-csv {subject_of(test).nunique()} "
          "(sanity: should be many, not 1-2)")
    shared = shared_subjects(full, test)
    for sid in shared:
        s_full, s_test = subject_of(full) == sid, subject_of(test) == sid
        print(f"shared subject {sid}: {int(s_full.sum())} pairs in train-csv, {int(s_test.sum())} in test-csv | "
              f"examples: train {full[s_full][ORIG_COL].unique()[:2].tolist()} test {test[s_test][ORIG_COL].unique()[:2].tolist()}")
    clean = drop_subjects(test, shared)
    clean.to_csv(args.out, header=False, index=False)
    print(f"removed {len(test) - len(clean)} of {len(test)} test pairs -> {len(clean)} left, saved to {args.out}")
    print("shared subjects now:", len(shared_subjects(full, clean)))


if __name__ == "__main__":
    main()
