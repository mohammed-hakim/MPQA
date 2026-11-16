#!/usr/bin/env python
"""Single entry point for every step.

    python main.py train      --set data_dir=/path/to/LabelMRI4 --set out_dir=runs/main
    python main.py evaluate   --ckpt runs/main/best.pth --set data_dir=/path/to/LabelMRI4
    python main.py smoke      --set data_dir=...        # head-variant test (activation x dropout)
    python main.py baselines  --set data_dir=...        # PSNR, SSIM, LPIPS, ... (needs pyiqa)
    python main.py diagnose   --ckpt runs/main/best.pth --set data_dir=...
    python main.py sanity     --set data_dir=...        # leakage + metadata-only baseline (CPU)
    python main.py clean-test --set data_dir=... --out test_clean.csv

Every command accepts --config FILE and repeated --set key=value (see configs/default.yaml),
and `python main.py <command> --help` shows that command's own options.
"""
import runpy
import sys
from pathlib import Path

COMMANDS = {
    "train": "train.py",
    "evaluate": "evaluate.py",
    "smoke": "smoke_test.py",
    "baselines": "baselines.py",
    "diagnose": "diagnose.py",
    "sanity": "sanity_checks.py",
    "clean-test": "clean_test.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"unknown command '{cmd}'. Choose from: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    script = Path(__file__).resolve().parent / "scripts" / COMMANDS[cmd]
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
