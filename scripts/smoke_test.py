#!/usr/bin/env python
"""Smoke test: which head setting avoids the collapse of the per-dimension scores?

Short, small-data trainings (same code, seed and data order) for a 2x2 grid: activation {relu, gelu} x head dropout
{0.5, 0}. It compares VALIDATION accuracy on non-tied pairs per dimension; the test set is never touched.
  relu_d0.5 is the authors' original setup (the CONTROL): it must collapse here too, otherwise the test is too short
  to say anything. Healthy = per-dimension accuracy clearly above 50% (chance); collapsed = far below 50%.

    python scripts/smoke_test.py                          # all 4 variants (2 GPUs -> two run at a time)
    python scripts/smoke_test.py gelu_d0 relu_d0          # only these
    python scripts/smoke_test.py --single                 # force one process
Extra overrides apply to every variant, e.g.  --set epochs=10 --set subset=0.5
"""
import argparse
import json
import os
import selectors
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpqa.cli import add_config_args  # noqa: E402
from mpqa.config import apply_overrides, load_config  # noqa: E402

VARIANTS = {
    "relu_d0.5": dict(act="relu", head_drop=0.5),   # control = the authors' setup
    "gelu_d0.5": dict(act="gelu", head_drop=0.5),
    "relu_d0": dict(act="relu", head_drop=0.0),
    "gelu_d0": dict(act="gelu", head_drop=0.0),
}
SMOKE = dict(epochs=6, subset=0.25, save_checkpoints=False, multi_gpu=False, workers=2)   # one GPU per run
DIM4 = ["SNR", "Contrast", "Resolution", "Artifacts"]
ALIVE_THRESHOLD = 55.0       # mean per-dimension accuracy on non-tied pairs; chance = 50 (a heuristic)


def build_cfg(args, name):
    cfg = load_config(args.config)
    cfg.update(SMOKE)                      # smoke defaults < user --set < the variant's own settings
    apply_overrides(cfg, args.set)
    cfg.update(VARIANTS[name])
    return cfg


def run(names, args, save_to=None):
    from mpqa.trainer import fit
    results = {}
    for name in names:
        print(f"\n################ variant {name}: {VARIANTS[name]} ################", flush=True)
        results[name] = fit(build_cfg(args, name))
        if save_to:                        # keep finished variants even if a later one crashes
            with open(save_to, "w") as f:
                json.dump(results, f)
    return results


def run_parallel(names, nprocs, args):
    """One worker process per GPU (CUDA_VISIBLE_DEVICES pins it), variants dealt out alternately.
    Worker output is streamed here with a [gpuN] prefix; results come back through JSON files."""
    tmp = Path(tempfile.mkdtemp(prefix="mpqa_smoke_"))
    sel, procs, bufs = selectors.DefaultSelector(), {}, {}
    common = (["--config", args.config] if args.config else []) + [x for s in args.set for x in ("--set", s)]
    for g in range(nprocs):
        grp = names[g::nprocs]
        if not grp:
            continue
        out = tmp / f"results_gpu{g}.json"
        p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--worker", str(out), *common, *grp],
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs[g], bufs[g] = (p, out), b""
        sel.register(p.stdout, selectors.EVENT_READ, g)
        print(f"[launcher] gpu{g}: {grp}", flush=True)
    while sel.get_map():
        for key, _ in sel.select():
            g = key.data
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                sel.unregister(key.fileobj)
                continue
            bufs[g] += chunk
            *lines, bufs[g] = bufs[g].replace(b"\r", b"\n").split(b"\n")
            for ln in lines:
                if ln.strip():
                    print(f"[gpu{g}]", ln.decode(errors="replace"), flush=True)
    merged = {}
    for g, (p, out) in procs.items():
        if p.wait() != 0:
            print(f"[launcher] WARNING: worker on gpu{g} exited with code {p.returncode}")
        if out.exists():
            merged.update(json.loads(out.read_text()))
    return {n: merged[n] for n in names if n in merged}


def summarize(results):
    if not results:
        raise SystemExit("No variant produced results (see the errors printed above).")
    rows, curves = [], {}
    for name, hist in results.items():
        mean_dims = [sum(h["val"][d] for d in DIM4) / 4 for h in hist]
        curves[name] = mean_dims
        last = hist[-1]["val"]
        rows.append(dict(variant=name, **{d: round(last[d], 1) for d in DIM4}, mean_dims=round(mean_dims[-1], 1),
                         Overall=round(last["Overall"], 1), best_Overall=round(max(h["val"]["Overall"] for h in hist), 1),
                         dims_alive=mean_dims[-1] > ALIVE_THRESHOLD))
    df = pd.DataFrame(rows)
    print("\n=== last-epoch VALIDATION accuracy on non-tied pairs (%) ===")
    print(df.to_string(index=False))
    print("\nmean per-dimension accuracy by epoch (watch for a collapse as epochs go by):")
    for n, c in curves.items():
        print(f"  {n:10s}", " ".join(f"{v:5.1f}" for v in c))

    ctrl = df[df.variant == "relu_d0.5"]
    if len(ctrl) and bool(ctrl.dims_alive.iloc[0]):
        print("\nVERDICT: the control did NOT collapse in this short run, so the test cannot separate the variants."
              "\n         Increase epochs / subset (e.g. --set epochs=15), or look at the per-epoch curves above.")
        return df
    ok = df[df.dims_alive].sort_values("Overall", ascending=False)
    if len(ok):
        pre = ("control collapsed (as in the authors' setup)" if len(ctrl)
               else "control (relu_d0.5) is not in this run - check that it collapsed elsewhere")
        print(f"\nVERDICT: {pre}; healthy variant(s): {', '.join(ok.variant)}."
              f"\n         Best by Overall: {ok.iloc[0].variant}.  Use its act / head_drop for the full run.")
    else:
        print("\nVERDICT: no variant kept the dimension scores alive -> do not launch a long run."
              "\n         Next to try: a plain Linear score head (no hidden layer) and/or a stronger weight on the"
              "\n         per-dimension ranking terms.")
    return df


def main():
    parser = add_config_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter))
    parser.add_argument("variants", nargs="*", help=f"subset of {list(VARIANTS)} (default: all)")
    parser.add_argument("--worker", default=None, help="internal: JSON file this worker writes its results to")
    parser.add_argument("--procs", type=int, default=None, help="parallel processes (default: number of GPUs)")
    parser.add_argument("--single", action="store_true", help="run everything in one process")
    args = parser.parse_args()
    bad = [v for v in args.variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; choose from {list(VARIANTS)}")
    names = args.variants or list(VARIANTS)

    if args.worker:
        run(names, args, save_to=args.worker)
        return
    import torch
    nprocs = 1 if args.single else (args.procs if args.procs is not None else torch.cuda.device_count())
    nprocs = min(nprocs, len(names))
    if nprocs >= 2:
        print(f"{nprocs} variants run at the same time, one process (GPU) each")
        results = run_parallel(names, nprocs, args)
    else:
        print("variants run one after another in this process")
        results = run(names, args)
    summarize(results)


if __name__ == "__main__":
    main()
