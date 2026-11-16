"""Training, validation, checkpointing (best / last / resume) and checkpoint evaluation."""
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import load_frames, make_loader
from .labels import DIMS
from .losses import jq_loss
from .metrics import pair_counts
from .model import MPQA

PAPER_TEST = {"Overall": 82.51, "SNR": 80.81, "Contrast": 66.55, "Resolution": 82.43, "Artifacts": 82.42}


def log(*args):
    print(*args, flush=True)          # flush: epoch lines show up immediately in Kaggle / nohup logs


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg, device, pretrained=None):
    """-> (core, model). `core` is the plain module (used for checkpoints); `model` may be wrapped in DataParallel."""
    use_pretrained = cfg.pretrained if pretrained is None else pretrained
    core = MPQA(cfg.arch, cfg.act, cfg.head_drop, pretrained=use_pretrained).to(device)
    multi = cfg.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1
    return core, (nn.DataParallel(core) if multi else core)


def load_weights(core, path, device):
    """Accepts both best.pth (a state_dict) and last.pth (a dict holding the state_dict under 'model')."""
    ck = torch.load(path, map_location=device, weights_only=False)
    core.load_state_dict(ck["model"] if isinstance(ck, dict) and "opt" in ck else ck)


def _forward(model, img1, img2, device, amp):
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        o1, o2 = model(img1), model(img2)
    return [t.float() for t in o1], [t.float() for t in o2]       # loss in fp32 (exp() can overflow in fp16)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    """-> (loss, acc_pref, acc_all); the accuracies are dicts over DIMS in %, see mpqa.metrics."""
    model.eval()
    amp = cfg.amp and device.type == "cuda"
    ok_pref, n_pref, ok_all, n, loss_sum = torch.zeros(5), torch.zeros(5), torch.zeros(5), 0, 0.0
    for batch in loader:
        img1, img2, imp1, imp2, y = [t.to(device, non_blocking=True) for t in batch]
        o1, o2 = _forward(model, img1, img2, device, amp)
        loss, yh1, yh2 = jq_loss(o1, o2, imp1, imp2, y, cfg.eps, cfg.lam)
        a, b, c, m = pair_counts(yh1 - yh2, y, cfg.eps)
        ok_pref += a.cpu(); n_pref += b.cpu(); ok_all += c.cpu(); n += m; loss_sum += loss.item() * m
    pct = lambda ok, total: dict(zip(DIMS, (ok / total * 100).tolist()))
    return loss_sum / n, pct(ok_pref, n_pref.clamp(min=1)), pct(ok_all, n)


def _write_history(history, path):
    rows = [{"epoch": h["epoch"], "train_loss": h["train_loss"], "val_loss": h["val_loss"],
             **{f"val_{k}": v for k, v in h["val"].items()}, "val_all_Overall": h["val_all"]["Overall"]}
            for h in history]
    pd.DataFrame(rows).to_csv(path, index=False)


def fit(cfg):
    """Train with per-epoch validation. Returns the history (list of per-epoch dicts).
    With cfg.save_checkpoints it writes best.pth (best validation Overall), last.pth (full state, for resume)
    and history.csv into cfg.out_dir. The test set is never touched here."""
    seed_all(cfg.seed)
    device = get_device()
    amp = cfg.amp and device.type == "cuda"
    out_dir = Path(cfg.out_dir)
    if cfg.save_checkpoints:
        out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, _ = load_frames(cfg)
    train_loader, val_loader = make_loader(train_df, cfg, True), make_loader(val_df, cfg, False)

    core, model = build_model(cfg, device)
    opt = torch.optim.SGD(core.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    log("GPUs used:", torch.cuda.device_count() if isinstance(model, nn.DataParallel) else (1 if device.type == "cuda" else 0))

    best_acc, history, start_epoch = -1.0, [], 1
    best_path, last_path = out_dir / "best.pth", out_dir / "last.pth"
    resume_path = Path(cfg.resume_from) if cfg.resume_from else last_path
    if cfg.save_checkpoints and resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        core.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        best_acc, history, start_epoch = ck["best_acc"], ck["history"], ck["epoch"] + 1
        log(f"resumed from {resume_path}: epoch {ck['epoch']} done, best val overall so far {best_acc:.2f}%")
        if not best_path.exists():                                # new session: bring the earlier best.pth along
            prev_best = resume_path.parent / "best.pth"
            if prev_best.exists():
                shutil.copy(prev_best, best_path)
            else:
                best_acc = -1.0
                log("WARNING: earlier best.pth not found next to last.pth; best checkpoint is chosen among resumed epochs only")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        run = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False, mininterval=120):
            img1, img2, imp1, imp2, y = [t.to(device, non_blocking=True) for t in batch]
            opt.zero_grad(set_to_none=True)
            o1, o2 = _forward(model, img1, img2, device, amp)
            loss, _, _ = jq_loss(o1, o2, imp1, imp2, y, cfg.eps, cfg.lam)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item()
        train_loss = run / len(train_loader)
        val_loss, acc, acc_all = evaluate(model, val_loader, device, cfg)
        log(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | " +
            " ".join(f"{k}:{v:.1f}" for k, v in acc.items()) +
            f" | overall(all pairs, ties scored): {acc_all['Overall']:.1f}")
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val=acc, val_all=acc_all))
        if acc["Overall"] > best_acc:
            best_acc = acc["Overall"]
            if cfg.save_checkpoints:
                torch.save(core.state_dict(), best_path)
            log(f"   -> new best val overall acc (non-tied pairs) {best_acc:.2f}%")
        if cfg.save_checkpoints:
            tmp = str(last_path) + ".tmp"                         # temp file first: a kill cannot corrupt last.pth
            torch.save(dict(model=core.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch, best_acc=best_acc, history=history), tmp)
            os.replace(tmp, last_path)
            _write_history(history, out_dir / "history.csv")
    return history


def evaluate_checkpoint(cfg, ckpt, df=None):
    """Evaluate a saved checkpoint on `df` (default: the test set). -> (loss, acc_pref, acc_all)."""
    device = get_device()
    core, model = build_model(cfg, device, pretrained=False)      # weights come from the checkpoint: no download
    load_weights(core, ckpt, device)
    if df is None:
        _, _, df = load_frames(cfg, verbose=False)
    loader = make_loader(df, cfg, False, batch_size=cfg.batch_size * 2)
    return evaluate(model, loader, device, cfg)


def print_report(acc, acc_all, title="TEST"):
    log(f"\n{title} accuracy on non-tied pairs (%, the paper-style metric):", {k: round(v, 2) for k, v in acc.items()})
    log(f"{title} accuracy on all pairs incl. ties (%):                    ", {k: round(v, 2) for k, v in acc_all.items()})
    log("paper (Table 2/3): " + " | ".join(f"{k} {v}" for k, v in PAPER_TEST.items()))
