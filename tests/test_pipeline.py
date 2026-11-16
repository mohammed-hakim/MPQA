"""End-to-end checks on a fake dataset with a tiny ViT (CPU, no downloads, a few seconds)."""
import pandas as pd

from mpqa.config import load_config
from mpqa.data import load_frames
from mpqa.labels import DIMS, shared_subjects
from mpqa.trainer import evaluate_checkpoint, fit


def test_fit_writes_files_and_resumes(tiny_cfg, capsys):
    history = fit(tiny_cfg)
    out = pathlib_out(tiny_cfg)
    assert len(history) == 1 and set(history[0]["val"]) == set(DIMS)
    assert all(0 <= v <= 100 for v in history[0]["val"].values())
    assert (out / "best.pth").exists() and (out / "last.pth").exists() and not (out / "last.pth.tmp").exists()
    assert len(pd.read_csv(out / "history.csv")) == 1

    tiny_cfg.epochs = 2                                       # same out_dir -> resumes after epoch 1
    history = fit(tiny_cfg)
    assert [h["epoch"] for h in history] == [1, 2]
    assert "resumed from" in capsys.readouterr().out
    assert len(pd.read_csv(out / "history.csv")) == 2


def test_evaluate_checkpoint_from_best_and_last(tiny_cfg):
    fit(tiny_cfg)
    out = pathlib_out(tiny_cfg)
    for name in ("best.pth", "last.pth"):                     # both checkpoint formats load
        loss, acc, acc_all = evaluate_checkpoint(tiny_cfg, out / name)
        assert loss > 0 and set(acc) == set(DIMS) == set(acc_all)
        assert all(0 <= v <= 100 for v in acc.values()) and all(0 <= v <= 100 for v in acc_all.values())


def test_nothing_is_written_when_checkpoints_are_off(tiny_cfg):
    tiny_cfg.save_checkpoints = False
    fit(tiny_cfg)
    assert not pathlib_out(tiny_cfg).exists()


def test_frames_have_no_leakage_and_drop_option(synthetic, tiny_cfg):
    train, val, test = load_frames(tiny_cfg, verbose=False)
    assert not set(train[1]) & set(val[1]) and not set(train[1]) & set(test[1])
    full = pd.read_csv(synthetic / "train.csv", header=None)
    leaky = pd.concat([pd.read_csv(synthetic / "test.csv", header=None), full.iloc[:6]], ignore_index=True)
    leaky.to_csv(synthetic / "test_leaky.csv", header=False, index=False)
    cfg = load_config(overrides=[f"data_dir={synthetic}", "train_csv=train.csv", "test_csv=test_leaky.csv"])
    assert len(load_frames(cfg, verbose=False)[2]) == len(leaky)                 # default keeps the Kaggle test set
    cfg.drop_shared_subjects = True
    cleaned = load_frames(cfg, verbose=False)[2]
    assert len(cleaned) == len(leaky) - 6 and not shared_subjects(full, cleaned)


def pathlib_out(cfg):
    from pathlib import Path
    return Path(cfg.out_dir)
