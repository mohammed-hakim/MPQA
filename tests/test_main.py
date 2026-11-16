import subprocess
import sys
from pathlib import Path

MAIN = str(Path(__file__).resolve().parents[1] / "main.py")


def run(*args):
    return subprocess.run([sys.executable, MAIN, *args], capture_output=True, text=True)


def test_help_lists_every_command():
    r = run("--help")
    assert r.returncode == 0
    for cmd in ["train", "evaluate", "smoke", "baselines", "diagnose", "sanity", "clean-test"]:
        assert cmd in r.stdout


def test_unknown_command_fails():
    r = run("trian")
    assert r.returncode == 2 and "unknown command" in r.stderr


def test_command_help_is_forwarded_to_the_script():
    r = run("train", "--help")
    assert r.returncode == 0 and "--set" in r.stdout and "--no-test" in r.stdout


def test_train_through_main_runs_end_to_end(synthetic, tmp_path):
    r = run("train", f"--set=data_dir={synthetic}", "--set", "train_csv=train.csv", "--set", "test_csv=test.csv",
            "--set", "arch=vit_tiny_patch16_224", "--set", "img_size=224", "--set", "pretrained=false",
            "--set", "workers=0", "--set", "batch_size=4", "--set", "epochs=1", "--set", "amp=false",
            "--set", f"out_dir={tmp_path / 'run'}")
    assert r.returncode == 0, r.stderr[-800:]
    assert "TEST accuracy on non-tied pairs" in r.stdout and (tmp_path / "run" / "best.pth").exists()
