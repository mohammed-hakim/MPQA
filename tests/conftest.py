import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpqa.config import load_config  # noqa: E402

TAG_NAMES = ["noise", "contrast", "crop", "motion", "herring", "mask", "denoise"]


def _make_split(root, n_subjects, pairs_per_subject, rng, first_id):
    rows = []
    for s in range(n_subjects):
        sid = first_id + s
        images = []
        for i in range(4):
            tags = list(rng.choice(TAG_NAMES, size=int(rng.integers(1, 3)), replace=False))
            name = f"{sid}-{i}-" + "".join(f"-{t}-" for t in tags) + ".png"
            Image.fromarray((rng.random((40, 40)) * 255).astype(np.uint8)).save(root / "noisedImages" / name)
            images.append((name, len(tags)))
        for _ in range(pairs_per_subject):
            a, b = rng.choice(4, 2, replace=False)
            lab = 1 if images[a][1] < images[b][1] else (3 if images[a][1] > images[b][1] else 2)
            rows.append([f"{sid}-orig.png", images[a][0], images[b][0], lab, lab, 2, lab, lab])
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic(tmp_path):
    """A tiny fake dataset on disk: 12 train subjects, 4 test subjects, 6 pairs each."""
    root = tmp_path / "data"
    (root / "noisedImages").mkdir(parents=True)
    rng = np.random.default_rng(0)
    train = _make_split(root, 12, 6, rng, 100)
    test = _make_split(root, 4, 6, rng, 500)
    train.to_csv(root / "train.csv", header=False, index=False)
    test.to_csv(root / "test.csv", header=False, index=False)
    return root


@pytest.fixture
def tiny_cfg(synthetic, tmp_path):
    """Config that runs the whole pipeline in seconds on CPU with a tiny ViT and no downloads."""
    return load_config(overrides=[
        f"data_dir={synthetic}", "train_csv=train.csv", "test_csv=test.csv",
        "arch=vit_tiny_patch16_224", "img_size=224", "pretrained=false",
        "workers=0", "batch_size=4", "epochs=1", "amp=false", "multi_gpu=false",
        f"out_dir={tmp_path / 'out'}",
    ])
