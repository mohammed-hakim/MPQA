"""Datasets and loaders for the MRIQP preference pairs."""
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .analysis import leakage_report
from .labels import (IMG1_COL, IMG2_COL, LABEL_COLS, drop_subjects, impact_label, shared_subjects,
                     signed_labels, subject_split)


def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),   # what timm's pretrained ViT-B/16-384 expects
    ])


class PairDataset(Dataset):
    """One item = (img1, img2, impact1, impact2, y): two images, their degradation-impact 4-vectors
    (training target only, parsed from the filenames) and the 5 signed preference labels."""
    def __init__(self, df, root, transform):
        self.rows, self.root, self.tf = df.values, root, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        f1, f2 = str(r[IMG1_COL]), str(r[IMG2_COL])
        img1 = self.tf(Image.open(os.path.join(self.root, f1)).convert("RGB"))
        img2 = self.tf(Image.open(os.path.join(self.root, f2)).convert("RGB"))
        y = torch.from_numpy(signed_labels(r[LABEL_COLS]))                      # (5,) in {-1, 0, 1}
        return img1, img2, torch.from_numpy(impact_label(f1)), torch.from_numpy(impact_label(f2)), y


def _csv_path(cfg, name):
    return Path(cfg.data_dir) / cfg[name]          # an absolute path in cfg[name] wins automatically


def read_csvs(cfg):
    full = pd.read_csv(_csv_path(cfg, "train_csv"), header=None)    # train + val pairs
    test = pd.read_csv(_csv_path(cfg, "test_csv"), header=None)
    return full, test


def load_frames(cfg, verbose=True):
    """-> (train_df, val_df, test_df). Subject-level train/val split; optional subset; optional shared-subject removal."""
    full, test = read_csvs(cfg)
    shared = shared_subjects(full, test)
    if cfg.drop_shared_subjects and shared:
        n0 = len(test)
        test = drop_subjects(test, shared)
        if verbose:
            print(f"dropped {n0 - len(test)} test pairs of subjects also present in the train CSV: {shared}", flush=True)
    train, val = subject_split(full, cfg.val_frac, cfg.seed)
    if cfg.subset < 1.0:
        train = train.sample(frac=cfg.subset, random_state=cfg.seed).reset_index(drop=True)
    if verbose:
        print(f"pairs  train={len(train)}  val={len(val)}  test={len(test)}", flush=True)
        if not cfg.drop_shared_subjects:
            csv_overlap = leakage_report(full, test)
            print(
                f"train-csv/test-csv overlap: subject IDs={csv_overlap['shared_subjects']}, "
                f"scans={csv_overlap['shared_scans']}, image files={csv_overlap['shared_files']}, "
                f"test pairs touching a train file={csv_overlap['pairs_touching_a_file']:.1f}%, "
                f"identical pairs={csv_overlap['identical_pairs']}",
                flush=True,
            )
    return train, val, test


def make_loader(df, cfg, shuffle, batch_size=None):
    ds = PairDataset(df, os.path.join(cfg.data_dir, cfg.img_dir), make_transform(cfg.img_size))
    return DataLoader(ds, batch_size=batch_size or cfg.batch_size, shuffle=shuffle, num_workers=cfg.workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle)
