"""Label parsing and subject-level splitting. Pure Python / pandas: no torch needed.

CSV layout (no header):  0 original | 1 img1 | 2 img2 | 3..7 = SNR, Contrast, Resolution, Artifacts, Overall
Each label is 1 (img1 better) / 2 (equal) / 3 (img2 better); `2 - x` turns it into +1 / 0 / -1.
"""
import numpy as np

DIMS = ["SNR", "Contrast", "Resolution", "Artifacts", "Overall"]
ORIG_COL, IMG1_COL, IMG2_COL = 0, 1, 2
LABEL_COLS = slice(3, 8)

# Which of the 4 quality dimensions (SNR, Contrast, Resolution, Artifacts) each degradation hurts.
# Same mapping as the authors' released code. Matched on the FILENAME only (never given to the model as input).
FIELD = {"-noise-": [0], "-contrast-": [1], "-crop-": [2], "-motion-": [3],
         "-herring-": [3], "-mask": [2, 3], "-denoise-": [0, 2]}
TAGS = list(FIELD)


def impact_label(filename):
    """4-vector: 1 where a degradation tagged in the filename affects that quality dimension."""
    y = np.zeros(4, dtype=np.float32)
    for tag, dims in FIELD.items():
        if tag in str(filename):
            y[dims] = 1
    return y


def signed_labels(raw):
    """Raw CSV labels (1/2/3) -> +1 (img1 better) / 0 (equal) / -1 (img2 better)."""
    return 2.0 - np.asarray(raw, dtype=np.float32)


def subject_of(df):
    """Subject id = the digits at the start of the original-image filename (falls back to the whole name)."""
    names = df[ORIG_COL].astype(str)
    return names.str.extract(r"^(\d+)")[0].fillna(names)


def subject_split(df, val_frac, seed):
    """Split by subject so no subject appears in both parts (the paper: avoid data leakage)."""
    subj = subject_of(df)
    ids = np.array(sorted(subj.unique()))
    np.random.RandomState(seed).shuffle(ids)
    val_ids = set(ids[:int(len(ids) * val_frac)])
    is_val = subj.isin(val_ids).values
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def shared_subjects(a, b):
    return sorted(set(subject_of(a)) & set(subject_of(b)))


def drop_subjects(df, subjects):
    return df[~subject_of(df).isin(set(subjects))].reset_index(drop=True)
