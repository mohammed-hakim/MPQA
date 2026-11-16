"""Dataset sanity analyses that need no images: leakage between splits and a metadata-only baseline."""
import numpy as np

from .labels import IMG1_COL, IMG2_COL, ORIG_COL, TAGS, signed_labels, subject_of


def _files(df):
    return set(df[IMG1_COL].astype(str)) | set(df[IMG2_COL].astype(str))


def _pair_keys(df):
    return {frozenset((str(a), str(b))) for a, b in zip(df[IMG1_COL], df[IMG2_COL])}


def leakage_report(a, b):
    """How much split `b` overlaps split `a`. Everything should be ~0."""
    files_a = _files(a)
    touching = b.apply(lambda r: str(r[IMG1_COL]) in files_a or str(r[IMG2_COL]) in files_a, axis=1).mean() * 100
    return dict(shared_scans=len(set(a[ORIG_COL].astype(str)) & set(b[ORIG_COL].astype(str))),
                shared_subjects=len(set(subject_of(a)) & set(subject_of(b))),
                shared_files=len(files_a & _files(b)),
                pairs_touching_a_file=round(float(touching), 1),
                identical_pairs=len(_pair_keys(a) & _pair_keys(b)))


def tag_features(df):
    """Degradation tags of image 1 minus image 2 (antisymmetric), parsed from the filenames."""
    tags = lambda name: np.array([float(t in str(name)) for t in TAGS])
    return np.stack([tags(a) - tags(b) for a, b in zip(df[IMG1_COL], df[IMG2_COL])])


def metadata_baseline(train, test, label_col=7):
    """Predict the preference from the degradation tags alone (no pixels), on non-tied pairs.
    -> (accuracy %, {tag: weight}, n_test_pairs). Needs scikit-learn."""
    from sklearn.linear_model import LogisticRegression

    def prep(df):
        y = signed_labels(df[label_col].values)
        keep = y != 0
        return tag_features(df)[keep], (y[keep] > 0).astype(int)

    xtr, ytr = prep(train)
    xte, yte = prep(test)
    clf = LogisticRegression(fit_intercept=False, C=10.0, max_iter=1000).fit(xtr, ytr)
    acc = float((clf.predict(xte) == yte).mean() * 100)
    return acc, {t.strip("-"): round(float(c), 2) for t, c in zip(TAGS, clf.coef_[0])}, len(yte)
