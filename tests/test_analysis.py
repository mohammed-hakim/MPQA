import numpy as np
import pandas as pd
import pytest

from mpqa.analysis import leakage_report, metadata_baseline, tag_features  # noqa: E402
from mpqa.labels import ORIG_COL, TAGS, subject_of  # noqa: E402


def test_leakage_report_distinguishes_subject_from_scan_overlap(synthetic):
    train = pd.read_csv(synthetic / "train.csv", header=None)
    test = pd.read_csv(synthetic / "test.csv", header=None)
    subject = subject_of(train).iloc[0]
    test.iloc[0, ORIG_COL] = f"{subject}-another-orig.png"

    report = leakage_report(train, test)

    assert report == dict(shared_scans=0, shared_subjects=1, shared_files=0,
                          pairs_touching_a_file=0.0, identical_pairs=0)


def test_leakage_report_detects_a_planted_leak(synthetic):
    train = pd.read_csv(synthetic / "train.csv", header=None)
    test = pd.read_csv(synthetic / "test.csv", header=None)
    clean = leakage_report(train, test)
    assert clean == dict(shared_scans=0, shared_subjects=0, shared_files=0, pairs_touching_a_file=0.0, identical_pairs=0)
    leaky = pd.concat([test, train.iloc[:5]], ignore_index=True)
    r = leakage_report(train, leaky)
    assert r["identical_pairs"] >= 1 and r["shared_files"] > 0 and r["shared_subjects"] == 1 and r["pairs_touching_a_file"] > 0


def test_tag_features_are_antisymmetric():
    df = pd.DataFrame([["o", "1--noise--.png", "2--crop--.png"], ["o", "2--crop--.png", "1--noise--.png"]])
    f = tag_features(df)
    assert f.shape == (2, len(TAGS)) and np.allclose(f[0], -f[1])
    assert f[0][TAGS.index("-noise-")] == 1 and f[0][TAGS.index("-crop-")] == -1


def test_metadata_baseline_learns_the_synthetic_rule(synthetic):
    pytest.importorskip("sklearn")
    # in the fake data the image with FEWER degradation tags always wins, so tags alone should predict well
    train = pd.read_csv(synthetic / "train.csv", header=None)
    test = pd.read_csv(synthetic / "test.csv", header=None)
    acc, coef, n = metadata_baseline(train, test)
    assert acc > 90 and n > 0 and set(coef) == {t.strip("-") for t in TAGS}
    assert all(w < 0 for w in coef.values())          # having any degradation makes an image look worse
