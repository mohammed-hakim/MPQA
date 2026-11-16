import numpy as np
import pandas as pd

from mpqa.labels import (drop_subjects, impact_label, shared_subjects, signed_labels, subject_of,
                         subject_split)


def test_impact_label_mapping():
    assert impact_label("100-1--noise--.png").tolist() == [1, 0, 0, 0]
    assert impact_label("x-contrast-.png").tolist() == [0, 1, 0, 0]
    assert impact_label("x-crop-.png").tolist() == [0, 0, 1, 0]
    assert impact_label("x-motion-.png").tolist() == [0, 0, 0, 1]
    assert impact_label("x-herring-.png").tolist() == [0, 0, 0, 1]
    assert impact_label("x-maskD1-.png").tolist() == [0, 0, 1, 1]          # '-mask' also matches -maskD1-, -mask-D1D2-
    assert impact_label("x-denoise-.png").tolist() == [1, 0, 1, 0]
    assert impact_label("x-noise--crop-.png").tolist() == [1, 0, 1, 0]      # combined degradations add up
    assert impact_label("plain.png").tolist() == [0, 0, 0, 0]


def test_signed_labels():
    assert signed_labels([1, 2, 3, 1, 3]).tolist() == [1, 0, -1, 1, -1]


def test_subject_of():
    df = pd.DataFrame({0: ["100307-slice.png", "abc.png", "42_x.png"]})
    assert subject_of(df).tolist() == ["100307", "abc.png", "42"]


def test_subject_split_is_disjoint(synthetic):
    df = pd.read_csv(synthetic / "train.csv", header=None)
    train, val = subject_split(df, 0.2, seed=42)
    assert len(train) + len(val) == len(df)
    assert not set(subject_of(train)) & set(subject_of(val))
    assert not set(train[1]) & set(val[1])                  # no degraded image file on both sides
    assert subject_of(val).nunique() == int(12 * 0.2)
    train2, val2 = subject_split(df, 0.2, seed=42)           # deterministic for a given seed
    assert val.equals(val2)


def test_shared_and_drop(synthetic):
    train = pd.read_csv(synthetic / "train.csv", header=None)
    test = pd.read_csv(synthetic / "test.csv", header=None)
    assert shared_subjects(train, test) == []
    leaky = pd.concat([test, train.iloc[:6]], ignore_index=True)
    assert shared_subjects(train, leaky) == ["100"]
    cleaned = drop_subjects(leaky, ["100"])
    assert len(cleaned) == len(test) and shared_subjects(train, cleaned) == []
