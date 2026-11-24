import torch

from mpqa.metrics import pair_counts


def test_pair_counts_hand_checked():
    d = torch.tensor([[0.5, -0.5, 0.05, 0.3, 0.0]])
    y = torch.tensor([[1.0, 1.0, -1.0, 0.0, 0.0]])
    ok_pref, n_pref = pair_counts(d, y, eps=0.1)
    assert ok_pref.tolist() == [1, 0, 0, 0, 0]       # only dim 0 has the right sign among preferred pairs
    assert n_pref.tolist() == [1, 1, 1, 0, 0]        # tied pairs (y == 0) are excluded from the paper-style metric


def test_exact_score_tie_is_wrong_on_a_preferred_pair():
    ok_pref, n_pref = pair_counts(torch.zeros(1, 5), torch.ones(1, 5), eps=0.1)
    assert ok_pref.sum() == 0 and n_pref.sum() == 5
