"""Accuracy on preference pairs.

Only pairs where the experts preferred one image (label != 0) are scored.
"""
import torch


def pair_counts(d, y, eps):
    """d: (B,5) score1 - score2; y: (B,5) in {-1,0,1}.

    Returns (ok_pref, n_pref), counting only expert-preferred pairs.
    """
    has_pref = y != 0
    ok_pref = ((torch.sign(d) == y) & has_pref).float().sum(dim=0)
    n_pref = has_pref.float().sum(dim=0)
    return ok_pref, n_pref
