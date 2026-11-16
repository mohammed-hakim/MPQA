"""Accuracy on preference pairs. Two views of the same predictions:

 acc_pref : only pairs where the experts preferred one image (label != 0); prediction = sign(score1 - score2).
            This is the paper-style number: the paper's PSNR value (67.01) equals correct / non-tied pairs.
 acc_all  : every pair; predicting "equal" needs |score1 - score2| <= eps, so ties are scored too.
"""
import torch


def pair_counts(d, y, eps):
    """d: (B,5) score1 - score2;  y: (B,5) in {-1,0,1}.  -> (ok_pref, n_pref, ok_all, n), per dimension."""
    has_pref = y != 0
    ok_pref = ((torch.sign(d) == y) & has_pref).float().sum(dim=0)
    n_pref = has_pref.float().sum(dim=0)
    pred = torch.where(d > eps, 1.0, torch.where(d < -eps, -1.0, 0.0))
    ok_all = (pred == y).float().sum(dim=0)
    return ok_pref, n_pref, ok_all, y.size(0)
