"""Joint quality (JQ) loss = QP (ZLPR on degradation-impact labels) + RQ (pairwise ranking). Paper Eqs. 4-8."""
import torch
import torch.nn.functional as F

from .model import overall_vector


def zlpr(logits, y):
    """Per-sample ZLPR (paper Eq. 4):  log(1 + sum_pos e^-s) + log(1 + sum_neg e^s), averaged over the batch.
    Written with logsumexp so large logits cannot overflow to inf/NaN."""
    zero = torch.zeros_like(logits[:, :1])
    ninf = torch.full_like(logits, float("-inf"))
    pos = torch.logsumexp(torch.cat([zero, torch.where(y == 1, -logits, ninf)], dim=1), dim=1)
    neg = torch.logsumexp(torch.cat([zero, torch.where(y == 0, logits, ninf)], dim=1), dim=1)
    return (pos + neg).mean()


def rank_term(yhat1, yhat2, y, eps):
    """Paper Eq. 6, element-wise. y=+1: image 1 better, -1: image 2 better, 0: equal."""
    d = yhat1 - yhat2
    return torch.where(y == 0, d.abs(), F.relu(-y * d + eps))


def jq_loss(out1, out2, imp1, imp2, y, eps, lam):
    """out1/out2 = (scores, weight_logits) of the two images; imp = degradation-impact labels; y = (B,5) preferences.
    Returns (loss, yhat1, yhat2) where yhat = (B,5) [4 dimension scores, overall]."""
    (s1, w1), (s2, w2) = out1, out2
    # QP loss. NOTE: the authors' released code applies ZLPR to the weight logits (w), not to the scores (s) as
    # Eq. 5 is written. It is also what makes the paper's Fig. 4 work (weight ~1 on the impacted dimensions).
    qp = 0.5 * (zlpr(w1, imp1) + zlpr(w2, imp2))
    yh1, yh2 = overall_vector(s1, w1), overall_vector(s2, w2)
    rq = rank_term(yh1, yh2, y, eps)                                          # (B, 5)
    rq = torch.cat([rq[:, :4], lam * rq[:, 4:]], dim=1).sum(dim=1).mean()     # Eq. 7
    return qp + rq, yh1.detach(), yh2.detach()                                # Eq. 8
