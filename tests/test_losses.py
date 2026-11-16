import numpy as np
import pytest
import torch

from mpqa.losses import jq_loss, rank_term, zlpr
from mpqa.model import overall_vector


def naive_zlpr(logits, y):
    """Paper Eq. 4 written literally with exp()."""
    pos = np.log1p((np.exp(-logits) * y).sum(1))
    neg = np.log1p((np.exp(logits) * (1 - y)).sum(1))
    return (pos + neg).mean()


def test_zlpr_matches_the_paper_formula():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(16, 4)); y = (rng.random((16, 4)) > 0.5).astype(np.float64)
    got = zlpr(torch.tensor(logits), torch.tensor(y)).item()
    assert got == pytest.approx(naive_zlpr(logits, y), rel=1e-9)


def test_zlpr_is_stable_for_huge_logits():
    logits = torch.tensor([[300.0, -300.0, 250.0, -250.0]], requires_grad=True)
    y = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    loss = zlpr(logits, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_zlpr_rewards_correct_direction():
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    good = torch.tensor([[5.0, -5.0, 5.0, -5.0]]); bad = -good
    assert zlpr(good, y) < zlpr(bad, y)


def test_rank_term_matches_the_three_branches_of_eq6():
    rng = np.random.default_rng(1)
    yh1, yh2 = rng.normal(size=(50, 5)), rng.normal(size=(50, 5))
    y = rng.choice([-1.0, 0.0, 1.0], size=(50, 5)); eps = 0.1
    expected = np.zeros_like(yh1)
    expected[y == 1] = np.maximum(yh2 - yh1 + eps, 0)[y == 1]       # image 1 better: want yh1 > yh2 + eps
    expected[y == 0] = np.abs(yh2 - yh1)[y == 0]                    # equal: want yh1 == yh2
    expected[y == -1] = np.maximum(yh1 - yh2 + eps, 0)[y == -1]     # image 2 better
    got = rank_term(torch.tensor(yh1), torch.tensor(yh2), torch.tensor(y), eps).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_overall_vector():
    s = torch.tensor([[1.0, 2.0, 3.0, 4.0]]); w = torch.zeros(1, 4)       # equal weights -> plain mean
    out = overall_vector(s, w)
    assert out.shape == (1, 5) and out[0, :4].tolist() == [1, 2, 3, 4] and out[0, 4].item() == pytest.approx(2.5)
    w = torch.tensor([[0.0, 0.0, 0.0, 50.0]])                              # all weight on the last dimension
    assert overall_vector(s, w)[0, 4].item() == pytest.approx(4.0, abs=1e-4)


def _fake_batch(b=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = lambda: (torch.randn(b, 4, generator=g, requires_grad=True), torch.randn(b, 4, generator=g, requires_grad=True))
    imp = lambda: (torch.rand(b, 4, generator=g) > 0.5).float()
    y = torch.tensor(np.random.default_rng(seed).choice([-1.0, 0.0, 1.0], size=(b, 5)), dtype=torch.float32)
    return out(), out(), imp(), imp(), y


def test_jq_loss_backward_and_shapes():
    o1, o2, i1, i2, y = _fake_batch()
    loss, yh1, yh2 = jq_loss(o1, o2, i1, i2, y, eps=0.1, lam=4.0)
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss) and yh1.shape == yh2.shape == (6, 5)
    assert all(torch.isfinite(t.grad).all() for t in (*o1, *o2))


def test_lambda_only_scales_the_overall_term():
    o1, o2, i1, i2, y = _fake_batch(seed=3)
    l0, yh1, yh2 = jq_loss(o1, o2, i1, i2, y, eps=0.1, lam=0.0)
    l4, _, _ = jq_loss(o1, o2, i1, i2, y, eps=0.1, lam=4.0)
    overall_term = rank_term(yh1, yh2, y, 0.1)[:, 4].mean()
    assert (l4 - l0).item() == pytest.approx(4.0 * overall_term.item(), rel=1e-5)
