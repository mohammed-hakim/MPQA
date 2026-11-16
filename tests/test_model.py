import pytest
import torch
import torch.nn as nn

from mpqa.model import MPQA


def test_forward_shapes():
    m = MPQA("vit_tiny_patch16_224", pretrained=False).eval()
    s, w = m(torch.randn(3, 3, 224, 224))
    assert s.shape == (3, 4) and w.shape == (3, 4)


def test_head_options():
    m = MPQA("vit_tiny_patch16_224", act="relu", head_drop=0.5, pretrained=False)
    assert isinstance(m.mlp_s[1], nn.ReLU) and m.mlp_s[2].p == 0.5 and m.mlp_w[2].p == 0.5
    m = MPQA("vit_tiny_patch16_224", act="gelu", head_drop=0.0, pretrained=False)
    assert isinstance(m.mlp_s[1], nn.GELU) and m.mlp_s[2].p == 0.0
    with pytest.raises(KeyError):
        MPQA("vit_tiny_patch16_224", act="swish", pretrained=False)
