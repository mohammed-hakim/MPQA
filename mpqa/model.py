"""MPQA model: a timm backbone with two small heads (paper Fig. 3a)."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPQA(nn.Module):
    """backbone -> mlp_s: 4 quality scores (SNR, contrast, resolution, artifacts)
                -> mlp_w: 4 weight logits (how much each dimension counts for the overall score)"""
    def __init__(self, arch="vit_base_patch16_384", act="gelu", head_drop=0.0, dim=4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)   # -> (B, features)
        d = self.backbone.num_features
        A = {"relu": nn.ReLU, "gelu": nn.GELU}[act]
        self.mlp_s = nn.Sequential(nn.Linear(d, 256), A(), nn.Dropout(head_drop), nn.Linear(256, dim))
        self.mlp_w = nn.Sequential(nn.Linear(d, 512), A(), nn.Dropout(head_drop), nn.Linear(512, dim))

    def forward(self, x):
        f = self.backbone(x)
        return self.mlp_s(f), self.mlp_w(f)


def overall_vector(scores, w_logits):
    """(B,4), (B,4) -> (B,5): the 4 dimension scores followed by overall = sum(softmax(w) * s)."""
    overall = (F.softmax(w_logits, dim=-1) * scores).sum(dim=1, keepdim=True)
    return torch.cat([scores, overall], dim=1)
