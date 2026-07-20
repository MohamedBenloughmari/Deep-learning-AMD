from typing import Sequence

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """MLP classification head over a fused feature vector.

    Matches the trimodal notebooks: Linear -> BN -> GELU -> Dropout (x2) -> Linear(out_dim).
    """

    def __init__(
        self,
        fusion_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (1024, 512),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_dim = fusion_dim
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DualBranchHead(nn.Module):
    """Head for dual-branch models: Linear -> BN -> ReLU -> Dropout (x2) -> Linear(out_dim).

    Mirrors ``DualBranchMedViT.classifier`` from the notebooks.
    """

    def __init__(
        self,
        fusion_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (512, 256),
        dropout: float = 0.4,
        dropout2: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_dim = fusion_dim
        for i, h in enumerate(hidden):
            d = dropout if i == 0 else dropout2
            act = nn.ReLU() if i == 0 else nn.ReLU()
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                act,
                nn.Dropout(d),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
