import importlib
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from amd_oct.utils.imports import require_extra
from amd_oct.utils.logging import get_logger

log = get_logger("amd_oct.models.medvit")


def _import_medvit_class(repo_path: Optional[str] = None):
    """Import ``MedViT`` from the external MedViTV2 repo.

    Tries (in order):
      1. ``repo_path`` from config (added to sys.path)
      2. an already-installed ``medvit`` / ``MedViT`` module on the path
      3. ``external/MedViTV2`` relative to the package
    """
    candidates = []
    if repo_path:
        candidates.append(Path(repo_path))
    candidates.append(Path("external/MedViTV2"))
    for cand in candidates:
        if cand.exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    try:
        from MedViT import MedViT  # type: ignore
        return MedViT
    except ImportError:
        pass
    try:
        mod = importlib.import_module("medvit")
        return getattr(mod, "MedViT")
    except ImportError:
        require_extra(
            "MedViT",
            "medvit",
            hint="Clone https://github.com/Omid-Nejati/MedViTV2.git and set "
                 "model.backbone.repo_path, or add it to PYTHONPATH. "
                 "Also install natten.",
        )
    return None


class MedViTEncoder(nn.Module):
    """MedViT backbone (MedViTV2 repo) with classification head removed.

    Returns ``(B, embed_dim)`` where ``embed_dim`` defaults to the MedViT
    ``proj_head`` input feature dim (512 for MedViT-Small).
    """

    in_channels = 3

    def __init__(
        self,
        embed_dim: int = 512,
        pretrained: bool = False,
        freeze: bool = False,
        repo_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        stem_chs=(64, 32, 64),
        depths=(2, 2, 6, 2),
        dims=(64, 128, 320, 512),
        path_dropout: float = 0.2,
        num_classes_head: int = 1000,
        **kwargs,
    ):
        super().__init__()
        MedViT = _import_medvit_class(repo_path)
        self.backbone = MedViT(
            num_classes=num_classes_head,
            stem_chs=list(stem_chs),
            depths=list(depths),
            dims=list(dims),
            path_dropout=path_dropout,
        )
        feature_dim = self.backbone.proj_head[0].in_features
        self.backbone.proj_head = nn.Identity()
        self.feature_dim = feature_dim
        if embed_dim is None or embed_dim == feature_dim:
            self.fc = nn.Identity()
            self.out_dim = feature_dim
        else:
            self.fc = nn.Linear(feature_dim, embed_dim)
            self.out_dim = embed_dim
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(sd, strict=False)
            log.info(f"Loaded MedViT weights from {weights_path}")
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))
