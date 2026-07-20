from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from amd_oct.losses.coral import CoralLoss, coral_predict, coral_probabilities
from amd_oct.losses.corn import CornLoss, corn_predict, corn_probabilities
from amd_oct.losses.focal import FocalLoss

ORDINAL = {"corn", "coral"}


def is_ordinal(name: str) -> bool:
    return str(name).lower() in ORDINAL


def loss_output_dim(name: str, n_classes: int) -> int:
    return n_classes - 1 if is_ordinal(name) else n_classes


def _class_weight_tensor(class_counts: Optional[np.ndarray], cfg: DictConfig) -> Optional[torch.Tensor]:
    use_weights = bool(cfg.get("class_weights", True)) if class_counts is not None else False
    if not use_weights:
        return None
    weight = 1.0 / np.asarray(class_counts, dtype=np.float64)
    weight = weight / weight.sum() * len(weight)
    return torch.as_tensor(weight, dtype=torch.float32)


def build_loss(
    loss_cfg: DictConfig,
    n_classes: int,
    class_counts: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Build a loss function from ``cfg.loss``.

    Supported: ``cross_entropy``, ``focal``, ``corn``, ``coral``.
    Class weights (``1 / class_count``, normalized) are applied to CE and Focal.
    """
    name = str(loss_cfg.get("name", "cross_entropy")).lower()

    if name == "cross_entropy":
        weight = _class_weight_tensor(class_counts, loss_cfg)
        criterion = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        )
    elif name == "focal":
        weight = _class_weight_tensor(class_counts, loss_cfg)
        criterion = FocalLoss(
            gamma=float(loss_cfg.get("gamma", 2.0)),
            weight=weight,
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        )
    elif name == "corn":
        criterion = CornLoss(n_classes=n_classes)
    elif name == "coral":
        criterion = CoralLoss(n_classes=n_classes)
    else:
        raise ValueError(f"Unknown loss: {name}")

    if device is not None:
        criterion = criterion.to(device)
    return criterion


def predict_probabilities(logits: torch.Tensor, loss_name: str, n_classes: int) -> torch.Tensor:
    name = str(loss_name).lower()
    if name == "corn":
        return corn_probabilities(logits, n_classes)
    if name == "coral":
        return coral_probabilities(logits)
    return F.softmax(logits, dim=1)


def predict_classes(logits: torch.Tensor, loss_name: str, n_classes: int) -> torch.Tensor:
    name = str(loss_name).lower()
    if name == "corn":
        return corn_predict(logits, n_classes)
    if name == "coral":
        return coral_predict(logits)
    return logits.argmax(dim=1)
