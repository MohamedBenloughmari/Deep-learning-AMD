import torch
import torch.nn as nn
import torch.nn.functional as F


class CornLoss(nn.Module):
    """Conditional Ordinal Regression Network loss (Shi et al., 2021).

    The model outputs ``n_classes - 1`` logits eta_0, ..., eta_{K-2}, where
    eta_k parametrises the conditional probability P(y > k | y > k-1).

    For a sample with label y:
      - k = 0..y-1 : positive branch  -> BCE(eta_k, 1)
      - k = y       : negative branch  -> BCE(eta_k, 0)
      - k > y       : masked out (conditional not reached)
    """

    def __init__(self, n_classes: int, reduction: str = "mean"):
        super().__init__()
        self.n_classes = n_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        k = self.n_classes - 1
        ranks = torch.arange(k, device=targets.device).unsqueeze(0)
        labels = targets.unsqueeze(1)
        target = (labels > ranks).float()
        mask = (labels >= ranks).float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        masked = bce * mask
        if self.reduction == "sum":
            return masked.sum()
        denom = mask.sum().clamp(min=1.0)
        return masked.sum() / denom


def corn_probabilities(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert K-1 conditional logits to a full K-class probability matrix."""
    cond = torch.sigmoid(logits)
    batch = logits.size(0)
    probs = torch.zeros(batch, n_classes, device=logits.device)
    cum = torch.ones(batch, device=logits.device)
    for k in range(n_classes - 1):
        probs[:, k] = cum * (1.0 - cond[:, k])
        cum = cum * cond[:, k]
    probs[:, n_classes - 1] = cum
    return probs


def corn_predict(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    return corn_probabilities(logits, n_classes).argmax(dim=1)
