import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralLoss(nn.Module):
    """Rank-consistent ordinal cross-entropy (Cao et al., 2020).

    The model outputs ``n_classes - 1`` cumulative logits eta_k parametrising
    P(y > k). All K-1 ranks share weights with per-rank biases; here we treat
    the provided logits directly as the K-1 cumulative logits (the model head
    is responsible for producing them). Loss = sum of per-rank BCE.
    """

    def __init__(self, n_classes: int, reduction: str = "mean"):
        super().__init__()
        self.n_classes = n_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ranks = torch.arange(self.n_classes - 1, device=targets.device).unsqueeze(0)
        labels = targets.unsqueeze(1)
        target = (labels > ranks).float()
        if self.reduction == "sum":
            return F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


def coral_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 cumulative logits to K-class probabilities.

    P(y=k) = P(y>k-1) - P(y>k), with P(y>-1)=1 and P(y>K-1)=0.
    """
    cum = torch.sigmoid(logits)
    batch = logits.size(0)
    ones = torch.ones(batch, 1, device=logits.device)
    zeros = torch.zeros(batch, 1, device=logits.device)
    cum_full = torch.cat([ones, cum, zeros], dim=1)
    probs = cum_full[:, :-1] - cum_full[:, 1:]
    return probs


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    return coral_probabilities(logits).argmax(dim=1)
