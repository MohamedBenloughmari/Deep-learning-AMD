import torch
import torch.nn as nn
from torchvision import models


class EfficientNetV2SEncoder(nn.Module):
    """EfficientNetV2-S backbone (torchvision ImageNet weights) -> (B, embed_dim)."""

    in_channels = 3
    feature_dim = 1280

    def __init__(self, embed_dim: int = 768, pretrained: bool = True, freeze: bool = False, **kwargs):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_v2_s(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(in_features, embed_dim)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))
