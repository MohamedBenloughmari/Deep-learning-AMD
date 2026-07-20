import torch
import torch.nn as nn
from torchvision import models


class ConvNeXtBaseEncoder(nn.Module):
    """ConvNeXt-Base backbone (torchvision ImageNet weights) -> (B, embed_dim).

    The localizer branch can freeze all but the last ``freeze_unfreeze_last`` blocks.
    """

    in_channels = 3
    feature_dim = 1024

    def __init__(
        self,
        embed_dim: int = 768,
        pretrained: bool = True,
        freeze: bool = False,
        unfreeze_last: int = 0,
        **kwargs,
    ):
        super().__init__()
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_base(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(nn.Flatten(1))
        self.fc = nn.Linear(in_features, embed_dim)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last > 0:
                for p in self.backbone.features[-unfreeze_last:].parameters():
                    p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))
