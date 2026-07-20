from typing import Optional

import torch
import torch.nn as nn

from amd_oct.models.heads import FusionHead


class TrimodalFusion(nn.Module):
    """OCT image + localizer + tabular fusion model.

    Two image encoders (one for OCT, one for localizer), a tabular encoder,
    an optional LayerNorm on the image features, and an MLP fusion head.
    Forward returns ``(logits, features)`` where ``features`` is the fused vector.
    """

    modality = "trimodal"
    in_channels = 3

    def __init__(
        self,
        image_encoder: nn.Module,
        localiser_encoder: nn.Module,
        tab_encoder: Optional[nn.Module],
        embed_dim: int,
        d_model: int,
        out_dim: int,
        use_layer_norm: bool = False,
        dropout: float = 0.3,
        head_hidden=(1024, 512),
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.localiser_encoder = localiser_encoder
        self.tab_encoder = tab_encoder
        self.img_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        self.loc_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        tab_out = d_model if tab_encoder is not None else 0
        fusion_dim = 2 * embed_dim + tab_out
        self.head = FusionHead(fusion_dim, out_dim, hidden=head_hidden, dropout=dropout)

    def forward(
        self,
        image: torch.Tensor,
        localiser: torch.Tensor,
        tab: Optional[torch.Tensor] = None,
    ):
        img_f = self.img_norm(self.image_encoder(image))
        loc_f = self.loc_norm(self.localiser_encoder(localiser))
        feats = [img_f, loc_f]
        if self.tab_encoder is not None and tab is not None:
            feats.append(self.tab_encoder(tab))
        combined = torch.cat(feats, dim=-1)
        logits = self.head(combined)
        return logits, combined
