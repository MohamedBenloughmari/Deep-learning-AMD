import torch
import torch.nn as nn

from amd_oct.models.heads import DualBranchHead


class DualBranch(nn.Module):
    """Two independent image encoders (OCT + localizer) fused into a head.

    Used by the MedViT dual-branch model. Forward returns ``(logits, features)``.
    """

    modality = "dual"
    in_channels = 3

    def __init__(
        self,
        image_encoder: nn.Module,
        localiser_encoder: nn.Module,
        embed_dim: int,
        out_dim: int,
        use_layer_norm: bool = False,
        dropout: float = 0.4,
        dropout2: float = 0.3,
        head_hidden=(512, 256),
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.localiser_encoder = localiser_encoder
        self.img_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        self.loc_norm = nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity()
        fusion_dim = 2 * embed_dim
        self.head = DualBranchHead(
            fusion_dim, out_dim, hidden=head_hidden, dropout=dropout, dropout2=dropout2
        )

    def forward(self, image: torch.Tensor, localiser: torch.Tensor):
        img_f = self.img_norm(self.image_encoder(image))
        loc_f = self.loc_norm(self.localiser_encoder(localiser))
        combined = torch.cat([img_f, loc_f], dim=-1)
        logits = self.head(combined)
        return logits, combined
