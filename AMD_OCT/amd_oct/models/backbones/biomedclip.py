import torch
import torch.nn as nn

from amd_oct.utils.imports import require_extra


class BiomedCLIPEncoder(nn.Module):
    """BiomedCLIP ViT-B/16 visual trunk (open_clip) -> (B, embed_dim).

    Requires the ``[biomedclip]`` extra: ``pip install -e .[biomedclip]``.
    Weights are downloaded from the HuggingFace Hub on first use.
    """

    in_channels = 3
    feature_dim = 768
    HF_REPO = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, embed_dim: int = 768, pretrained: bool = True, freeze: bool = False, **kwargs):
        super().__init__()
        require_extra("open_clip", "biomedclip", hint="open_clip_torch provides the BiomedCLIP backbone.")
        from open_clip import create_model_from_pretrained

        model, _ = create_model_from_pretrained(self.HF_REPO) if pretrained else (None, None)
        if model is None:
            from open_clip import create_model

            model = create_model("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.backbone = model.visual.trunk
        biomed_dim = self.backbone.embed_dim
        self.fc = nn.Linear(biomed_dim, embed_dim)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))
