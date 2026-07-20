import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from amd_oct.utils.imports import require_extra
from amd_oct.utils.logging import get_logger

log = get_logger("amd_oct.models.mirage")


def _pair(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (int(x), int(x))


def _import_mirage(repo_path: Optional[str] = None):
    candidates = []
    if repo_path:
        candidates.append(Path(repo_path))
    candidates.append(Path("external/MIRAGE"))
    for cand in candidates:
        if cand.exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
    try:
        from mirage.input_adapters import PatchedInputAdapter  # type: ignore
        from mirage.model import MIRAGEModel  # type: ignore
        from mirage.utils import pair  # type: ignore
        return PatchedInputAdapter, MIRAGEModel, pair
    except ImportError:
        require_extra(
            "mirage",
            "mirage",
            hint="Clone https://github.com/j-morano/MIRAGE.git and set "
                 "model.backbone.repo_path, or add it to PYTHONPATH. "
                 "Also install safetensors and huggingface_hub: "
                 "pip install -e .[mirage]",
        )
    return None, None, None


class _Args:
    pass


def build_mirage_base(
    input_size=512,
    patch_size=32,
    modalities="bscan-slo",
    repo_path: Optional[str] = None,
) -> Tuple[nn.Module, _Args]:
    PatchedInputAdapter, MIRAGEModel, pair = _import_mirage(repo_path)

    in_domains = modalities.split("-")
    input_size = pair(input_size)
    patch_size_pair = pair(patch_size)

    args = _Args()
    args.in_domains = in_domains
    args.out_domains = in_domains
    args.model = "miragepre_base"
    args.num_global_tokens = 1
    args.drop_path = 0.0
    args.decoder_dim = 768
    args.decoder_depth = 2
    args.decoder_num_heads = 12
    args.decoder_use_task_queries = True
    args.decoder_use_xattn = True
    args.patch_size = {}
    args.input_size = {}
    args.grid_size = {}
    args.grid_sizes = {}
    for domain in in_domains:
        args.patch_size[domain] = list(patch_size_pair)
        args.input_size[domain] = list(input_size)
        gs = (input_size[0] // patch_size_pair[0], input_size[1] // patch_size_pair[1])
        args.grid_size[domain] = list(gs)
        args.grid_sizes[domain] = list(gs)

    input_adapters = {
        domain: PatchedInputAdapter(
            num_channels=1,
            stride_level=1,
            patch_size_full=patch_size_pair,
            image_size=input_size,
        )
        for domain in in_domains
    }

    model = MIRAGEModel(
        args=args,
        input_adapters=input_adapters,
        output_adapters=None,
        num_global_tokens=1,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
    )
    _load_mirage_weights(model)
    return model, args


def _load_mirage_weights(model: nn.Module) -> None:
    require_extra("safetensors", "mirage")
    require_extra("huggingface_hub", "mirage")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sf_path = hf_hub_download(repo_id="j-morano/MIRAGE-Base", filename="model.safetensors")
    state_dict = load_file(sf_path)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[len("model."):] if k.startswith("model.") else k] = v
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in cleaned.items() if k in model_keys}
    msg = model.load_state_dict(filtered, strict=False)
    log.info(
        f"Loaded MIRAGE weights: {len(filtered)} keys, "
        f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
    )


class MIRAGEClassifier(nn.Module):
    """MIRAGE-Base dual-input (bscan + slo) classifier.

    Inputs are 1-channel grayscale tensors in [0, 1] of shape ``(B, 1, S, S)``.
    Forward signature mirrors ``DualBranch``: ``forward(image, localiser)``
    returns ``(logits, features)``.
    """

    modality = "dual"
    in_channels = 1

    def __init__(
        self,
        n_classes: int,
        out_dim: Optional[int] = None,
        input_size: int = 512,
        patch_size: int = 32,
        modalities: str = "bscan-slo",
        repo_path: Optional[str] = None,
        freeze_backbone: bool = True,
        embed_dim: int = 768,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.mirage_model, self.args = build_mirage_base(
            input_size=input_size,
            patch_size=patch_size,
            modalities=modalities,
            repo_path=repo_path,
        )
        self.modalities: List[str] = modalities.split("-")
        self.num_global_tokens = self.args.num_global_tokens
        self.embed_dim = embed_dim
        if freeze_backbone:
            for p in self.mirage_model.parameters():
                p.requires_grad = False
        self.norm = nn.LayerNorm(self.embed_dim)
        head_out = out_dim if out_dim is not None else n_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, head_out),
        )

    def forward(self, image: torch.Tensor, localiser: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_dict = {}
        if "bscan" in self.modalities:
            x_dict["bscan"] = image
        if "slo" in self.modalities:
            x_dict["slo"] = localiser
        encoder_tokens, _masks = self.mirage_model(x_dict, mask_inputs=False)
        patch_features = encoder_tokens[:, :-self.num_global_tokens, :].mean(dim=1)
        features = self.norm(patch_features)
        logits = self.classifier(features)
        return logits, features
