
import torch.nn as nn
from omegaconf import DictConfig

from amd_oct.config import head_output_dim
from amd_oct.models.dual_branch import DualBranch
from amd_oct.models.fusion import TrimodalFusion
from amd_oct.models.tabular_encoders import build_tabular_encoder
from amd_oct.utils.logging import get_logger

log = get_logger("amd_oct.models.registry")


def _backbone_factory(name: str, cfg: DictConfig, embed_dim: int) -> nn.Module:
    name = name.lower()
    if name in ("efficientnet_v2_s", "efficientnet", "effnet"):
        from amd_oct.models.backbones.efficientnet import EfficientNetV2SEncoder
        return EfficientNetV2SEncoder(embed_dim=embed_dim, **_bb_kwargs(cfg))
    if name in ("convnext_base", "convnext"):
        from amd_oct.models.backbones.convnext import ConvNeXtBaseEncoder
        return ConvNeXtBaseEncoder(embed_dim=embed_dim, **_bb_kwargs(cfg))
    if name in ("biomedclip", "biomed_clip"):
        from amd_oct.models.backbones.biomedclip import BiomedCLIPEncoder
        return BiomedCLIPEncoder(embed_dim=embed_dim, **_bb_kwargs(cfg))
    if name == "medvit":
        from amd_oct.models.backbones.medvit import MedViTEncoder
        return MedViTEncoder(embed_dim=embed_dim, **_bb_kwargs(cfg))
    raise ValueError(f"Unknown backbone: {name}")


def _bb_kwargs(cfg: DictConfig) -> dict:
    bb = cfg.get("backbone", {}) or {}
    return {
        "pretrained": bool(bb.get("pretrained", True)),
        "freeze": bool(bb.get("freeze", False)),
        **{k: v for k, v in bb.items() if k not in ("pretrained", "freeze")},
    }


def build_model(
    model_cfg: DictConfig,
    n_classes: int,
    tab_dim: int = 0,
    loss_name: str = "cross_entropy",
) -> nn.Module:
    """Construct a model from ``cfg.model``.

    Supports five architectures selected by ``cfg.model.name``:
      - ``efficientnet_v2_s_trimodal`` (or ``efficientnet``) -> TrimodalFusion
      - ``convnext_base_trimodal`` (or ``convnext``)         -> TrimodalFusion
      - ``biomedclip_trimodal`` (or ``biomedclip``)          -> TrimodalFusion
      - ``medvit_dual`` (or ``medvit``)                      -> DualBranch
      - ``mirage_dual`` (or ``mirage``)                      -> MIRAGEClassifier

    The head output dim is ``n_classes`` for CE/Focal and ``n_classes - 1``
    for ordinal losses (CORN/CORAL).
    """
    name = str(model_cfg.name).lower()
    out_dim = head_output_dim(loss_name, n_classes)
    embed_dim = int(model_cfg.get("embed_dim", 768))
    d_model = int(model_cfg.get("d_model", embed_dim))
    dropout = float(model_cfg.get("dropout", 0.3))
    use_layer_norm = bool(model_cfg.get("use_layer_norm", False))

    if "mirage" in name:
        from amd_oct.models.backbones.mirage import MIRAGEClassifier
        bb = model_cfg.get("backbone", {}) or {}
        model = MIRAGEClassifier(
            n_classes=n_classes,
            out_dim=out_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            input_size=int(bb.get("input_size", 512)),
            patch_size=int(bb.get("patch_size", 32)),
            modalities=str(bb.get("modalities", "bscan-slo")),
            repo_path=bb.get("repo_path"),
            freeze_backbone=bool(bb.get("freeze", True)),
        )
        log.info(f"Built MIRAGEClassifier (out_dim={out_dim}, embed_dim={embed_dim})")
        return model

    if "medvit" in name:
        bb = _bb_kwargs(model_cfg)
        image_encoder = _backbone_factory("medvit", model_cfg, embed_dim)
        localiser_encoder = _backbone_factory("medvit", model_cfg, embed_dim)
        model = DualBranch(
            image_encoder=image_encoder,
            localiser_encoder=localiser_encoder,
            embed_dim=embed_dim,
            out_dim=out_dim,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        log.info(f"Built DualBranch(MedViT) (out_dim={out_dim}, embed_dim={embed_dim})")
        return model

    backbone_name = name.replace("_trimodal", "")
    image_encoder = _backbone_factory(backbone_name, model_cfg, embed_dim)
    localiser_encoder = _backbone_factory(backbone_name, model_cfg, embed_dim)
    tab_encoder = None
    if tab_dim > 0 and not bool(model_cfg.get("disable_tabular", False)):
        tab_cfg = model_cfg.get("tabular_encoder", {"name": "mlp"})
        tab_encoder = build_tabular_encoder(tab_cfg, tab_dim, d_model)
    model = TrimodalFusion(
        image_encoder=image_encoder,
        localiser_encoder=localiser_encoder,
        tab_encoder=tab_encoder,
        embed_dim=embed_dim,
        d_model=d_model,
        out_dim=out_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
        head_hidden=tuple(model_cfg.get("head_hidden", (1024, 512))),
    )
    log.info(
        f"Built TrimodalFusion({backbone_name}) "
        f"(out_dim={out_dim}, embed_dim={embed_dim}, tab_dim={tab_dim}, "
        f"tab_encoder={'mlp' if tab_encoder is None else tab_cfg.get('name','mlp')})"
    )
    return model
