from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


ORDINAL_LOSSES = {"corn", "coral"}


def is_ordinal_loss(loss_name: str) -> bool:
    return str(loss_name).lower() in ORDINAL_LOSSES


def head_output_dim(loss_name: str, n_classes: int) -> int:
    """Number of logits the model head must produce for a given loss."""
    return n_classes - 1 if is_ordinal_loss(loss_name) else n_classes


def model_modality(model_cfg: DictConfig) -> str:
    """Return ``'trimodal'`` or ``'dual'`` based on model config."""
    name = str(model_cfg.get("name", "")).lower()
    if "mirage" in name or "medvit" in name or "dual" in name:
        return "dual"
    return model_cfg.get("modality", "trimodal")


def model_in_channels(model_cfg: DictConfig) -> int:
    """Expected input channels for the model's image branch."""
    name = str(model_cfg.get("name", "")).lower()
    if "mirage" in name:
        return 1
    return model_cfg.get("in_channels", 3)


def to_container(cfg: DictConfig, resolve: bool = True) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=resolve)


def dump_config(cfg: DictConfig, path: str) -> None:
    OmegaConf.save(cfg, path)
