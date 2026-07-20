import os
from typing import Optional

import torch

from amd_oct.utils.logging import get_logger

log = get_logger("amd_oct.checkpoint")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)
    log.info(f"Saved checkpoint -> {path} (epoch={epoch})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location=None,
    strict: bool = True,
) -> dict:
    if map_location is None:
        map_location = "cpu"
    state = torch.load(path, map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=strict)
    if missing:
        log.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    log.info(f"Loaded checkpoint <- {path}")
    return state


def load_state_dict_flexible(path: str, map_location=None) -> dict:
    if map_location is None:
        map_location = "cpu"
    return torch.load(path, map_location=map_location, weights_only=False)
