import os
from typing import Optional

import torch
from omegaconf import DictConfig

from amd_oct.train import evaluate_model
from amd_oct.utils.logging import get_logger, log_metrics

log = get_logger("amd_oct.evaluate")


def evaluate_cli(cfg: DictConfig) -> dict:
    """Driver for ``amd-oct evaluate``. Loads a checkpoint and evaluates a split."""
    from amd_oct.data.loaders import build_dataloaders
    from amd_oct.models.registry import build_model
    from amd_oct.utils.checkpoint import load_checkpoint
    from amd_oct.metrics import format_classification_report

    device = _device(cfg)
    n_classes = int(cfg.data.get("n_classes", 13))
    loaders = build_dataloaders(cfg, n_classes=n_classes, device=device)
    tab_dim = loaders["tab_dim"]

    loss_name = str(cfg.loss.get("name", "cross_entropy"))
    model = build_model(
        cfg.model, n_classes=n_classes, tab_dim=tab_dim, loss_name=loss_name
    ).to(device)

    checkpoint = cfg.get("checkpoint", "best.pth")
    state = load_checkpoint(checkpoint, model, map_location=str(device))
    stored_loss = (state or {}).get("loss") or loss_name
    log.info(f"Checkpoint epoch={state.get('epoch')} metrics={state.get('metrics')}")

    from amd_oct.losses import build_loss

    criterion = build_loss(cfg.loss, n_classes=n_classes, device=device)

    split = str(cfg.get("split", "test"))
    loader = {"train": loaders["train_loader"], "val": loaders["val_loader"], "test": loaders["test_loader"]}[split]
    modality = getattr(model, "modality", "trimodal")
    amp = bool(cfg.training.get("amp", False)) if cfg.get("training") else False

    metrics, preds, labels, _ = evaluate_model(
        model, loader, criterion, device, modality,
        loss_name=stored_loss, n_classes=n_classes, amp=amp, return_preds=True,
    )
    log_metrics(log, split, metrics)
    log.info("Classification report:\n" + format_classification_report(labels, preds))
    return metrics


def _device(cfg: DictConfig):
    from amd_oct.utils.device import get_device

    return get_device(cfg.get("device"))
