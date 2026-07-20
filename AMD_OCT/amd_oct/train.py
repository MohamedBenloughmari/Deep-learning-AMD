import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from amd_oct.inference import forward_model, gather_predictions
from amd_oct.losses import build_loss, predict_classes, predict_probabilities
from amd_oct.metrics import (
    composite_score,
    compute_metrics,
    format_classification_report,
    monitor_value,
)
from amd_oct.utils.checkpoint import save_checkpoint
from amd_oct.utils.device import autocast_context, get_device
from amd_oct.utils.logging import get_logger, log_metrics
from amd_oct.utils.seed import set_seed

log = get_logger("amd_oct.train")


def build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 1e-2))
    head_lr = cfg.training.get("head_lr", None)
    if head_lr is None or float(head_lr) == lr:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head.") or name.startswith("classifier") or ".head." in name or ".classifier" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr})
    if head_params:
        groups.append({"params": head_params, "lr": float(head_lr)})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_scheduler(optimizer, cfg: DictConfig, steps_per_epoch: int):
    sched_cfg = cfg.training.get("scheduler", {})
    name = str(sched_cfg.get("name", "cosine")).lower()
    epochs = int(cfg.training.get("epochs", 5))
    step_mode = str(sched_cfg.get("step", "epoch")).lower()
    t_max = steps_per_epoch * epochs if step_mode == "batch" else epochs
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(t_max, 1))
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    modality: str,
    loss_name: str,
    n_classes: int,
    scheduler=None,
    scheduler_step: str = "epoch",
    amp: bool = False,
    scaler=None,
) -> dict:
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        logits, labels = forward_model(model, batch, device, modality)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None and scheduler_step == "batch":
            scheduler.step()
        running_loss += loss.item()
        preds = predict_classes(logits, loss_name, n_classes)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    metrics = compute_metrics(all_labels, all_preds, n_classes=n_classes)
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    modality: str,
    loss_name: str,
    n_classes: int,
    amp: bool = False,
    return_preds: bool = False,
):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        with autocast_context(device, amp=amp):
            logits, labels = forward_model(model, batch, device, modality)
            loss = criterion(logits, labels)
        running_loss += loss.item()
        probs = predict_probabilities(logits, loss_name, n_classes)
        preds = predict_classes(logits, loss_name, n_classes)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())
    probs = np.concatenate(all_probs, axis=0) if all_probs else None
    metrics = compute_metrics(all_labels, all_preds, y_probs=probs, n_classes=n_classes)
    metrics["loss"] = running_loss / max(len(loader), 1)
    metrics["score"] = composite_score(metrics)
    if return_preds:
        return metrics, all_preds, all_labels, probs
    return metrics


def train(cfg: DictConfig) -> dict:
    """Full training driver invoked by ``amd-oct train``."""
    from amd_oct.data.loaders import build_dataloaders
    from amd_oct.models.registry import build_model

    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = get_device(cfg.get("device"))
    log.info(f"Device: {device}")

    n_classes = int(cfg.data.get("n_classes", 13))
    loaders = build_dataloaders(cfg, n_classes=n_classes, device=device)
    class_counts = loaders["class_counts"]
    tab_dim = loaders["tab_dim"]

    loss_name = str(cfg.loss.get("name", "cross_entropy"))
    model = build_model(
        cfg.model, n_classes=n_classes, tab_dim=tab_dim, loss_name=loss_name
    ).to(device)

    criterion = build_loss(cfg.loss, n_classes=n_classes, class_counts=class_counts, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(loaders["train_loader"]))
    scheduler_step = str(cfg.training.get("scheduler", {}).get("step", "epoch")).lower()

    modality = getattr(model, "modality", "trimodal")
    amp = bool(cfg.training.get("amp", False))
    scaler = torch.cuda.amp.GradScaler() if (amp and device.type == "cuda") else None
    epochs = int(cfg.training.get("epochs", 5))
    monitor = str(cfg.training.get("monitor", "score"))
    out_dir = cfg.get("output_dir", os.path.join("outputs", cfg.get("run_name", "run")))
    os.makedirs(out_dir, exist_ok=True)

    best = -float("inf")
    best_path = os.path.join(out_dir, "best.pth")
    last_path = os.path.join(out_dir, "last.pth")
    history: List[dict] = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, loaders["train_loader"], criterion, optimizer, device,
            modality, loss_name, n_classes,
            scheduler=scheduler, scheduler_step=scheduler_step, amp=amp, scaler=scaler,
        )
        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()
        val_metrics = evaluate_model(
            model, loaders["val_loader"], criterion, device,
            modality, loss_name, n_classes, amp=amp,
        )
        log_metrics(log, "train", train_metrics, epoch=epoch)
        log_metrics(log, "val", val_metrics, epoch=epoch)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        current = monitor_value(val_metrics, monitor)
        if current > best:
            best = current
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch=epoch,
                metrics=val_metrics, extra={"monitor": monitor, "loss": loss_name},
            )
        save_checkpoint(last_path, model, optimizer, scheduler, epoch=epoch, metrics=val_metrics)

    test_metrics, test_preds, test_labels, _ = evaluate_model(
        model, loaders["test_loader"], criterion, device,
        modality, loss_name, n_classes, amp=amp, return_preds=True,
    )
    log_metrics(log, "test", test_metrics)
    log.info("Test classification report:\n" + format_classification_report(test_labels, test_preds))

    return {"best": best, "best_path": best_path, "test": test_metrics, "history": history}
