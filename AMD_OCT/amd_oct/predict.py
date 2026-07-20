import os

import pandas as pd
from omegaconf import DictConfig

from amd_oct.inference import gather_predictions
from amd_oct.utils.logging import get_logger

log = get_logger("amd_oct.predict")


def _read_split_df(cfg: DictConfig, split: str) -> pd.DataFrame:
    data_cfg = cfg.data
    explicit = data_cfg.get("csv", {}).get(split)
    path = explicit or os.path.join(data_cfg.get("task_root", "data/Task_2"), f"df_task2_{split}.csv")
    return pd.read_csv(path)


def predict_cli(cfg: DictConfig) -> str:
    """Driver for ``amd-oct predict``. Writes a submission CSV for a split."""
    from amd_oct.data.loaders import build_dataloaders
    from amd_oct.models.registry import build_model
    from amd_oct.utils.checkpoint import load_checkpoint

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

    split = str(cfg.get("split", "test"))
    loader = {"train": loaders["train_loader"], "val": loaders["val_loader"], "test": loaders["test_loader"]}[split]
    modality = getattr(model, "modality", "trimodal")
    amp = bool(cfg.training.get("amp", False)) if cfg.get("training") else False

    preds, _labels, probs = gather_predictions(
        model, loader, device, modality,
        loss_name=stored_loss, n_classes=n_classes, amp=amp,
    )

    df = _read_split_df(cfg, split)
    id_col = "case" if "case" in df.columns else df.columns[0]
    out_df = pd.DataFrame({id_col: df[id_col].values[: len(preds)], "label": preds})
    if probs is not None:
        for k in range(probs.shape[1]):
            out_df[f"prob_{k}"] = probs[: len(preds), k]
    out_path = cfg.get("out", os.path.join("outputs", f"submission_{split}.csv"))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log.info(f"Wrote {len(out_df)} predictions -> {out_path}")
    return out_path


def _device(cfg: DictConfig):
    from amd_oct.utils.device import get_device

    return get_device(cfg.get("device"))
