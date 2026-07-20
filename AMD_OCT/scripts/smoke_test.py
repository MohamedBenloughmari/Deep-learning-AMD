"""Quick setup sanity check.

Run with:  python scripts/smoke_test.py

Verifies that the install is functional:
  - device detection (cuda / mps / cpu)
  - loss build + ordinal prediction for all four losses
  - metrics computation + composite score
  - tabular preprocessor (dynamic tab_dim, fixes the notebooks' hardcoded 7)
  - transforms for 1- and 3-channel inputs
  - a full EfficientNetV2-S trimodal model forward pass
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from amd_oct.data.tabular import TabularPreprocessor
from amd_oct.data.transforms import build_transforms
from amd_oct.losses import build_loss, predict_classes, predict_probabilities
from amd_oct.metrics import composite_score, compute_metrics
from amd_oct.models.registry import build_model
from amd_oct.utils import get_device


def main() -> None:
    n_classes = 13
    B = 4
    device = get_device()
    print(f"Device: {device}")

    print("\n[1/5] Losses + ordinal prediction")
    for name in ["cross_entropy", "focal", "corn", "coral"]:
        cfg = OmegaConf.create({"name": name, "class_weights": True, "gamma": 2.0})
        crit = build_loss(cfg, n_classes, class_counts=np.arange(1, n_classes + 1, dtype=float))
        out_dim = n_classes - 1 if name in ("corn", "coral") else n_classes
        logits = torch.randn(B, out_dim)
        labels = torch.randint(0, n_classes, (B,))
        loss = crit(logits, labels)
        preds = predict_classes(logits, name, n_classes)
        probs = predict_probabilities(logits, name, n_classes)
        assert preds.shape == (B,)
        assert probs.shape == (B, n_classes)
        assert abs(probs.sum(1).mean().item() - 1.0) < 1e-4
        print(f"  {name:14s} loss={loss.item():.4f}  OK")

    print("\n[2/5] Metrics + composite score")
    y = np.random.randint(0, n_classes, 100)
    p = np.random.randint(0, n_classes, 100)
    m = compute_metrics(y, p, n_classes=n_classes)
    print(f"  qwk={m['qwk']:.4f}  acc={m['accuracy']:.4f}  score={composite_score(m):.4f}  OK")

    print("\n[3/5] Tabular preprocessor (dynamic tab_dim)")
    import pandas as pd

    df = pd.DataFrame({
        "case": ["c1", "c2"], "label": [0, 1], "LOCALIZER": ["l", "l"],
        "split_type": ["train", "train"], "image": ["i", "i"],
        "age": [60, 70], "sex": ["M", "F"], "side_eye": ["L", "R"],
        "BScan": [3, 4], "num_current_visit": [1, 2],
    })
    tp = TabularPreprocessor()
    tp.fit_transform(df)
    print(f"  tab_dim={tp.tab_dim} (dynamic, not hardcoded)  OK")

    print("\n[4/5] Transforms (1- and 3-channel)")
    img = Image.new("RGB", (50, 50))
    tr3, _ = build_transforms(image_size=32, channels=3)
    tr1, _ = build_transforms(image_size=32, channels=1, normalize=False)
    assert tr3(img).shape[0] == 3 and tr1(img).shape[0] == 1
    print("  rgb=(3,32,32) gray=(1,32,32)  OK")

    print("\n[5/5] EfficientNetV2-S trimodal forward pass")
    cfg = OmegaConf.create({
        "name": "efficientnet_v2_s_trimodal", "embed_dim": 64, "d_model": 64,
        "dropout": 0.3, "use_layer_norm": False, "head_hidden": [64, 32],
        "backbone": {"pretrained": False, "freeze": False},
        "tabular_encoder": {"name": "mlp", "dropout": 0.3, "hidden": 32},
    })
    model = build_model(cfg, n_classes=n_classes, tab_dim=tp.tab_dim, loss_name="cross_entropy").to(device)
    model.eval()
    with torch.no_grad():
        img_t = torch.randn(2, 3, 32, 32, device=device)
        loc_t = torch.randn(2, 3, 32, 32, device=device)
        tab_t = torch.randn(2, tp.tab_dim, device=device)
        logits, feats = model(img_t, loc_t, tab_t)
    print(f"  logits={tuple(logits.shape)} features={tuple(feats.shape)} modality={model.modality}  OK")

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
