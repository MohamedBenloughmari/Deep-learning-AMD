# AMD-OCT

A clean, modular PyTorch package for **Age-related Macular Degeneration (AMD) progression grading** on the [MICCAI Task 2](https://www.kaggle.com/datasets/mohabenloughmari/miccai-task2-full) OCT dataset. It consolidates 17 experimental notebooks into a single, configurable, device-agnostic codebase.

The task is **13-class ordinal classification** of OCT B-scan volumes. Inputs are trimodal:

- **OCT B-scan** image
- **LOCALIZER** (fundus / SLO) image
- **Tabular** features (`age`, `sex`, `side_eye`, `BScan`, `num_current_visit`)

The primary metric is **Quadratic Weighted Kappa (QWK)**, reported alongside a composite score
`0.1·F1 + 0.1·Specificity + 0.6·QWK + 0.2·MCC`.

---

## Highlights

- **5 backbone architectures** behind a unified model registry
- **Trimodal fusion** (image + localizer + tabular) as the default; image-only is a config flag
- **Ordinal-aware losses** (CORN, CORAL) to directly target the QWK metric, alongside CrossEntropy and FocalLoss
- **CUDA + Apple MPS + CPU** support via a single `get_device()` helper
- **Hydra configuration** with composable experiment configs
- **CLI** for the full lifecycle: `amd-oct {prepare-data, train, evaluate, predict}`

---

## Installation

```bash
cd AMD_OCT
pip install -e .                       # core (EfficientNet, ConvNeXt)
pip install -e .[biomedclip]           # + BiomedCLIP
pip install -e .[mirage]               # + MIRAGE ViT-Base
pip install -e .[medvit]               # + MedViT (requires natten)
pip install -e .[all]                  # everything except natten
```

### External backbone repos

Two backbones are not on PyPI and must be cloned alongside this package:

```bash
# MedViT (https://github.com/Omid-Nejati/MedViTV2)
git clone https://github.com/Omid-Nejati/MedViTV2.git external/MedViTV2
pip install natten==0.17.3             # match your torch/cuda build

# MIRAGE (https://github.com/j-morano/MIRAGE)
git clone https://github.com/j-morano/MIRAGE.git external/MIRAGE
```

Set their paths in the model config (`medvit.repo_path`, `mirage.repo_path`) or add them to `PYTHONPATH`. Weights for MIRAGE are downloaded automatically from the HuggingFace Hub (`j-morano/MIRAGE-Base`).

---

## Project layout

```
AMD_OCT/
├── pyproject.toml
├── requirements.txt
├── README.md
├── amd_oct/
│   ├── __init__.py
│   ├── __main__.py            # python -m amd_oct
│   ├── cli.py                 # amd-oct <command>  (Hydra entry point)
│   ├── config.py              # OmegaConf schema + defaults
│   ├── metrics.py             # QWK, MCC, F1, specificity, composite score, AUC
│   ├── train.py               # training loop + driver
│   ├── evaluate.py            # evaluation driver
│   ├── predict.py             # inference + submission CSV
│   ├── utils/                 # device, seed, logging, checkpoint, import guards
│   ├── data/                  # prepare, dataset, transforms, tabular, samplers, loaders
│   ├── models/                # registry, fusion, dual_branch, heads, tabular_encoders
│   │   └── backbones/         # efficientnet, convnext, biomedclip, medvit, mirage
│   ├── losses/                # focal, corn, coral, factory
│   └── configs/               # Hydra YAML configs (model/loss/experiment)
└── scripts/
```

---

## Data preparation

Datasets are downloaded with `kagglehub` (recommended) or reassembled from split zip parts.

```bash
# Option A: download from Kaggle via kagglehub
amd-oct prepare-data --config-name=default prepare.source=kaggle prepare.out_dir=data/

# Option B: reassemble Task_2.zip.001/.002/... and extract
amd-oct prepare-data --config-name=default prepare.source=zip prepare.parts_dir=data/ prepare.out_dir=data/

# Optionally build a 10% subset for fast iteration
amd-oct prepare-data --config-name=default prepare.source=subset prepare.src_dir=data/Task_2 prepare.out_dir=data_out10/Task_2 prepare.fraction=0.1
```

This produces:

```
data/Task_2/{train,val,test}/                # images
data/df_task2_{train,val,test}.csv           # split manifests
```

The expected CSV schema (auto-built by `prepare-data`) is:

| case | image | LOCALIZER | label | split_type | BScan | age | sex | side_eye | num_current_visit |
|------|-------|-----------|-------|------------|-------|-----|-----|----------|-------------------|

`label` is an integer in `[0, 12]`. The `case` column groups B-scans from the same patient visit and is excluded from training features.

---

## Training

```bash
# Best baseline (EfficientNetV2-S trimodal, 420px, composite-score selection)
amd-oct train --config-name=efficientnet_v2_s_best

# ConvNeXt-Base trimodal
amd-oct train --config-name=convnext_base

# BiomedCLIP trimodal (requires [biomedclip] extra)
amd-oct train --config-name=biomedclip

# MedViT dual-branch (requires MedViTV2 repo)
amd-oct train --config-name=medvit_dual

# MIRAGE ViT-Base dual-input (requires MIRAGE repo + [mirage] extra)
amd-oct train --config-name=mirage_dual
```

Override anything on the command line:

```bash
amd-oct train --config-name=efficientnet_v2_s_best \
    training.epochs=20 \
    training.lr=3e-4 \
    loss.name=focal loss.gamma=2.0 \
    model.backbone.pretrained=false \
    data.image_size=384
```

### Switching the loss to target QWK

The notebooks plateau at QWK ≈ 0.003–0.07 despite 70–84% accuracy because cross-entropy
ignores the ordinal structure of the 13 labels. Use an ordinal loss:

```bash
amd-oct train --config-name=efficientnet_v2_s_best loss.name=corn
amd-oct train --config-name=efficientnet_v2_s_best loss.name=coral
```

CORN/CORAL output `n_classes - 1` cumulative logits; prediction is handled automatically
by the postprocessing in `amd_oct.losses`.

### Device support

`get_device()` picks the best available accelerator:

| Platform | Device | AMP |
|----------|--------|-----|
| NVIDIA GPU | `cuda` | `torch.cuda.amp.autocast` (bfloat16/fp16) |
| Apple Silicon | `mps` | disabled by default (enable with `training.amp=true`) |
| other | `cpu` | disabled |

Force a device with `device=cpu`, `device=mps`, or `device=cuda:1`.

---

## Evaluation

```bash
amd-oct evaluate --config-name=efficientnet_v2_s_best \
    checkpoint=outputs/efficientnet_v2_s_best/best.pth
```

Prints per-class precision/recall/F1 plus the composite score and QWK.

## Prediction / submission

```bash
amd-oct predict --config-name=efficientnet_v2_s_best \
    checkpoint=outputs/efficientnet_v2_s_best/best.pth \
    split=test \
    out=outputs/submission.csv
```

Writes `case_id,label` (or `image,label`) predictions for the requested split.

---

## Models

| Config name | Backbone | Modality | Pretrained | Image size |
|-------------|----------|----------|------------|------------|
| `efficientnet_v2_s_trimodal` | EfficientNetV2-S (torchvision) | trimodal | ImageNet | 420 |
| `convnext_base_trimodal` | ConvNeXt-Base (torchvision) | trimodal | ImageNet | 224 |
| `biomedclip_trimodal` | BiomedCLIP ViT-B/16 (open_clip) | trimodal | PubMedBERT | 224 |
| `medvit_dual` | MedViT-Small (MedViTV2 repo) | dual-branch | from scratch | 336 |
| `mirage_dual` | MIRAGE-Base ViT-B (MIRAGE repo) | dual-input | HF `j-morano/MIRAGE-Base` | 512 |

The tabular encoder is swappable (`TabularMLP` or `TabularAttention`) for trimodal models.

## Losses

| Name | Output dim | Notes |
|------|-----------|-------|
| `cross_entropy` | `n_classes` | class-weighted (`1/class_count`) |
| `focal` | `n_classes` | `gamma` configurable |
| `corn` | `n_classes - 1` | Conditional Ordinal Regression (Shi et al. 2021) |
| `coral` | `n_classes - 1` | Rank-consistent ordinal CE (Cao et al. 2020) |

## Metrics

All of `accuracy`, `f1` (micro + weighted), `matthews_corrcoef`, mean per-class `specificity`,
`quadratic_weighted_kappa`, one-vs-rest `auc`, and the composite
`score = 0.1·F1 + 0.1·Spec + 0.6·QWK + 0.2·MCC`.

---

## Programmatic use

```python
from amd_oct.utils.device import get_device
from amd_oct.models.registry import build_model
from amd_oct.losses import build_loss, predict_classes
from amd_oct.data.loaders import build_dataloaders
from amd_oct.train import train

device = get_device()                 # cuda | mps | cpu
loaders = build_dataloaders(cfg.data, device)
model = build_model(cfg.model, n_classes=13, tab_dim=loaders["tab_dim"]).to(device)
criterion = build_loss(cfg.loss, n_classes=13, class_counts=loaders["class_counts"]).to(device)
train(cfg, model, loaders, criterion, device)
```

---

## Notes on the refactor

- The original notebooks hardcoded `tab_dim=7`; this package derives `tab_dim` dynamically from the fitted tabular preprocessor (see `amd_oct/data/tabular.py`).
- Class weights use `1/class_count` and are applied to both the loss and the `WeightedRandomSampler`.
- The MIRAGE per-batch `scheduler.step()` is preserved as a configurable option (`training.scheduler_step=batch`).
- Best-model selection defaults to the composite `score` (matches the best notebook); switch with `training.monitor=accuracy|qwk|score`.

## License

MIT — see `LICENSE` in the repository root.
