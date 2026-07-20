from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def specificity(y_true, y_pred) -> float:
    """Mean per-class specificity: TN / (TN + FP)."""
    eps = 1e-6
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + (cm.sum(axis=1) - np.diag(cm)) + tp)
    return float(np.mean(tn / (tn + fp + eps)))


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None,
    n_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute the full metric suite used by the MICCAI Task 2 notebooks.

    Returns accuracy, micro/weighted F1, precision, recall, MCC, mean per-class
    specificity, QWK, and (optionally) one-vs-rest AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": specificity(y_true, y_pred),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }
    if y_probs is not None:
        try:
            if n_classes is None:
                n_classes = y_probs.shape[1]
            from sklearn.preprocessing import label_binarize

            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            if y_bin.shape[1] == 1:
                metrics["auc"] = float("nan")
            else:
                metrics["auc"] = float(roc_auc_score(y_bin, y_probs, multi_class="ovr", average="weighted"))
        except (ValueError, IndexError):
            metrics["auc"] = float("nan")
    return metrics


def composite_score(metrics: Dict[str, float]) -> float:
    """Composite validation score from the best MICCAI notebook.

    score = 0.1*F1 + 0.1*Specificity + 0.6*QWK + 0.2*MCC
    Uses the micro-averaged F1 to match ``miccai-oct3(1) copy 2.ipynb``.
    """
    return (
        0.1 * metrics.get("f1_micro", 0.0)
        + 0.1 * metrics.get("specificity", 0.0)
        + 0.6 * metrics.get("qwk", 0.0)
        + 0.2 * metrics.get("mcc", 0.0)
    )


def monitor_value(metrics: Dict[str, float], monitor: str = "score") -> float:
    """Extract the metric used for best-model selection."""
    if monitor == "score":
        return composite_score(metrics)
    if monitor not in metrics:
        raise KeyError(f"Unknown monitor '{monitor}'. Available: {sorted(metrics)}")
    return metrics[monitor]


def format_classification_report(y_true, y_pred, target_names=None) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
