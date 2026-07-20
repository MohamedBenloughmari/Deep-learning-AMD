from typing import Optional, Tuple

import torch


def move_batch(batch, device: torch.device, modality: str) -> Tuple:
    if modality == "trimodal":
        image, localiser, tab, label = batch
        return (
            image.to(device),
            localiser.to(device),
            tab.to(device),
            label.long().to(device),
        )
    image, localiser, label = batch
    return image.to(device), localiser.to(device), label.long().to(device)


def forward_model(model, batch, device: torch.device, modality: str):
    """Run the model on a batch and return (logits, labels)."""
    if modality == "trimodal":
        image, localiser, tab, label = move_batch(batch, device, modality)
        logits, _ = model(image, localiser, tab)
    else:
        image, localiser, label = move_batch(batch, device, modality)
        logits, _ = model(image, localiser)
    return logits, label


@torch.no_grad()
def gather_predictions(
    model,
    loader,
    device: torch.device,
    modality: str,
    loss_name: str,
    n_classes: int,
    amp: bool = False,
):
    """Collect predictions, labels and probabilities over a loader."""
    from amd_oct.losses import predict_classes, predict_probabilities
    from amd_oct.utils.device import autocast_context

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        with autocast_context(device, amp=amp):
            logits, labels = forward_model(model, batch, device, modality)
        probs = predict_probabilities(logits, loss_name, n_classes)
        preds = predict_classes(logits, loss_name, n_classes)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())
    import numpy as np

    probs = np.concatenate(all_probs, axis=0) if all_probs else None
    return all_preds, all_labels, probs
