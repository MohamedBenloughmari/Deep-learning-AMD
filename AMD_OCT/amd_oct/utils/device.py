import logging
import os
import warnings

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Return the best available torch device.

    Priority order (unless ``preferred`` is given):
        1. cuda          (NVIDIA / ROCm)
        2. mps           (Apple Silicon)
        3. cpu

    ``preferred`` accepts anything torch understands: ``"cuda"``, ``"cuda:1"``,
    ``"mps"``, ``"cpu"``.
    """
    if preferred:
        device = torch.device(preferred)
        _warn_unavailable(device)
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.backends.mps.is_built()
        except Exception:
            pass
        return torch.device("mps")
    return torch.device("cpu")


def _warn_unavailable(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back may fail.", RuntimeWarning)
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            warnings.warn("MPS requested but not available.", RuntimeWarning)


def device_supports_amp(device: torch.device) -> bool:
    """Whether autocast AMP is safe to enable on ``device``."""
    return device.type == "cuda"


def autocast_context(device: torch.device, amp: bool = False):
    """Return an autocast context manager that is a no-op on unsupported devices."""
    if amp and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    if amp and device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def set_cudnn(device: torch.device, benchmark: bool = True, deterministic: bool = False) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic


def worker_init_seed(worker_id: int) -> None:
    """PyTorch DataLoader worker_init_fn for reproducible augmentation."""
    import random

    import numpy as np

    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if os.environ.get("PYTHONHASHSEED") is None:
        os.environ["PYTHONHASHSEED"] = "0"
    return logging.getLogger("amd_oct")
