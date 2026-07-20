import logging
from typing import Optional


def get_logger(name: str = "amd_oct") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_metrics(logger: logging.Logger, prefix: str, metrics: dict, epoch: Optional[int] = None) -> None:
    parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
    header = f"[{prefix}]"
    if epoch is not None:
        header += f" epoch={epoch}"
    logger.info(f"{header} " + " | ".join(parts))
