"""
common.py
─────────
Shared utilities used across both the translation and sentiment modules.
Includes: random seed setting, GPU configuration, directory creation, logging.
"""

import os
import random
import logging
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow to ensure reproducibility.

    Args:
        seed: Integer seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# GPU Configuration
# ─────────────────────────────────────────────────────────────────────────────

def configure_gpu() -> None:
    """
    Enable TensorFlow GPU memory growth to prevent OOM errors on shared machines.
    Falls back silently if no GPU is available or TensorFlow is not installed.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] {len(gpus)} GPU(s) configured with memory growth enabled.")
        else:
            print("[INFO] No GPU detected — running on CPU.")
    except Exception as exc:
        print(f"[WARNING] GPU configuration failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# File System Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: str) -> None:
    """
    Create one or more directories (and their parents) if they do not exist.

    Args:
        *dirs: Variable number of directory path strings.
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def bytes_to_human(num_bytes: int) -> str:
    """Convert a byte count to a human-readable string (B / KB / MB / GB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a named logger with a consistent format.

    Args:
        name:  Logger name — usually __name__ of the calling module.
        level: Logging level (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Miscellaneous
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header to stdout."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
