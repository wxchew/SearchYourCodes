"""
Device Detection

Centralized device detection for ML model inference.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_device: str | None = None


def get_device() -> str:
    """Get the best available compute device (cuda > mps > cpu)."""
    global _device
    if _device is None:
        try:
            import torch
            if torch.cuda.is_available():
                _device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                _device = "mps"
            else:
                _device = "cpu"
        except ImportError:
            _device = "cpu"
    return _device
