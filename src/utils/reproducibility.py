"""Reproducibility utilities for training.

This module provides functions for ensuring reproducible training runs.

Single responsibility: Random seed management only.
"""

import torch
import numpy as np
import random


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: Random seed value
        deterministic: If True, enables fully deterministic mode
                      (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_generator(seed: int) -> torch.Generator:
    """Create a seeded PyTorch Generator.

    Useful for reproducible data splitting and shuffling.

    Args:
        seed: Random seed value

    Returns:
        Seeded torch.Generator
    """
    return torch.Generator().manual_seed(seed)


__all__ = ['set_seed', 'get_generator']
