# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Checkpoint loading utilities with numpy version compatibility.

This module provides backward-compatible checkpoint loading that handles
numpy version changes (numpy._core -> numpy.core).

Single responsibility: Checkpoint I/O with version compatibility.
"""

import pickle
from pathlib import Path
from typing import Any

import torch


class NumpyBackwardsCompatUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy._core -> numpy.core renaming.

    PyTorch checkpoints saved with newer numpy versions use numpy._core
    which older versions cannot load. This unpickler handles the renaming.
    """

    def find_class(self, module: str, name: str) -> Any:
        """Override find_class to handle numpy module renaming.

        Args:
            module: Module name from pickle
            name: Class name from pickle

        Returns:
            The resolved class
        """
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_checkpoint_compat(
    path: Path | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint with numpy version compatibility.

    Attempts standard torch.load first, falls back to custom unpickler
    if numpy._core ModuleNotFoundError is encountered.

    Args:
        path: Path to checkpoint file
        map_location: Device to map tensors to

    Returns:
        Loaded checkpoint dictionary

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ModuleNotFoundError: If a non-numpy module is missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            with open(path, "rb") as f:
                unpickler = NumpyBackwardsCompatUnpickler(f)
                return unpickler.load()
        raise


def save_checkpoint(
    checkpoint: dict[str, Any],
    path: Path | str,
) -> None:
    """Save checkpoint to file.

    Args:
        checkpoint: Checkpoint dictionary to save
        path: Destination path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def get_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract model state dict from checkpoint, handling various key formats.

    Different checkpoint versions use different key names:
    - model_state_dict: New standard format (homeostatic_rich)
    - model_state: v5.11 format (v5_11_homeostasis, v5_11_structural)
    - model: v5.5 format

    Args:
        checkpoint: Full checkpoint dictionary

    Returns:
        Model state dict

    Raises:
        ValueError: If no model state can be found
    """
    # Try different key formats in order of preference
    for key in ["model_state_dict", "model_state", "model"]:
        if key in checkpoint:
            return checkpoint[key]

    # If no known key found, check if checkpoint looks like a state dict
    if isinstance(checkpoint, dict) and any(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        return checkpoint

    raise ValueError(
        f"Cannot find model state in checkpoint. Keys: {list(checkpoint.keys())}"
    )


def extract_model_state(
    checkpoint: dict[str, Any],
    prefix: str,
    strip_prefix: bool = True,
) -> dict[str, torch.Tensor]:
    """Extract model state dict with optional prefix filtering.

    Args:
        checkpoint: Full checkpoint dictionary
        prefix: Key prefix to filter (e.g., 'encoder_A.')
        strip_prefix: Whether to remove prefix from keys

    Returns:
        Filtered state dict
    """
    model_state = get_model_state_dict(checkpoint)
    prefix_dot = f"{prefix}." if not prefix.endswith(".") else prefix

    filtered = {}
    for key, value in model_state.items():
        if key.startswith(prefix_dot):
            new_key = key[len(prefix_dot):] if strip_prefix else key
            filtered[new_key] = value

    return filtered


__all__ = [
    "NumpyBackwardsCompatUnpickler",
    "load_checkpoint_compat",
    "save_checkpoint",
    "get_model_state_dict",
    "extract_model_state",
]
