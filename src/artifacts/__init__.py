# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DEPRECATED: Use src.training.checkpoint_manager instead.

This module has been consolidated into src/training/ for better organization.
Imports are redirected for backward compatibility.
"""

import warnings

warnings.warn(
    "src.artifacts is deprecated. Use src.training.checkpoint_manager instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility redirect
from src.training.checkpoint_manager import AsyncCheckpointSaver, CheckpointManager

__all__ = ["CheckpointManager", "AsyncCheckpointSaver"]
