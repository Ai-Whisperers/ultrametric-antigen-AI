# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Artifact management components.

This module handles checkpoint and artifact lifecycle management:
- CheckpointManager: Save/load checkpoints with metadata
- ArtifactRepository: Artifact promotion (raw → validated → production)
- Metadata: Checkpoint and model metadata handling
"""

from .checkpoint_manager import CheckpointManager

__all__ = [
    'CheckpointManager'
]
