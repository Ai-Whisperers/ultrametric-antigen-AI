# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Contrastive Learning module.

Provides self-supervised contrastive learning methods
with p-adic structure awareness.

Key Components:
- PAdicContrastiveLoss: Use p-adic distance for positive sampling
- MultiScaleContrastive: Hierarchical contrastive learning
- SimCLREncoder: Projection head for contrastive learning

Example:
    from src.contrastive import PAdicContrastiveLoss

    loss_fn = PAdicContrastiveLoss(temperature=0.1)
    loss = loss_fn(embeddings, indices)
"""

__all__ = [
    "PAdicContrastiveLoss",
    "MultiScaleContrastive",
    "SimCLREncoder",
]
