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
- MomentumContrastEncoder: MoCo-style momentum contrast

Example:
    from src.contrastive import PAdicContrastiveLoss

    loss_fn = PAdicContrastiveLoss(temperature=0.1)
    loss = loss_fn(embeddings, indices)
"""

from src.contrastive.codon_sampler import (
    CodonContrastiveDataset,
    CodonPositiveSampler,
    CodonSamplerConfig,
    build_synonymous_codon_groups,
    build_wobble_variants,
)
from src.contrastive.padic_contrastive import (
    ContrastiveConfig,
    ContrastiveDataAugmentation,
    MomentumContrastEncoder,
    MultiScaleContrastive,
    PAdicContrastiveLoss,
    PAdicPositiveSampler,
    SimCLREncoder,
)

__all__ = [
    # P-adic contrastive learning
    "PAdicContrastiveLoss",
    "MultiScaleContrastive",
    "SimCLREncoder",
    "MomentumContrastEncoder",
    "PAdicPositiveSampler",
    "ContrastiveDataAugmentation",
    "ContrastiveConfig",
    # Codon-aware sampling
    "CodonPositiveSampler",
    "CodonSamplerConfig",
    "CodonContrastiveDataset",
    "build_synonymous_codon_groups",
    "build_wobble_variants",
]
