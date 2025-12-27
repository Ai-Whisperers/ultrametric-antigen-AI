# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Equivariant Neural Networks module.

Provides layers that respect symmetry groups:
- SO(3): 3D rotations
- SE(3): 3D rigid transformations
- Codon symmetries: Wobble position and synonymous codons

Key Components:
- SO3Layer: SO(3)-equivariant message passing
- CodonSymmetryLayer: Codon-specific symmetries
- LorentzMLP: Lorentz model operations

Example:
    from src.equivariant import SO3Layer, CodonSymmetryLayer

    so3_layer = SO3Layer(in_features=64, out_features=64)
    codon_layer = CodonSymmetryLayer(hidden_dim=64)
"""

__all__ = [
    "SO3Layer",
    "CodonSymmetryLayer",
]
