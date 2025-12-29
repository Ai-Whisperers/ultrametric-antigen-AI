# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Equivariant Neural Networks module.

Provides neural network layers that respect various symmetry groups:

**Rotational Symmetries (SO(3)):**
- `SO3Layer`: SO(3)-equivariant message passing using spherical harmonics
- `SO3Linear`: SO(3)-equivariant linear transformation
- `SO3Convolution`: SO(3)-equivariant graph convolution
- `SO3GNN`: Full SO(3)-equivariant graph neural network

**Rigid Transformation Symmetries (SE(3)):**
- `SE3Layer`: SE(3)-equivariant layer for 3D point clouds
- `SE3Transformer`: SE(3)-equivariant transformer architecture
- `EGNN`: E(n) Equivariant Graph Neural Network
- `EGNNLayer`: Single EGNN layer

**Biological Symmetries (Codons):**
- `CodonSymmetryLayer`: Layer respecting codon synonymy and wobble position
- `CodonEmbedding`: Embedding with synonymous codon awareness
- `CodonAttention`: Attention mechanism with synonymy bias
- `CodonTransformer`: Full transformer for codon sequences

**Utilities:**
- `SphericalHarmonics`: Spherical harmonic computation
- `RadialBasisFunctions`: Distance-based features
- `ClebschGordanCoefficients`: Angular momentum coupling

Example:
    >>> from src.equivariant import SO3Layer, SE3Layer, CodonSymmetryLayer
    >>>
    >>> # SO(3)-equivariant GNN
    >>> so3_layer = SO3Layer(in_features=64, out_features=64, lmax=2)
    >>>
    >>> # SE(3)-equivariant layer
    >>> se3_layer = SE3Layer(hidden_dim=128)
    >>>
    >>> # Codon symmetry layer
    >>> codon_layer = CodonSymmetryLayer(hidden_dim=64)
"""

from .codon_symmetry import (
    AMINO_ACIDS,
    GENETIC_CODE,
    CodonAttention,
    CodonEmbedding,
    CodonSymmetryLayer,
    CodonTransformer,
    CodonTransformerLayer,
    SynonymousPooling,
    WobbleAwareConv,
    codon_to_index,
    get_synonymous_groups,
    get_wobble_equivalences,
    index_to_codon,
)
from .se3_layer import (
    EGNN,
    EGNNLayer,
    SE3Layer,
    SE3Linear,
    SE3MessagePassing,
    SE3Transformer,
)
from .so3_layer import (
    SO3GNN,
    RadialBasisFunctions,
    SmoothCutoff,
    SO3Convolution,
    SO3Layer,
    SO3Linear,
)
from .spherical_harmonics import (
    ClebschGordanCoefficients,
    SphericalHarmonics,
    associated_legendre,
    spherical_harmonics_manual,
    wigner_d_matrix,
)

__all__ = [
    # Spherical harmonics
    "SphericalHarmonics",
    "ClebschGordanCoefficients",
    "associated_legendre",
    "spherical_harmonics_manual",
    "wigner_d_matrix",
    # SO(3) layers
    "RadialBasisFunctions",
    "SmoothCutoff",
    "SO3Linear",
    "SO3Convolution",
    "SO3Layer",
    "SO3GNN",
    # SE(3) layers
    "SE3Linear",
    "SE3MessagePassing",
    "SE3Layer",
    "SE3Transformer",
    "EGNN",
    "EGNNLayer",
    # Codon symmetry
    "GENETIC_CODE",
    "AMINO_ACIDS",
    "codon_to_index",
    "index_to_codon",
    "get_synonymous_groups",
    "get_wobble_equivalences",
    "CodonEmbedding",
    "SynonymousPooling",
    "WobbleAwareConv",
    "CodonSymmetryLayer",
    "CodonAttention",
    "CodonTransformer",
    "CodonTransformerLayer",
]
