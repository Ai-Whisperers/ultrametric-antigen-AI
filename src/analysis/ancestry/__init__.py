# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Ancestral Sequence Reconstruction Module.

Implements ancestral state reconstruction using hyperbolic geometry:
- GeodesicInterpolator: Interpolate along geodesics for ancestral reconstruction
- PhylogeneticReconstructor: Full phylogenetic tree reconstruction
- AncestralDecoder: Decode ancestral latent representations to sequences

Key Insight:
In hyperbolic space, evolutionary trees embed naturally with low distortion.
Geodesic interpolation between descendant sequences passes through the
ancestral state, enabling principled reconstruction.

Mathematical Background:
For points x, y on the Poincare ball, the geodesic is:
    gamma(t) = x (+) (t (*) ((-x) (+) y))

where (+) is Mobius addition and (*) is scalar Mobius multiplication.

At t=0.5, we get the geodesic midpoint (approximate ancestor of x and y).

References:
- Nickel & Kiela (2017): Poincare Embeddings
- Mardia & Jupp (2000): Directional Statistics (Riemannian averaging)
- Yang et al. (1995): Maximum Likelihood Ancestral Reconstruction
"""

from src.analysis.ancestry.geodesic_interpolator import (
    AncestralNode,
    AncestralState,
    GeodesicInterpolator,
    PhylogeneticReconstructor,
    ReconstructionConfig,
    TreeNode,
)

__all__ = [
    "GeodesicInterpolator",
    "PhylogeneticReconstructor",
    "AncestralNode",
    "AncestralState",
    "TreeNode",
    "ReconstructionConfig",
]
