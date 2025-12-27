# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tropical Geometry module.

Provides tropical algebraic tools for neural network analysis
and phylogenetic tree distances.

Key Components:
- TropicalPolynomial: Tropical polynomial representation
- TropicalNNAnalyzer: Analyze ReLU networks as tropical computations
- TropicalPhylogeneticDistance: Tree distances in tropical space

Example:
    from src.tropical import TropicalNNAnalyzer

    analyzer = TropicalNNAnalyzer(relu_network)
    n_regions = analyzer.compute_linear_regions()
"""

__all__ = [
    "TropicalPolynomial",
    "TropicalNNAnalyzer",
    "TropicalPhylogeneticDistance",
]
