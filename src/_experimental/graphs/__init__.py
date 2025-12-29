# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hyperbolic Graph Neural Networks module.

This module implements graph neural networks that operate in hyperbolic
space, which naturally captures hierarchical structure in biological
networks (protein interaction networks, phylogenetic trees, etc.).

Key Features:
    - Poincare ball model operations (Mobius addition, exp/log maps)
    - Lorentz (hyperboloid) model operations
    - Hyperbolic message passing for GNNs
    - Multi-scale spectral wavelet decomposition
    - HyboWaveNet: Combined wavelet + hyperbolic architecture

Mathematical Operations:
    PoincareOperations: Operations in the Poincare ball model B^n_c
    LorentzOperations: Operations on the hyperboloid H^n

Neural Network Layers:
    HyperbolicLinear: Linear layer operating in hyperbolic space
    HyperbolicGraphConv: Graph convolution in Poincare ball
    LorentzMLP: Multi-layer perceptron in Lorentz model

Complete Models:
    SpectralWavelet: Multi-scale graph wavelet decomposition
    HyboWaveNet: Full wavelet + hyperbolic GNN architecture

Example:
    >>> from src.graphs import HyperbolicGraphConv, HyboWaveNet
    >>>
    >>> # Single hyperbolic graph convolution layer
    >>> conv = HyperbolicGraphConv(
    ...     in_channels=64,
    ...     out_channels=64,
    ...     curvature=1.0,
    ...     use_attention=True
    ... )
    >>> node_features = conv(x, edge_index)
    >>>
    >>> # Full HyboWaveNet model
    >>> model = HyboWaveNet(
    ...     in_channels=64,
    ...     hidden_channels=128,
    ...     out_channels=32,
    ...     n_scales=4,
    ...     curvature=1.0
    ... )
    >>> embeddings = model(x, edge_index)
    >>> graph_embedding = model.encode_graph(x, edge_index)

References:
    - Chami et al. (2019): Hyperbolic Graph Convolutional Neural Networks
    - Liu et al. (2019): Hyperbolic Graph Neural Networks
    - Ganea et al. (2018): Hyperbolic Neural Networks
"""

from .hyperbolic_gnn import (
    HyboWaveNet,
    HyperbolicGraphConv,
    HyperbolicLinear,
    LorentzMLP,
    LorentzOperations,
    PoincareOperations,
    SpectralWavelet,
)

__all__ = [
    # Mathematical Operations
    "PoincareOperations",
    "LorentzOperations",
    # Neural Network Layers
    "HyperbolicLinear",
    "HyperbolicGraphConv",
    "LorentzMLP",
    # Complete Models
    "SpectralWavelet",
    "HyboWaveNet",
]

__version__ = "1.0.0"
