# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Graph Neural Networks module.

Provides hyperbolic and Euclidean graph neural networks
for biological network analysis.

Key Components:
- HyperbolicGraphConv: Message passing in Poincare ball
- LorentzMLP: Lorentz model operations
- HyboWaveNet: Multi-scale wavelet + hyperbolic GNN
- PoincareOperations: Mathematical operations in Poincare ball
- LorentzOperations: Mathematical operations in Lorentz model

Example:
    from src.graphs import HyperbolicGraphConv

    conv = HyperbolicGraphConv(in_channels=64, out_channels=64)
    out = conv(x, edge_index, edge_attr)
"""

from src.graphs.hyperbolic_gnn import (
    HyperbolicGraphConv,
    HyperbolicLinear,
    LorentzMLP,
    HyboWaveNet,
    PoincareOperations,
    LorentzOperations,
    SpectralWavelet,
)

__all__ = [
    "HyperbolicGraphConv",
    "HyperbolicLinear",
    "LorentzMLP",
    "HyboWaveNet",
    "PoincareOperations",
    "LorentzOperations",
    "SpectralWavelet",
]
