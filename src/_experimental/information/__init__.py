# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Information Geometry module.

Provides tools for analyzing neural networks through the lens of
differential geometry on statistical manifolds.

Key Components:
- FisherInfo: Container for Fisher information matrix and properties
- FisherInformationEstimator: Estimate Fisher information matrix
- NaturalGradientOptimizer: Optimize using natural gradients
- KFACOptimizer: K-FAC second-order optimizer
- InformationGeometricAnalyzer: Analyze training dynamics

Example:
    from src.information import NaturalGradientOptimizer

    optimizer = NaturalGradientOptimizer(model.parameters(), lr=0.01)
"""

from src._experimental.information.fisher_geometry import (
    FisherInfo,
    FisherInformationEstimator,
    InformationGeometricAnalyzer,
    KFACOptimizer,
    NaturalGradientOptimizer,
)

__all__ = [
    "FisherInfo",
    "FisherInformationEstimator",
    "NaturalGradientOptimizer",
    "KFACOptimizer",
    "InformationGeometricAnalyzer",
]
