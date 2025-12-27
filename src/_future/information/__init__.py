# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Information Geometry module.

Provides tools for analyzing neural networks through the lens of
differential geometry on statistical manifolds.

Key Components:
- FisherInformationEstimator: Estimate Fisher information matrix
- NaturalGradientOptimizer: Optimize using natural gradients
- InformationGeometricAnalyzer: Analyze training dynamics

Example:
    from src.information import NaturalGradientOptimizer

    optimizer = NaturalGradientOptimizer(model.parameters(), lr=0.01)
"""

__all__ = [
    "FisherInfo",
    "FisherInformationEstimator",
    "NaturalGradientOptimizer",
    "InformationGeometricAnalyzer",
]
