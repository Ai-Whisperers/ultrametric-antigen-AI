# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Statistical Physics module.

Provides physics-inspired methods for analyzing biological systems,
including spin glass models and replica exchange methods.

Key Components:
- SpinGlassLandscape: Model protein folding as spin glass
- ReplicaExchange: Parallel tempering for sampling
- UltrametricTreeExtractor: Extract ultrametric structure

Example:
    from src.physics import SpinGlassLandscape, ReplicaExchange

    landscape = SpinGlassLandscape(n_residues=100)
    sampler = ReplicaExchange(n_replicas=8)
    samples = sampler.sample(landscape)
"""

__all__ = [
    "SpinGlassLandscape",
    "ReplicaExchange",
    "UltrametricTreeExtractor",
]
