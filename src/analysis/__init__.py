# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Analysis modules for geometric and biological data.

This package provides analysis tools for studying biological sequences
and structures through geometric and p-adic lenses.

Modules:
    - geometry: Geometric analysis of embeddings (hyperbolicity, distances)
    - extremophile_codons: Codon usage patterns in extremophile organisms
"""

from .extremophile_codons import ExtremophileCategory, ExtremophileCodonAnalyzer
from .geometry import compute_delta_hyperbolicity, compute_pairwise_distances

__all__ = [
    "compute_pairwise_distances",
    "compute_delta_hyperbolicity",
    "ExtremophileCodonAnalyzer",
    "ExtremophileCategory",
]
