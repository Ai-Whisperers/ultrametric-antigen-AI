# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""High-level optimization workflows for sequence and structure design.

This package provides optimization workflows for designing biological
sequences with specific properties, particularly focused on avoiding
autoimmune triggers through p-adic analysis.

Note:
    This module contains application-specific optimization WORKFLOWS.
    For low-level optimizer implementations (Riemannian, NSGA-II, etc.),
    see src/optimizers/ instead.

    - src/optimization/ (this) = What to optimize (sequence design workflows)
    - src/optimizers/ = How to optimize (optimizer algorithms)

Modules:
    - citrullination_optimizer: Codon optimization for citrullination safety
"""

from .citrullination_optimizer import (
    CitrullinationBoundaryOptimizer,
    CodonChoice,
    CodonContextOptimizer,
    OptimizationResult,
    PAdicBoundaryAnalyzer,
)

__all__ = [
    "CitrullinationBoundaryOptimizer",
    "PAdicBoundaryAnalyzer",
    "CodonContextOptimizer",
    "OptimizationResult",
    "CodonChoice",
]
