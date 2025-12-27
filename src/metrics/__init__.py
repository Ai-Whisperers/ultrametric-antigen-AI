# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DEPRECATED: Use src.core.metrics instead.

This module has been consolidated into src/core/ for better organization.
Imports are redirected for backward compatibility.
"""

import warnings

warnings.warn(
    "src.metrics is deprecated. Use src.core.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility redirect
from src.core.metrics import (
    compute_3adic_valuation,
    compute_ranking_correlation_hyperbolic,
    poincare_distance,
    project_to_poincare,
)

__all__ = [
    "project_to_poincare",
    "poincare_distance",
    "compute_3adic_valuation",
    "compute_ranking_correlation_hyperbolic",
]
