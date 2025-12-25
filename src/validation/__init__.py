# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Validation modules for testing hypotheses.

This package provides tools for validating the Goldilocks Zone hypothesis
and other theoretical predictions against experimental data.

Modules:
    - nobel_immune: Validation using immune recognition thresholds
"""

from .nobel_immune import (
    GoldilocksZoneValidator,
    ImmuneThresholdData,
    NobelImmuneValidator,
    ValidationResult,
)

__all__ = [
    "NobelImmuneValidator",
    "GoldilocksZoneValidator",
    "ImmuneThresholdData",
    "ValidationResult",
]
