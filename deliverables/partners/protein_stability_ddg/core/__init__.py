# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Core modules for Jose Colbes package - self-contained p-adic math."""

from .padic_math import padic_valuation, padic_norm, padic_distance
from .constants import (
    AMINO_ACIDS,
    HYDROPHOBICITY,
    CHARGES,
    VOLUMES,
    FLEXIBILITY,
    MOLECULAR_WEIGHTS,
    HELIX_PROPENSITY,
    SHEET_PROPENSITY,
)

__all__ = [
    "padic_valuation",
    "padic_norm",
    "padic_distance",
    "AMINO_ACIDS",
    "HYDROPHOBICITY",
    "CHARGES",
    "VOLUMES",
    "FLEXIBILITY",
    "MOLECULAR_WEIGHTS",
    "HELIX_PROPENSITY",
    "SHEET_PROPENSITY",
]
