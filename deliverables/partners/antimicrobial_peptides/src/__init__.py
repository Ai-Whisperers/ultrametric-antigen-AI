# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Carlos Brizuela - Peptide Foundry Package.

Self-contained package for antimicrobial peptide optimization.
"""

from __future__ import annotations

from .constants import (
    AMINO_ACIDS,
    HYDROPHOBICITY,
    CHARGES,
    VOLUMES,
    MOLECULAR_WEIGHTS,
    FLEXIBILITY,
    AA_PROPERTIES,
    WHO_CRITICAL_PATHOGENS,
    WHO_HIGH_PATHOGENS,
)
from .peptide_utils import (
    compute_peptide_properties,
    compute_amino_acid_composition,
    compute_ml_features,
    compute_physicochemical_descriptors,
    validate_sequence,
)
from .uncertainty import (
    UncertaintyPredictor,
    bootstrap_prediction_interval,
    ensemble_prediction_interval,
)

__all__ = [
    # Constants
    "AMINO_ACIDS",
    "HYDROPHOBICITY",
    "CHARGES",
    "VOLUMES",
    "MOLECULAR_WEIGHTS",
    "FLEXIBILITY",
    "AA_PROPERTIES",
    "WHO_CRITICAL_PATHOGENS",
    "WHO_HIGH_PATHOGENS",
    # Utilities
    "compute_peptide_properties",
    "compute_amino_acid_composition",
    "compute_ml_features",
    "compute_physicochemical_descriptors",
    "validate_sequence",
    # Uncertainty
    "UncertaintyPredictor",
    "bootstrap_prediction_interval",
    "ensemble_prediction_interval",
]

__version__ = "1.0.0"
