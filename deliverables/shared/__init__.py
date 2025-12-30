# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Shared infrastructure for all deliverables.

This module provides common utilities used by all partner implementations:
- VAE service for latent space encoding/decoding
- Configuration management
- Common utilities and constants
"""

from __future__ import annotations

from .config import Config, get_config
from .vae_service import VAEService, get_vae_service
from .constants import AMINO_ACIDS, CODON_TABLE, HYDROPHOBICITY, CHARGES, VOLUMES
from .peptide_utils import (
    AA_PROPERTIES,
    compute_peptide_properties,
    compute_ml_features,
    compute_amino_acid_composition,
    decode_latent_to_sequence,
    decode_latent_with_vae,
    compute_physicochemical_descriptors,
    validate_sequence,
)
from .uncertainty import (
    bootstrap_prediction_interval,
    ensemble_prediction_interval,
    quantile_prediction_interval,
    calibrate_prediction_interval,
    UncertaintyPredictor,
    compute_prediction_metrics_with_uncertainty,
)
from .hemolysis_predictor import (
    HemolysisPredictor,
    get_hemolysis_predictor,
)
from .logging_utils import (
    get_logger,
    setup_logging,
    LogContext,
    log_function_call,
)
from .primer_design import (
    PrimerDesigner,
    PrimerResult,
    get_primer_designer,
    calculate_tm,
    calculate_gc,
    reverse_complement,
)

__all__ = [
    # Config
    "Config",
    "get_config",
    # VAE Service
    "VAEService",
    "get_vae_service",
    # Constants
    "AMINO_ACIDS",
    "CODON_TABLE",
    "HYDROPHOBICITY",
    "CHARGES",
    "VOLUMES",
    # Peptide utilities
    "AA_PROPERTIES",
    "compute_peptide_properties",
    "compute_ml_features",
    "compute_amino_acid_composition",
    "decode_latent_to_sequence",
    "decode_latent_with_vae",
    "compute_physicochemical_descriptors",
    "validate_sequence",
    # Uncertainty quantification
    "bootstrap_prediction_interval",
    "ensemble_prediction_interval",
    "quantile_prediction_interval",
    "calibrate_prediction_interval",
    "UncertaintyPredictor",
    "compute_prediction_metrics_with_uncertainty",
    # Hemolysis prediction
    "HemolysisPredictor",
    "get_hemolysis_predictor",
    # Logging
    "get_logger",
    "setup_logging",
    "LogContext",
    "log_function_call",
    # Primer design
    "PrimerDesigner",
    "PrimerResult",
    "get_primer_designer",
    "calculate_tm",
    "calculate_gc",
    "reverse_complement",
]
