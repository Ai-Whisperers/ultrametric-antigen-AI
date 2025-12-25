# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Utility functions for Ternary VAE.

This module contains:
- Coverage tracking utilities
- Reproducibility utilities (seed management)
- Precomputed ternary LUTs (P1 optimization)

For data generation, use src.data instead.
For hyperbolic metrics, use src.metrics instead.
"""

from .metrics import (CoverageTracker, compute_diversity_score,
                      compute_latent_entropy, evaluate_coverage)
from .reproducibility import get_generator, set_seed
from .ternary_lut import (TERNARY_LUT, VALUATION_LUT, get_3adic_distance,
                          get_3adic_distance_batch, get_ternary_batch,
                          get_valuation_batch)

# P-adic shift operations
from .padic_shift import (
    PAdicCodonAnalyzer,
    PAdicSequenceEncoder,
    PAdicShiftResult,
    batch_padic_distance,
    codon_padic_distance,
    codon_to_index,
    index_to_codon,
    padic_digits,
    padic_distance,
    padic_distance_matrix,
    padic_norm,
    padic_shift,
    padic_valuation,
    sequence_padic_encoding,
)

# Checkpoint utilities
from .checkpoint import (
    NumpyBackwardsCompatUnpickler,
    extract_model_state,
    load_checkpoint_compat,
    save_checkpoint,
)

__all__ = [
    # Coverage metrics
    "evaluate_coverage",
    "compute_latent_entropy",
    "compute_diversity_score",
    "CoverageTracker",
    # Reproducibility
    "set_seed",
    "get_generator",
    # Ternary LUTs (P1 optimization)
    "VALUATION_LUT",
    "TERNARY_LUT",
    "get_valuation_batch",
    "get_ternary_batch",
    "get_3adic_distance",
    "get_3adic_distance_batch",
    # P-adic shift operations
    "padic_shift",
    "padic_valuation",
    "padic_norm",
    "padic_distance",
    "padic_digits",
    "padic_distance_matrix",
    "batch_padic_distance",
    "codon_to_index",
    "index_to_codon",
    "codon_padic_distance",
    "sequence_padic_encoding",
    "PAdicShiftResult",
    "PAdicSequenceEncoder",
    "PAdicCodonAnalyzer",
    # Checkpoint utilities
    "NumpyBackwardsCompatUnpickler",
    "load_checkpoint_compat",
    "save_checkpoint",
    "extract_model_state",
]
