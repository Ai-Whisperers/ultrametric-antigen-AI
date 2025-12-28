# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified experiment framework for reproducible cross-disease evaluation.

This module provides standardized experiment infrastructure for:
- Cross-validation with fixed seeds
- Metric computation (Spearman, RMSE, AUC, etc.)
- Result logging to JSON
- Publication-quality figure generation

Example:
    >>> from src.experiments import CrossDiseaseExperiment
    >>> exp = CrossDiseaseExperiment(
    ...     diseases=["hiv", "sars_cov_2", "tuberculosis"],
    ...     model_class=TernaryVAE,
    ... )
    >>> results = exp.run()
"""

from .base_experiment import (
    BaseExperiment,
    ExperimentConfig,
    ExperimentResult,
    MetricComputer,
)
from .disease_experiment import (
    DiseaseExperiment,
    CrossDiseaseExperiment,
)

__all__ = [
    "BaseExperiment",
    "ExperimentConfig",
    "ExperimentResult",
    "MetricComputer",
    "DiseaseExperiment",
    "CrossDiseaseExperiment",
]
