# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Evaluation metrics and benchmarking tools.

This module provides standardized evaluation metrics inspired by:
- ProteinGym: Benchmark for protein fitness prediction
- ProteinBench: Comprehensive protein design evaluation
- RFdiffusion: Structure-based design metrics

Metric Categories:
    **Quality Metrics**:
        - Reconstruction accuracy
        - Predicted structure quality (pLDDT, TM-score)
        - Biological validity

    **Novelty Metrics**:
        - Sequence identity to training set
        - Structural novelty

    **Diversity Metrics**:
        - Cluster entropy
        - Pairwise diversity

    **Biological Metrics**:
        - Codon optimality (tAI, CAI)
        - Functional conservation
        - Resistance mutation coverage
"""

from .protein_metrics import (
    BiologicalValidityMetrics,
    DiversityMetrics,
    GenerationMetrics,
    NoveltyMetrics,
    ProteinGymEvaluator,
    QualityMetrics,
    evaluate_generated_sequences,
)

__all__ = [
    "GenerationMetrics",
    "QualityMetrics",
    "NoveltyMetrics",
    "DiversityMetrics",
    "BiologicalValidityMetrics",
    "ProteinGymEvaluator",
    "evaluate_generated_sequences",
]
