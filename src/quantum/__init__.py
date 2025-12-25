# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Quantum biology analysis modules.

This package provides tools for analyzing quantum-level phenomena in
biological systems using p-adic mathematics.

Modules:
    - descriptors: Quantum-chemical descriptor computation
    - biology: P-adic analysis of quantum-active biological sites
"""

from .biology import QuantumBiologyAnalyzer, QuantumEnzyme
from .descriptors import (
    AminoAcidQuantumProperties,
    QuantumBioDescriptor,
    QuantumDescriptorResult,
)

__all__ = [
    "QuantumBioDescriptor",
    "QuantumDescriptorResult",
    "AminoAcidQuantumProperties",
    "QuantumBiologyAnalyzer",
    "QuantumEnzyme",
]
