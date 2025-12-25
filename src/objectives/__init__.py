# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Multi-objective optimization functions for vaccine and therapeutic design.

This module provides objective functions that can be combined for Pareto-optimal
design of biological sequences. Each objective returns a scalar score where
LOWER is BETTER (for consistency with optimization algorithms).

Objectives:
    - GeometricObjective: Alignment with target geometric structure
    - BindingObjective: Predicted binding affinity to target
    - SolubilityObjective: Protein expression and solubility
    - StabilityObjective: Thermodynamic stability
    - ManufacturabilityObjective: Production feasibility score
    - ImmunogenicityObjective: Immune response prediction

Usage:
    from src.objectives import ObjectiveRegistry, SolubilityObjective

    registry = ObjectiveRegistry()
    registry.register("solubility", SolubilityObjective())
    registry.register("stability", StabilityObjective())

    scores = registry.evaluate(latent_vectors, decoded_sequences)
"""

from .base import Objective, ObjectiveRegistry, ObjectiveResult
from .binding import BindingObjective
from .manufacturability import ManufacturabilityObjective, ProductionCostObjective
from .solubility import SolubilityObjective, StabilityObjective

__all__ = [
    "Objective",
    "ObjectiveResult",
    "ObjectiveRegistry",
    "BindingObjective",
    "SolubilityObjective",
    "StabilityObjective",
    "ManufacturabilityObjective",
    "ProductionCostObjective",
]
