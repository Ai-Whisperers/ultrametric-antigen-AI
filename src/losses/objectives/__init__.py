# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Drug design objectives (binding, solubility, manufacturability)."""

from src.losses.objectives.base import (
    Objective,
    ObjectiveRegistry,
    ObjectiveResult,
)
from src.losses.objectives.binding import BindingObjective
from src.losses.objectives.manufacturability import ManufacturabilityObjective, ProductionCostObjective
from src.losses.objectives.solubility import SolubilityObjective, StabilityObjective

# Backward compatibility aliases
BaseObjective = Objective

__all__ = [
    "Objective",
    "BaseObjective",
    "ObjectiveRegistry",
    "ObjectiveResult",
    "BindingObjective",
    "ManufacturabilityObjective",
    "ProductionCostObjective",
    "SolubilityObjective",
    "StabilityObjective",
]
