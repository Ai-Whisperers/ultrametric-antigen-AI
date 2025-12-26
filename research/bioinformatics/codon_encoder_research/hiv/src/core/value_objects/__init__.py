"""
Value objects - immutable domain primitives.
"""
from .position import Position, HXB2Position
from .hla_allele import HLAAllele
from .drug import Drug, DrugClass
from .protein import Protein, HIVProtein

__all__ = [
    "Position",
    "HXB2Position",
    "HLAAllele",
    "Drug",
    "DrugClass",
    "Protein",
    "HIVProtein",
]
