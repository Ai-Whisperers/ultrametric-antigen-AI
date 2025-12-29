"""
P-adic arithmetic module.

Implements p-adic numbers and arithmetic for the encoding framework.
P-adic numbers naturally capture hierarchical genetic relationships.
"""
from .number import PadicNumber
from .arithmetic import padic_add, padic_multiply, padic_subtract
from .distance import padic_distance, padic_norm

__all__ = [
    "PadicNumber",
    "padic_add",
    "padic_multiply",
    "padic_subtract",
    "padic_distance",
    "padic_norm",
]
