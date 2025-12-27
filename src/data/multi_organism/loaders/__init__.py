"""Organism-specific data loaders.

Each loader implements the OrganismLoader interface for a specific organism.
"""

# Import loaders to register them
from . import hbv_loader

__all__ = ["hbv_loader"]
