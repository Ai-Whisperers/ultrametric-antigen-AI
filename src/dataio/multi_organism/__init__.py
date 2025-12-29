"""Multi-organism data loading framework.

This module provides unified data loading for multiple organisms:
- Viruses: HIV, HBV, HCV, Influenza, SARS-CoV-2
- Bacteria: TB, MRSA, E. coli
- Parasites: Plasmodium (Malaria)
- Proteins: Antibodies, TCRs, Kinases, GPCRs

All organisms are encoded using p-adic embeddings that capture
hierarchical evolutionary structure.

Usage:
    from src.data.multi_organism import OrganismLoader, OrganismType

    # Load HIV sequences
    loader = OrganismLoader(OrganismType.HIV)
    sequences, labels = loader.load_sequences()

    # Load with p-adic encoding
    encoded = loader.load_encoded(prime=3)
"""

from .base import OrganismLoader, OrganismType, SequenceRecord
from .registry import OrganismRegistry

__all__ = [
    "OrganismLoader",
    "OrganismType",
    "SequenceRecord",
    "OrganismRegistry",
]
