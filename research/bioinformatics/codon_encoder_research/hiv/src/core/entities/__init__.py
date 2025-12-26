"""
Core domain entities.
"""
from .codon import Codon
from .sequence import Sequence, SequenceType
from .mutation import Mutation
from .epitope import Epitope
from .embedding import Embedding

__all__ = [
    "Codon",
    "Sequence",
    "SequenceType",
    "Mutation",
    "Epitope",
    "Embedding",
]
