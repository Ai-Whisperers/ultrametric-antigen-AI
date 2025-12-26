"""
Core domain layer - the heart of the p-adic hyperbolic encoding system.

This layer has no external dependencies and defines:
- Entities: Codon, Sequence, Mutation, Epitope, Embedding
- Value Objects: Position, HLAAllele, Drug, Protein
- Interfaces: IEncoder, IPredictor, IRepository, IAnalyzer
- Exceptions: Domain-specific exceptions
"""
from .entities import Codon, Sequence, Mutation, Epitope, Embedding
from .interfaces import IEncoder, IPredictor, IRepository, IAnalyzer
from .exceptions import DomainError, ValidationError, EncodingError

__all__ = [
    # Entities
    "Codon",
    "Sequence",
    "Mutation",
    "Epitope",
    "Embedding",
    # Interfaces
    "IEncoder",
    "IPredictor",
    "IRepository",
    "IAnalyzer",
    # Exceptions
    "DomainError",
    "ValidationError",
    "EncodingError",
]
