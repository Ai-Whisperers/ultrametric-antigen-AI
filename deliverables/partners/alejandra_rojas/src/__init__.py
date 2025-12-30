# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Arbovirus Surveillance Package - Core library.

This module provides tools for arbovirus surveillance including:
- NCBI sequence downloading and caching
- Pan-arbovirus RT-PCR primer design
- Cross-reactivity analysis
- Serotype-specific targeting

Example:
    >>> from arbovirus_surveillance.src import NCBIClient, PrimerDesigner
    >>> client = NCBIClient()
    >>> db = client.load_or_download()
    >>> designer = PrimerDesigner(db)
    >>> primers = designer.design_primers("DENV-1")
"""

from __future__ import annotations

from .ncbi_client import (
    NCBIClient,
    VirusSequence,
    ArbovirusDatabase,
    RateLimiter,
)
from .primer_designer import (
    PrimerDesigner,
    PrimerPair,
    PrimerCandidate,
    CrossReactivityResult,
)
from .constants import (
    ARBOVIRUS_TARGETS,
    PRIMER_CONSTRAINTS,
    CONSERVED_REGIONS,
)

__all__ = [
    # NCBI Client
    "NCBIClient",
    "VirusSequence",
    "ArbovirusDatabase",
    "RateLimiter",
    # Primer Design
    "PrimerDesigner",
    "PrimerPair",
    "PrimerCandidate",
    "CrossReactivityResult",
    # Constants
    "ARBOVIRUS_TARGETS",
    "PRIMER_CONSTRAINTS",
    "CONSERVED_REGIONS",
]

__version__ = "1.0.0"
