# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Analysis modules for geometric and biological data.

This package provides analysis tools for studying biological sequences
and structures through geometric and p-adic lenses.

Modules:
    - geometry: Geometric analysis of embeddings (hyperbolicity, distances)
    - extremophile_codons: Codon usage patterns in extremophile organisms
    - extraterrestrial_aminoacids: Asteroid/meteorite amino acid analysis
    - crispr_offtarget: CRISPR off-target landscape analysis using hyperbolic geometry
"""

from .extremophile_codons import ExtremophileCategory, ExtremophileCodonAnalyzer
from .extraterrestrial_aminoacids import (
    AminoAcidSource,
    AsteroidAminoAcidAnalyzer,
    CompatibilityResult,
    ExtraterrestrialSample,
)
from .geometry import compute_delta_hyperbolicity, compute_pairwise_distances

# Protein energy landscape analysis
from .protein_landscape import (
    ConformationState,
    EnergyBasin,
    FoldingFunnelAnalyzer,
    LandscapeMetrics,
    ProteinLandscapeAnalyzer,
    TransitionPath,
    TransitionStateAnalyzer,
    UltrametricDistanceMatrix,
)

# CRISPR off-target landscape analysis
from .crispr_offtarget import (
    CRISPROfftargetAnalyzer,
    GuideDesignOptimizer,
    GuideSafetyProfile,
    HyperbolicOfftargetEmbedder,
    MismatchType,
    OffTargetSite,
    OfftargetActivityPredictor,
    PAdicSequenceDistance,
)

__all__ = [
    "compute_pairwise_distances",
    "compute_delta_hyperbolicity",
    "ExtremophileCodonAnalyzer",
    "ExtremophileCategory",
    "AsteroidAminoAcidAnalyzer",
    "ExtraterrestrialSample",
    "CompatibilityResult",
    "AminoAcidSource",
    # Protein energy landscape
    "ProteinLandscapeAnalyzer",
    "FoldingFunnelAnalyzer",
    "TransitionStateAnalyzer",
    "UltrametricDistanceMatrix",
    "ConformationState",
    "EnergyBasin",
    "TransitionPath",
    "LandscapeMetrics",
    # CRISPR off-target landscape
    "CRISPROfftargetAnalyzer",
    "GuideDesignOptimizer",
    "GuideSafetyProfile",
    "HyperbolicOfftargetEmbedder",
    "MismatchType",
    "OffTargetSite",
    "OfftargetActivityPredictor",
    "PAdicSequenceDistance",
]
