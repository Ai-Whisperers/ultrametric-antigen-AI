# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Disease-specific analysis modules.

This package provides specialized analyzers for understanding diseases
through the p-adic and geometric lens of the Ternary VAE framework.

Modules:
    - repeat_expansion: Trinucleotide repeat expansion diseases (HD, SCA, etc.)
    - long_covid: SARS-CoV-2 spike protein and Long COVID analysis
    - multiple_sclerosis: MS molecular mimicry and demyelination analysis
    - rheumatoid_arthritis: RA citrullination and Goldilocks Zone analysis

Usage:
    from src.diseases import RepeatExpansionAnalyzer, LongCOVIDAnalyzer

    # Huntington's disease analysis
    analyzer = RepeatExpansionAnalyzer()
    risk = analyzer.analyze_repeat_padic_distance("huntington", repeat_count=42)
"""

from .long_covid import LongCOVIDAnalyzer, SpikeVariantComparator
from .multiple_sclerosis import (
    DemyelinationPrediction,
    EpitopePair,
    HLABindingPredictor,
    MolecularMimicryDetector,
    MSRiskProfile,
    MSSubtype,
    MultipleSclerosisAnalyzer,
    MyelinTarget,
)
from .repeat_expansion import (RepeatDiseaseInfo, RepeatExpansionAnalyzer,
                               TrinucleotideRepeat)

# Rheumatoid Arthritis analysis
from .rheumatoid_arthritis import (
    CitrullinationPredictor,
    CitrullinationSite,
    EpitopeAnalysis,
    GoldilocksZoneDetector,
    PADEnzyme,
    PAdicCitrullinationShift,
    RARiskProfile,
    RASubtype,
    RheumatoidArthritisAnalyzer,
)

__all__ = [
    "RepeatExpansionAnalyzer",
    "RepeatDiseaseInfo",
    "TrinucleotideRepeat",
    "LongCOVIDAnalyzer",
    "SpikeVariantComparator",
    # Multiple Sclerosis
    "MultipleSclerosisAnalyzer",
    "MolecularMimicryDetector",
    "HLABindingPredictor",
    "MSSubtype",
    "MyelinTarget",
    "EpitopePair",
    "MSRiskProfile",
    "DemyelinationPrediction",
    # Rheumatoid Arthritis
    "RheumatoidArthritisAnalyzer",
    "CitrullinationPredictor",
    "PAdicCitrullinationShift",
    "GoldilocksZoneDetector",
    "CitrullinationSite",
    "RARiskProfile",
    "EpitopeAnalysis",
    "PADEnzyme",
    "RASubtype",
]
