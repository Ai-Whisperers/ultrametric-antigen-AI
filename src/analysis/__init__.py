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
    - classifiers: P-adic based classifiers (consolidated from classifiers/)
    - evolution: Viral evolution prediction (consolidated from evolution/)
    - codon_optimization: Codon optimization (consolidated from optimization/)
    - mrna_stability: mRNA stability prediction (consolidated from stability/)
    - immune_validation: Immune validation (consolidated from validation/)
"""

from .extremophile_codons import ExtremophileCategory, ExtremophileCodonAnalyzer
from .extraterrestrial_aminoacids import (
    AminoAcidSource,
    AsteroidAminoAcidAnalyzer,
    CompatibilityResult,
    ExtraterrestrialSample,
)
from .geometry import compute_delta_hyperbolicity, compute_pairwise_distances

# P-adic classifiers (consolidated from classifiers/)
from .classifiers import (
    ClassificationResult,
    CodonClassifier,
    GoldilocksZoneClassifier,
    PAdicClassifierBase,
    PAdicHierarchicalClassifier,
    PAdicKNN,
)

# Viral evolution prediction (consolidated from evolution/)
from .evolution import (
    AMINO_ACID_PROPERTIES,
    EscapeMutation,
    EscapePrediction,
    EvolutionaryPressure,
    EvolutionaryTrajectoryPredictor,
    MutationHotspot,
    RadiusMapping,
    SelectionType,
    TransmissibilityProfile,
    TransmissibilityRadiusMapper,
    ViralEvolutionPredictor,
)

# Codon optimization (consolidated from optimization/)
from .codon_optimization import (
    CitrullinationBoundaryOptimizer,
    CodonChoice,
    CodonContextOptimizer,
    OptimizationResult,
    PAdicBoundaryAnalyzer,
)

# mRNA stability prediction (consolidated from stability/)
from .mrna_stability import (
    CODON_STABILITY_SCORES,
    MFEEstimator,
    mRNAStabilityPredictor,
    SecondaryStructurePredictor,
    StabilityPrediction,
    UTROptimizer,
)

# Immune validation (consolidated from validation/)
from .immune_validation import (
    GoldilocksZoneValidator,
    ImmuneResponse,
    ImmuneThresholdData,
    MHCClass,
    NobelImmuneValidator,
    ValidationResult,
)

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

# Ancestral sequence reconstruction
from .ancestry import (
    AncestralNode,
    AncestralState,
    GeodesicInterpolator,
    PhylogeneticReconstructor,
    ReconstructionConfig,
    TreeNode,
)

__all__ = [
    # Geometry
    "compute_pairwise_distances",
    "compute_delta_hyperbolicity",
    # Extremophile codons
    "ExtremophileCodonAnalyzer",
    "ExtremophileCategory",
    # Extraterrestrial amino acids
    "AsteroidAminoAcidAnalyzer",
    "ExtraterrestrialSample",
    "CompatibilityResult",
    "AminoAcidSource",
    # P-adic classifiers (consolidated from classifiers/)
    "ClassificationResult",
    "CodonClassifier",
    "GoldilocksZoneClassifier",
    "PAdicClassifierBase",
    "PAdicHierarchicalClassifier",
    "PAdicKNN",
    # Viral evolution (consolidated from evolution/)
    "AMINO_ACID_PROPERTIES",
    "EscapeMutation",
    "EscapePrediction",
    "EvolutionaryPressure",
    "EvolutionaryTrajectoryPredictor",
    "MutationHotspot",
    "RadiusMapping",
    "SelectionType",
    "TransmissibilityProfile",
    "TransmissibilityRadiusMapper",
    "ViralEvolutionPredictor",
    # Codon optimization (consolidated from optimization/)
    "CitrullinationBoundaryOptimizer",
    "CodonChoice",
    "CodonContextOptimizer",
    "OptimizationResult",
    "PAdicBoundaryAnalyzer",
    # mRNA stability (consolidated from stability/)
    "CODON_STABILITY_SCORES",
    "MFEEstimator",
    "mRNAStabilityPredictor",
    "SecondaryStructurePredictor",
    "StabilityPrediction",
    "UTROptimizer",
    # Immune validation (consolidated from validation/)
    "GoldilocksZoneValidator",
    "ImmuneResponse",
    "ImmuneThresholdData",
    "MHCClass",
    "NobelImmuneValidator",
    "ValidationResult",
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
    # Ancestral sequence reconstruction
    "AncestralNode",
    "AncestralState",
    "GeodesicInterpolator",
    "PhylogeneticReconstructor",
    "ReconstructionConfig",
    "TreeNode",
]
