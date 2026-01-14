"""
Integration modules connecting data access to analysis pipelines.

Provides utilities for:
- HIV drug resistance analysis with p-adic hyperbolic encoding
- Sequence-to-codon extraction and mapping
- Cross-dataset integration for comprehensive analysis
"""
from .hiv_analysis import HIVAnalysisIntegration
from .sequence_processor import SequenceProcessor
from .results_extractor import ResultsExtractor, ComprehensiveResults, ExtractionResult

__all__ = [
    "HIVAnalysisIntegration",
    "SequenceProcessor",
    "ResultsExtractor",
    "ComprehensiveResults",
    "ExtractionResult",
]
