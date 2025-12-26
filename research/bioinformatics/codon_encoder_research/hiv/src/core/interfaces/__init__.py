"""
Core interfaces (protocols) for dependency inversion.

These protocols define the contracts that implementations must follow.
"""
from .encoder import IEncoder, ICodonEncoder, ISequenceEncoder
from .predictor import IPredictor, ITrainablePredictor, Prediction
from .repository import IRepository, ISequenceRepository, IMutationRepository
from .analyzer import IAnalyzer, AnalysisResult

__all__ = [
    # Encoder interfaces
    "IEncoder",
    "ICodonEncoder",
    "ISequenceEncoder",
    # Predictor interfaces
    "IPredictor",
    "ITrainablePredictor",
    "Prediction",
    # Repository interfaces
    "IRepository",
    "ISequenceRepository",
    "IMutationRepository",
    # Analyzer interfaces
    "IAnalyzer",
    "AnalysisResult",
]
