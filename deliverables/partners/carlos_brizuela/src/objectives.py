from abc import ABC, abstractmethod
import numpy as np


class ObjectiveFunction(ABC):
    """Abstract base class for biochemical property predictors."""

    @abstractmethod
    def predict(self, sequences: list[str]) -> np.ndarray:
        """Predict property scores for a list of sequences."""
        pass


class ToxicityPredictor(ObjectiveFunction):
    """Predicts peptide toxicity (lower is better/safer)."""

    def predict(self, sequences: list[str]) -> np.ndarray:
        # Placeholder: Mock implementation
        # Real implementation would call ToxinPred2 or load a regressor
        return np.random.random(len(sequences))


class ActivityPredictor(ObjectiveFunction):
    """Predicts antimicrobial activity (higher is better)."""

    def predict(self, sequences: list[str]) -> np.ndarray:
        # Placeholder: Mock implementation
        # Real implementation would use AMP Scanner or similar
        return np.random.random(len(sequences))
