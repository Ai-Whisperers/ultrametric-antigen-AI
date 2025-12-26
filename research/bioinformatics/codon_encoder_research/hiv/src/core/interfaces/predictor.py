"""
Predictor interfaces - protocols for prediction models.

Defines contracts for resistance prediction, escape prediction,
tropism classification, and other ML-based predictions.
"""
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, Optional, runtime_checkable


@dataclass(slots=True)
class Prediction:
    """
    Base prediction result.

    Attributes:
        value: The predicted value (interpretation depends on task)
        confidence: Confidence score (0-1)
        label: Optional categorical label
        probabilities: Optional class probabilities
        metadata: Additional prediction metadata
    """

    value: float
    confidence: float
    label: Optional[str] = None
    probabilities: Optional[dict[str, float]] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1: {self.confidence}")


T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output", bound=Prediction)


@runtime_checkable
class IPredictor(Protocol[T_Input, T_Output]):
    """Protocol for predictors."""

    def predict(self, input_data: T_Input) -> T_Output:
        """
        Make a single prediction.

        Args:
            input_data: Input for prediction

        Returns:
            Prediction result
        """
        ...

    def predict_batch(self, inputs: list[T_Input]) -> list[T_Output]:
        """
        Make batch predictions.

        Args:
            inputs: List of inputs

        Returns:
            List of predictions
        """
        ...

    @property
    def name(self) -> str:
        """Get predictor name."""
        ...


@runtime_checkable
class ITrainablePredictor(IPredictor[T_Input, T_Output], Protocol):
    """Protocol for trainable predictors (ML models)."""

    def train(
        self,
        train_data: list[tuple[T_Input, T_Output]],
        val_data: Optional[list[tuple[T_Input, T_Output]]] = None,
    ) -> dict:
        """
        Train the model.

        Args:
            train_data: Training data as (input, target) pairs
            val_data: Optional validation data

        Returns:
            Dictionary with training metrics
        """
        ...

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        ...

    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        ...

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        ...


@runtime_checkable
class IEnsemblePredictor(IPredictor[T_Input, T_Output], Protocol):
    """Protocol for ensemble predictors."""

    @property
    def models(self) -> list[IPredictor]:
        """Get constituent models."""
        ...

    def predict_with_uncertainty(
        self, input_data: T_Input
    ) -> tuple[T_Output, float]:
        """
        Predict with uncertainty estimate.

        Args:
            input_data: Input for prediction

        Returns:
            Tuple of (prediction, uncertainty)
        """
        ...
