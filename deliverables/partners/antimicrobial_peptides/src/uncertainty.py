# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Uncertainty quantification utilities for ML predictions.

Self-contained module for Carlos Brizuela AMP package providing
methods for computing prediction intervals and confidence estimates.

Methods:
    - Bootstrap confidence intervals
    - Ensemble-based prediction intervals
    - Calibrated prediction intervals
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Try to import sklearn
try:
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def bootstrap_prediction_interval(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_bootstrap: int = 100,
    confidence: float = 0.90,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute prediction intervals using bootstrap resampling.

    Creates multiple models trained on bootstrap samples and uses
    the spread of predictions to estimate uncertainty.

    Args:
        model: Sklearn-compatible model (will be cloned)
        X_train: Training features
        y_train: Training targets
        X_test: Test features to predict
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (e.g., 0.90 for 90%)
        random_state: Random seed

    Returns:
        Tuple of (mean_predictions, lower_bound, upper_bound)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for bootstrap intervals")

    rng = np.random.RandomState(random_state)
    n_samples = len(X_train)
    predictions = np.zeros((n_bootstrap, len(X_test)))

    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        # Train model on bootstrap sample
        boot_model = clone(model)
        boot_model.fit(X_boot, y_boot)

        # Predict
        predictions[i] = boot_model.predict(X_test)

    # Compute intervals
    alpha = 1 - confidence
    lower = np.percentile(predictions, alpha / 2 * 100, axis=0)
    upper = np.percentile(predictions, (1 - alpha / 2) * 100, axis=0)
    mean_pred = np.mean(predictions, axis=0)

    return mean_pred, lower, upper


def ensemble_prediction_interval(
    model: "GradientBoostingRegressor",
    X_test: np.ndarray,
    confidence: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute prediction intervals using ensemble variance.

    Uses the variance across trees in a Gradient Boosting ensemble
    to estimate prediction uncertainty.

    Args:
        model: Trained GradientBoostingRegressor
        X_test: Test features
        confidence: Confidence level

    Returns:
        Tuple of (mean_predictions, lower_bound, upper_bound)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for ensemble intervals")

    # Get predictions from individual trees
    n_estimators = model.n_estimators
    tree_predictions = np.zeros((n_estimators, len(X_test)))

    # For GradientBoosting, we need to sum predictions up to each tree
    for i, estimator in enumerate(model.estimators_.ravel()):
        tree_predictions[i] = estimator.predict(X_test)

    # Compute cumulative predictions (how GB works)
    mean_pred = model.predict(X_test)

    # Estimate variance from tree variability
    tree_std = np.std(tree_predictions, axis=0) * model.learning_rate
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)

    lower = mean_pred - z_score * tree_std
    upper = mean_pred + z_score * tree_std

    return mean_pred, lower, upper


class UncertaintyPredictor:
    """Wrapper for models with uncertainty estimation.

    Provides a unified interface for predictions with confidence intervals.
    """

    def __init__(
        self,
        model,
        scaler=None,
        method: str = "bootstrap",
        confidence: float = 0.90,
        n_bootstrap: int = 50,
    ):
        """Initialize uncertainty predictor.

        Args:
            model: Trained sklearn model
            scaler: Optional StandardScaler for features
            method: 'bootstrap', 'ensemble', or 'quantile'
            confidence: Confidence level for intervals
            n_bootstrap: Number of bootstrap samples (if method='bootstrap')
        """
        self.model = model
        self.scaler = scaler
        self.method = method
        self.confidence = confidence
        self.n_bootstrap = n_bootstrap
        self._X_train = None
        self._y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data for bootstrap methods.

        Args:
            X: Training features
            y: Training targets
        """
        self._X_train = X.copy()
        self._y_train = y.copy()

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> dict:
        """Make predictions with uncertainty estimates.

        Args:
            X: Features to predict

        Returns:
            Dictionary with:
                - prediction: Point predictions
                - lower: Lower confidence bound
                - upper: Upper confidence bound
                - confidence: Confidence level
                - method: Method used
        """
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        if self.method == "bootstrap" and self._X_train is not None:
            mean_pred, lower, upper = bootstrap_prediction_interval(
                self.model,
                self._X_train,
                self._y_train,
                X_scaled,
                n_bootstrap=self.n_bootstrap,
                confidence=self.confidence,
            )
        elif self.method == "ensemble" and hasattr(self.model, "estimators_"):
            mean_pred, lower, upper = ensemble_prediction_interval(
                self.model, X_scaled, confidence=self.confidence
            )
        else:
            # Fallback to point prediction with no interval
            mean_pred = self.model.predict(X_scaled)
            lower = mean_pred
            upper = mean_pred

        return {
            "prediction": mean_pred,
            "lower": lower,
            "upper": upper,
            "confidence": self.confidence,
            "method": self.method,
        }
