# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Uncertainty quantification utilities for ML predictions.

This module provides methods for computing prediction intervals and
confidence estimates for regression and classification models.

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
    from sklearn.model_selection import cross_val_predict
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
    # This is a simplified approach - sum all tree predictions
    for i, estimator in enumerate(model.estimators_.ravel()):
        tree_predictions[i] = estimator.predict(X_test)

    # Compute cumulative predictions (how GB works)
    # The final prediction is init + learning_rate * sum(tree_preds)
    mean_pred = model.predict(X_test)

    # Estimate variance from tree variability
    tree_std = np.std(tree_predictions, axis=0) * model.learning_rate
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)

    lower = mean_pred - z_score * tree_std
    upper = mean_pred + z_score * tree_std

    return mean_pred, lower, upper


def quantile_prediction_interval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.10,
    **model_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute prediction intervals using quantile regression.

    Trains separate models for lower and upper quantiles.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        alpha: Miscoverage rate (0.10 = 90% interval)
        **model_kwargs: Kwargs for GradientBoostingRegressor

    Returns:
        Tuple of (median_predictions, lower_bound, upper_bound)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for quantile intervals")

    default_kwargs = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_leaf": 2,
        "random_state": 42,
    }
    default_kwargs.update(model_kwargs)

    # Train lower quantile model
    lower_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha / 2,
        **default_kwargs
    )
    lower_model.fit(X_train, y_train)
    lower = lower_model.predict(X_test)

    # Train upper quantile model
    upper_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=1 - alpha / 2,
        **default_kwargs
    )
    upper_model.fit(X_train, y_train)
    upper = upper_model.predict(X_test)

    # Train median model
    median_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.5,
        **default_kwargs
    )
    median_model.fit(X_train, y_train)
    median = median_model.predict(X_test)

    return median, lower, upper


def calibrate_prediction_interval(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrate prediction intervals to achieve target coverage.

    Adjusts interval width to match observed coverage on a calibration set.

    Args:
        y_true: True values
        lower: Lower bounds
        upper: Upper bounds
        target_coverage: Desired coverage level

    Returns:
        Tuple of (calibrated_lower, calibrated_upper, achieved_coverage)
    """
    # Compute residuals relative to interval bounds
    width = upper - lower
    center = (upper + lower) / 2
    half_width = width / 2

    # Find scale factor to achieve target coverage
    # Sort by how far outside the interval each point is
    relative_positions = np.abs(y_true - center) / half_width

    # Find the quantile that achieves target coverage
    scale_factor = np.percentile(relative_positions, target_coverage * 100)

    # Apply calibration
    calibrated_lower = center - half_width * scale_factor
    calibrated_upper = center + half_width * scale_factor

    # Check achieved coverage
    in_interval = (y_true >= calibrated_lower) & (y_true <= calibrated_upper)
    achieved_coverage = np.mean(in_interval)

    return calibrated_lower, calibrated_upper, achieved_coverage


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


def compute_prediction_metrics_with_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict:
    """Compute metrics including uncertainty quality.

    Args:
        y_true: True values
        y_pred: Predicted values
        lower: Lower bounds
        upper: Upper bounds

    Returns:
        Dictionary of metrics
    """
    from scipy.stats import pearsonr

    # Standard regression metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r, p = pearsonr(y_true, y_pred)

    # Interval quality metrics
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    interval_width = np.mean(upper - lower)

    # Calibration: are wider intervals for harder predictions?
    pred_errors = np.abs(y_true - y_pred)
    interval_widths = upper - lower
    if len(set(interval_widths)) > 1:  # Not all same width
        width_error_corr, _ = pearsonr(interval_widths, pred_errors)
    else:
        width_error_corr = 0.0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "coverage": float(coverage),
        "mean_interval_width": float(interval_width),
        "width_error_correlation": float(width_error_corr),
    }
