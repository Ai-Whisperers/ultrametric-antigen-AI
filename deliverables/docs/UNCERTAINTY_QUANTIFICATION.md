# Uncertainty Quantification Guide

**Version**: 1.0.0
**Updated**: 2025-12-30
**Authors**: AI Whisperers

---

## Overview

Uncertainty quantification (UQ) is critical for making reliable predictions in bioinformatics. This module provides methods to estimate prediction confidence intervals, helping researchers understand when to trust model outputs.

## Why Uncertainty Matters

Point predictions without uncertainty estimates can be misleading:

```
# Without uncertainty:
Predicted MIC: 4.5 ug/mL  # Is this reliable?

# With uncertainty (90% confidence):
Predicted MIC: 4.5 ug/mL [2.1, 8.3]  # Now we know the range
```

Uncertainty estimates help with:
- **Decision making**: Should we synthesize this peptide?
- **Experimental design**: Where should we focus validation?
- **Model comparison**: Which model is more reliable?
- **Risk assessment**: What's the worst-case scenario?

---

## Available Methods

### 1. Bootstrap Prediction Intervals

The bootstrap method resamples training data to create multiple models and estimates uncertainty from their prediction spread.

**How it works**:
1. Create N bootstrap samples from training data
2. Train a model on each sample
3. Make predictions with all N models
4. Use prediction percentiles as confidence bounds

**Advantages**:
- Works with any model type
- Captures model uncertainty
- No distributional assumptions

**Disadvantages**:
- Computationally expensive (N model fits)
- May underestimate uncertainty for small datasets

```python
from shared import bootstrap_prediction_interval

mean_pred, lower, upper = bootstrap_prediction_interval(
    model=trained_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    n_bootstrap=100,      # Number of bootstrap samples
    confidence=0.90,      # 90% confidence interval
    random_state=42
)

# Results:
# mean_pred: Array of point predictions
# lower: Lower bound of confidence interval
# upper: Upper bound of confidence interval
```

### 2. Ensemble Prediction Intervals

For ensemble models (Random Forest, Gradient Boosting), use variance across trees.

**How it works**:
1. Get predictions from individual trees
2. Compute mean and standard deviation
3. Use z-score for confidence bounds

**Advantages**:
- Fast (no refitting required)
- Natural for tree-based models

**Disadvantages**:
- Only works for ensemble models
- May underestimate uncertainty

```python
from shared import ensemble_prediction_interval

mean_pred, lower, upper = ensemble_prediction_interval(
    model=gradient_boosting_model,
    X_test=X_test,
    confidence=0.90
)
```

### 3. Quantile Regression

Train separate models for different quantiles of the distribution.

**How it works**:
1. Train model for lower quantile (e.g., 5th percentile)
2. Train model for upper quantile (e.g., 95th percentile)
3. Train model for median (50th percentile)

**Advantages**:
- Direct prediction of intervals
- Can capture asymmetric uncertainty
- Works well for heteroscedastic data

**Disadvantages**:
- Requires 3x training time
- Bounds can cross (needs post-processing)

```python
from shared import quantile_prediction_interval

median, lower, upper = quantile_prediction_interval(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    alpha=0.10,  # 90% interval (5th to 95th percentile)
)
```

### 4. Calibration

Adjust prediction intervals to achieve target coverage on a calibration set.

**How it works**:
1. Compute intervals on calibration data
2. Calculate actual coverage
3. Scale intervals to match target

```python
from shared import calibrate_prediction_interval

calibrated_lower, calibrated_upper, achieved_coverage = calibrate_prediction_interval(
    y_true=y_calibration,
    lower=predicted_lower,
    upper=predicted_upper,
    target_coverage=0.90
)

print(f"Achieved coverage: {achieved_coverage:.1%}")
```

---

## UncertaintyPredictor Class

A unified interface for predictions with uncertainty:

```python
from shared import UncertaintyPredictor

# Create predictor
predictor = UncertaintyPredictor(
    model=trained_model,
    scaler=fitted_scaler,      # Optional StandardScaler
    method="bootstrap",         # "bootstrap", "ensemble", or "quantile"
    confidence=0.90,
    n_bootstrap=50              # For bootstrap method
)

# Store training data (required for bootstrap)
predictor.fit(X_train_scaled, y_train)

# Make predictions with uncertainty
result = predictor.predict_with_uncertainty(X_test)

print(f"Prediction: {result['prediction']}")
print(f"Lower bound: {result['lower']}")
print(f"Upper bound: {result['upper']}")
print(f"Confidence: {result['confidence']}")
print(f"Method: {result['method']}")
```

---

## Metrics with Uncertainty

Evaluate predictions including interval quality:

```python
from shared import compute_prediction_metrics_with_uncertainty

metrics = compute_prediction_metrics_with_uncertainty(
    y_true=y_test,
    y_pred=predictions,
    lower=lower_bounds,
    upper=upper_bounds
)

print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")
print(f"Pearson r: {metrics['pearson_r']:.3f}")
print(f"Coverage: {metrics['coverage']:.1%}")          # % within interval
print(f"Mean interval width: {metrics['mean_interval_width']:.3f}")
print(f"Width-error correlation: {metrics['width_error_correlation']:.3f}")
```

### Understanding Metrics

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Coverage | 90% | Should match confidence level |
| Mean interval width | Lower is better | Narrower = more precise |
| Width-error correlation | Positive | Wider intervals for harder cases |

---

## Practical Examples

### Example 1: AMP Activity Prediction with Uncertainty

```python
from carlos_brizuela.scripts.dramp_activity_loader import DRAMPLoader
from shared import UncertaintyPredictor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Load data
loader = DRAMPLoader()
db = loader.generate_curated_database()
X, y = db.get_training_data(target="Escherichia coli")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
model.fit(X_scaled, y)

# Create uncertainty predictor
predictor = UncertaintyPredictor(
    model=model,
    scaler=scaler,
    method="bootstrap",
    confidence=0.90,
    n_bootstrap=50
)
predictor.fit(X_scaled, y)

# Predict for new peptide
from shared import compute_ml_features
new_peptide = "KWKLFKKIGAVLKVL"
X_new = compute_ml_features(new_peptide).reshape(1, -1)

result = predictor.predict_with_uncertainty(X_new)
print(f"Predicted log2(MIC): {result['prediction'][0]:.2f}")
print(f"90% CI: [{result['lower'][0]:.2f}, {result['upper'][0]:.2f}]")
```

### Example 2: Comparing Uncertainty Methods

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from shared import (
    bootstrap_prediction_interval,
    ensemble_prediction_interval,
    quantile_prediction_interval,
)

# Prepare data
# ... (load and split data)

# Method 1: Bootstrap
mean_boot, lower_boot, upper_boot = bootstrap_prediction_interval(
    model, X_train, y_train, X_test, n_bootstrap=100
)

# Method 2: Ensemble (for GradientBoosting)
mean_ens, lower_ens, upper_ens = ensemble_prediction_interval(model, X_test)

# Method 3: Quantile Regression
median_q, lower_q, upper_q = quantile_prediction_interval(
    X_train, y_train, X_test
)

# Compare interval widths
print(f"Bootstrap mean width: {np.mean(upper_boot - lower_boot):.3f}")
print(f"Ensemble mean width: {np.mean(upper_ens - lower_ens):.3f}")
print(f"Quantile mean width: {np.mean(upper_q - lower_q):.3f}")
```

### Example 3: Visualizing Uncertainty

```python
import matplotlib.pyplot as plt
import numpy as np

# Get predictions with uncertainty
result = predictor.predict_with_uncertainty(X_test)

# Sort by prediction for visualization
idx = np.argsort(result['prediction'])

fig, ax = plt.subplots(figsize=(10, 6))

# Plot prediction intervals
ax.fill_between(
    range(len(idx)),
    result['lower'][idx],
    result['upper'][idx],
    alpha=0.3, label='90% CI'
)

# Plot predictions
ax.plot(result['prediction'][idx], 'b-', label='Prediction')

# Plot true values
ax.scatter(range(len(idx)), y_test[idx], c='red', s=20, label='True', zorder=5)

ax.set_xlabel('Sample (sorted)')
ax.set_ylabel('Value')
ax.legend()
plt.title('Predictions with Uncertainty Intervals')
plt.show()
```

---

## Guidelines for Method Selection

| Scenario | Recommended Method |
|----------|-------------------|
| Any model, need robust intervals | Bootstrap |
| GradientBoosting/RandomForest, fast inference | Ensemble |
| Heteroscedastic data, asymmetric uncertainty | Quantile |
| Need calibrated coverage | Bootstrap + Calibration |
| Very large dataset | Ensemble (fastest) |
| Small dataset (<50 samples) | Bootstrap with small n_bootstrap |

---

## Troubleshooting

### Issue: Coverage is too low

**Symptom**: Only 70% of true values fall within "90% confidence" intervals

**Solutions**:
1. Use calibration to adjust intervals
2. Increase n_bootstrap (for bootstrap method)
3. Check for model misspecification

### Issue: Intervals are too wide

**Symptom**: Intervals cover almost everything but are uninformative

**Solutions**:
1. Improve base model (more features, better hyperparameters)
2. Increase training data
3. Use quantile regression for tighter bounds

### Issue: Bootstrap is too slow

**Solutions**:
1. Reduce n_bootstrap (50 often sufficient)
2. Use ensemble method if applicable
3. Parallelize bootstrap (modify code for joblib)

---

## Mathematical Background

### Bootstrap Percentile Method

Given training data $(X, y)$ and test point $x^*$:

1. For $b = 1, ..., B$:
   - Sample $(X^{(b)}, y^{(b)})$ with replacement
   - Fit model $f^{(b)}$ on $(X^{(b)}, y^{(b)})$
   - Predict $\hat{y}^{(b)} = f^{(b)}(x^*)$

2. Interval: $[\text{percentile}(\hat{y}^{(1:B)}, \alpha/2), \text{percentile}(\hat{y}^{(1:B)}, 1-\alpha/2)]$

### Quantile Regression

Instead of minimizing MSE, minimize:

$$L_\tau(y, \hat{y}) = \sum_i \rho_\tau(y_i - \hat{y}_i)$$

where $\rho_\tau(u) = u(\tau - \mathbb{1}(u < 0))$

This gives estimates of the $\tau$-th quantile of $Y|X$.

---

## References

1. Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap. CRC press.

2. Meinshausen, N. (2006). Quantile regression forests. Journal of Machine Learning Research.

3. Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. JASA.
