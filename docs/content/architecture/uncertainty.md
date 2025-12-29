# Uncertainty Quantification

> **Integrated uncertainty methods for clinical decision support.**

**Module**: `src/diseases/uncertainty_aware_analyzer.py`
**Tests**: `tests/unit/diseases/test_uncertainty_integration.py` (21 tests)

---

## Overview

Uncertainty quantification allows:
- Flagging low-confidence predictions for expert review
- Risk-stratified clinical decision making
- Identification of out-of-distribution samples

### Sources of Uncertainty

| Type | Source | Reducible? |
|------|--------|------------|
| **Epistemic** | Model uncertainty, limited data | Yes (more data) |
| **Aleatoric** | Inherent data noise | No |

---

## Methods

### MC Dropout

Multiple forward passes with dropout enabled:

$$\mu = \frac{1}{T} \sum_{t=1}^T f_{\theta_t}(x)$$
$$\sigma^2 = \frac{1}{T} \sum_{t=1}^T (f_{\theta_t}(x) - \mu)^2$$

### Evidential Deep Learning

Single forward pass outputting distribution parameters:

$$p(y | \gamma, \nu, \alpha, \beta) = \text{NIG}(\gamma, \nu, \alpha, \beta)$$

**Uncertainty decomposition**:
- Epistemic: $\frac{\beta}{\nu(\alpha - 1)}$
- Aleatoric: $\frac{\beta}{\alpha - 1}$

### Ensemble

Multiple independently trained models:

$$\sigma^2_{\text{epistemic}} = \frac{1}{M} \sum_{m=1}^M (f_m(x) - \mu)^2$$

---

## Usage

```python
from src.diseases.uncertainty_aware_analyzer import (
    UncertaintyAwareAnalyzer,
    UncertaintyConfig,
    UncertaintyMethod,
)

# Configure uncertainty
config = UncertaintyConfig(
    method=UncertaintyMethod.EVIDENTIAL,
    n_samples=50,           # For MC Dropout
    confidence_level=0.95,
    calibrate=True,
    decompose=True,         # Epistemic/aleatoric split
)

# Create analyzer
analyzer = UncertaintyAwareAnalyzer(
    base_analyzer=hiv_analyzer,
    config=config,
    model=model,
)

# Analyze with uncertainty
results = analyzer.analyze_with_uncertainty(sequences, encodings=x)

# Access results
print(results["mean"])           # Point predictions
print(results["std"])            # Total uncertainty
print(results["epistemic"])      # Model uncertainty
print(results["aleatoric"])      # Data noise
print(results["confidence_interval"])  # 95% CI
```

---

## Method Comparison

| Method | Forward Passes | Memory | Calibration | Decomposition |
|--------|----------------|--------|-------------|---------------|
| MC Dropout | N (50) | 1x | Good | Epistemic only |
| Evidential | 1 | 1x | Very Good | Full |
| Ensemble | M (5) | Mx | Best | Epistemic only |

### Recommendations

- **Production**: Evidential (fast, full decomposition)
- **Research**: Ensemble (best accuracy)
- **Retrofit existing**: MC Dropout (minimal changes)

---

## Calibration

A model is **calibrated** if predicted confidence matches accuracy:

$$P(\text{correct} | \text{confidence} = p) = p$$

### Temperature Scaling

```python
from src.diseases.uncertainty_aware_analyzer import UncertaintyCalibrator

calibrator = UncertaintyCalibrator()
calibrator.fit(predictions, uncertainties, targets)
calibrated_pred, calibrated_unc = calibrator.calibrate(pred, unc)
```

---

## Clinical Decision Support

```python
def clinical_recommendation(prediction, uncertainty, drug):
    upper = prediction + 1.96 * uncertainty
    lower = prediction - 1.96 * uncertainty

    if lower > 0.7:
        return f"AVOID {drug}: High resistance (>{lower:.0%})"
    if upper < 0.3:
        return f"RECOMMEND {drug}: Low resistance (<{upper:.0%})"
    if uncertainty > 0.2:
        return f"EXPERT REVIEW: High uncertainty ({lower:.0%}-{upper:.0%})"
    return f"CAUTION {drug}: Moderate resistance ({prediction:.0%})"
```

---

_Last updated: 2025-12-28_
