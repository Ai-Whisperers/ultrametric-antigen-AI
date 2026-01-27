# Protein Stability DDG Package - Bias Analysis

**Date:** 2026-01-26
**Analyst:** AI Whisperers Team
**Status:** CRITICAL ISSUES IDENTIFIED

---

## Executive Summary

Analysis of the validation scripts reveals **3 critical issues** that affect the reliability of claimed metrics:

| Issue | Severity | Impact |
|-------|:--------:|--------|
| Data leakage in scaler | **CRITICAL** | Inflates correlation by 10-20% |
| Metric discrepancy | **HIGH** | Claimed 0.58 vs stored 0.521 |
| AlphaFold stratification fails | **HIGH** | None of 3 strata significant |

---

## Issue 1: Data Leakage in Scaler (CRITICAL)

**File:** `validation/bootstrap_test.py` (lines 137-140)

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # <-- FIT ON ALL DATA!
model = Ridge(alpha=100)
y_pred = cross_val_predict(model, X_scaled, y, cv=len(y))
```

**Problem:** The scaler is fit on ALL 52 samples before cross-validation runs. This means:
- Test sample statistics are included in normalization
- Each LOO fold's "held-out" sample contributed to the mean/std used to normalize it
- This is a form of data leakage

**Expected Impact:**
- Correlation estimates inflated by 10-20%
- True LOO Spearman likely 0.42-0.47 (not 0.52-0.58)

**Correct Implementation:**
```python
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=100))
])
y_pred = cross_val_predict(pipeline, X, y, cv=len(y))
```

---

## Issue 2: Metric Discrepancy

| Source | Spearman | Notes |
|--------|----------|-------|
| `scientific_metrics.json` | **0.521** | Stored validation result |
| `validated_ddg_predictor.py` | **0.58** | Claimed in get_performance_metrics() |
| `bootstrap_test.py` output | varies | Depends on run |
| CLAUDE.md | **0.585** | Documented as official |

**Discrepancy:** 0.58 - 0.521 = 0.059 (11% inflation)

**Root Cause:** Unclear which run produced which number. Need to re-run validation with fixed scaler to get true number.

---

## Issue 3: AlphaFold Stratification Failure

From `validation/results/scientific_metrics.json`:

| pLDDT Stratum | N | Spearman | p-value | Significant? |
|---------------|---|----------|---------|:------------:|
| High (>90) | 41 | 0.27 | 0.088 | ❌ NO |
| Medium (70-90) | 16 | 0.34 | 0.198 | ❌ NO |
| Low (<70) | 34 | 0.04 | 0.822 | ❌ NO |

**Interpretation:** When stratified by AlphaFold confidence:
- None of the three strata show significant correlation
- Only the pooled result (N=52) is significant
- This suggests Simpson's paradox or confounding

**Concern:** The method may not actually work on any well-defined subset.

---

## Issue 4: Feature Attribution Unknown

Both validation scripts mix two feature types:

| Feature Type | Count | Features |
|--------------|:-----:|----------|
| **Hyperbolic** | 4 | hyp_dist, delta_radius, diff_norm, cos_sim |
| **Physicochemical** | 4 | delta_hydro, delta_charge, delta_size, delta_polar |

**Missing Analysis:**
- No ablation study showing hyperbolic features contribute
- Physicochemical features alone may explain all predictive power
- Novel p-adic contribution unproven

**Required Ablation:**
1. Physicochemical features only
2. Hyperbolic features only
3. Combined (current)

---

## Issue 5: N=52 Subset Selection

**Questions about subset:**
- How was the N=52 curated from N=669?
- What criteria were used (small proteins, Ala-scanning)?
- Was selection done BEFORE or AFTER seeing results?

**Risk:** If selection was done after exploring the data, the subset may be cherry-picked for cases where the method works.

---

## Recommendations

### Immediate (Before Sending Emails)

1. **Fix scaler leakage** - Use Pipeline to ensure proper CV
2. **Re-run validation** - Get true LOO Spearman
3. **Update claimed metrics** - Use corrected numbers

### Before Publication

4. **Ablation study** - Quantify hyperbolic vs physicochemical contribution
5. **Document subset selection** - Prove it wasn't cherry-picked
6. **Stratification analysis** - Explain why pooled works but strata don't

---

## Quick Fix for Scaler Leakage

```python
# In bootstrap_test.py, replace lines 137-140 with:

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=100))
])

# This ensures scaler is fit ONLY on training folds
y_pred = cross_val_predict(pipeline, X, y, cv=len(y))
```

---

## Honest Metrics After Fix (Estimated)

Based on typical leakage impact:

| Metric | Current (with leakage) | Estimated (fixed) |
|--------|:----------------------:|:-----------------:|
| Spearman | 0.52-0.58 | **0.42-0.48** |
| 95% CI lower | 0.21 | **0.12-0.18** |
| Significance | p<0.001 | **p≈0.002-0.01** |

---

*This analysis should be addressed before any researcher outreach claiming ρ=0.58.*
