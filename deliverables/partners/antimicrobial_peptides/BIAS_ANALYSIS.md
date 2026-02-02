# Antimicrobial Peptides Package - Bias Analysis

**Date:** 2026-01-26 (Updated: 2026-02-02)
**Analyst:** AI Whisperers Team
**Status:** ALL ISSUES RESOLVED - ALL 5 MODELS SIGNIFICANT

---

## Executive Summary

All technical issues have been resolved. The dataset was expanded and all 5 models now show statistical significance.

| Issue | Severity | Status | Resolution |
|-------|:--------:|:------:|------------|
| Scaler leakage in comprehensive_validation.py | CRITICAL | **FIXED** | Pipeline applied (commit 9b97793b) |
| Dataset size for Pseudomonas | HIGH | **FIXED** | Expanded from N=27 to N=100 |
| Sklearn model training leakage | CRITICAL | **FIXED** | Pipeline applied |
| PeptideVAE training | N/A | **CORRECT** | No fix needed |

---

## Issue 1: Data Leakage in Scaler - **FIXED**

**File:** `validation/comprehensive_validation.py`

**Previous (broken):**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # <-- FIT ON ALL DATA!
y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
```

**Current (fixed):**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(...))
])
y_pred = cross_val_predict(pipeline, X, y, cv=cv)  # Scaler fits per-fold
```

**Fix Applied:** Commit `9b97793b` (2026-01-26)

---

## Issue 2: Dataset Expansion - **FIXED**

The Pseudomonas dataset was expanded via literature curation (documented in `docs/LIMITATIONS_AND_FUTURE_WORK.md`):

**Before:** N=27, r=0.05, p=0.82 (NOT significant)
**After:** N=100, r=0.506, p<0.001 (HIGHLY significant)

Sources added: LL-37 derivatives, cathelicidins, marine AMPs, designed peptides from peer-reviewed literature (2020-2024).

---

## Issue 3: Sklearn Model Training - **FIXED**

**File:** `scripts/dramp_activity_loader.py`

**Fix Applied:** Pipeline pattern now used for both CV scoring and predictions.

---

## Issue 4: PeptideVAE Training - **CORRECT**

The PeptideVAE training dataset module correctly implements per-fold normalization. No fix needed.

---

## Current Validated Performance

**Authoritative Source:** `validation/results/comprehensive_validation.json`

| Target | N | Pearson r | p-value | Confidence | Status |
|--------|--:|:---------:|:-------:|:----------:|:------:|
| **General** | 425 | 0.608 | 2.4e-44 | HIGH | **Significant** |
| **P. aeruginosa** | 100 | 0.506 | 8.0e-08 | HIGH | **Significant** |
| **E. coli** | 133 | 0.492 | 1.8e-09 | HIGH | **Significant** |
| **A. baumannii** | 88 | 0.463 | 5.7e-06 | HIGH | **Significant** |
| **S. aureus** | 104 | 0.348 | 0.0003 | MODERATE | **Significant** |

**Summary:** 5/5 models significant, 4 HIGH confidence, 1 MODERATE confidence.

---

## For Outreach: All Models Now Validated

| Claim | Safe? | Notes |
|-------|:-----:|-------|
| "MIC prediction for E. coli (r=0.49, p<0.001)" | **YES** | HIGH confidence, N=133 |
| "MIC prediction for P. aeruginosa (r=0.51, p<0.001)" | **YES** | HIGH confidence, N=100 |
| "MIC prediction for A. baumannii (r=0.46, p<0.001)" | **YES** | HIGH confidence, N=88 |
| "MIC prediction for S. aureus (r=0.35, p=0.0003)" | **YES** | MODERATE confidence, N=104 |
| "General MIC prediction (r=0.61, p<0.001)" | **YES** | HIGH confidence, N=425 |

---

## Key Notes

1. **All 5 models are statistically significant** (p < 0.05)
2. **S. aureus has MODERATE confidence** - use for ranking candidates, combine with general model
3. **Toxicity/stability are heuristic, not ML models** - do not claim validated ML performance

---

*This document reflects the validated state from comprehensive_validation.json.*
*Previous versions contained stale metrics from before dataset expansion.*
