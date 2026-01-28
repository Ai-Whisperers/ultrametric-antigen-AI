# Antimicrobial Peptides Package - Bias Analysis

**Date:** 2026-01-26 (Updated: 2026-01-27)
**Analyst:** AI Whisperers Team
**Status:** ISSUES MOSTLY RESOLVED

---

## Executive Summary

Original analysis (2026-01-26) identified 3 issues. Status update:

| Issue | Severity | Status | Resolution |
|-------|:--------:|:------:|------------|
| Scaler leakage in comprehensive_validation.py | CRITICAL | **FIXED** | Pipeline applied (commit 9b97793b) |
| Metric discrepancy between files | HIGH | **DOCUMENTED** | Use CLAUDE.md conservative values |
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

## Issue 2: Metric Discrepancy - **DOCUMENTED**

Two validation runs exist with different results. The **conservative values from CLAUDE.md** are the authoritative reference:

| Target | N | Pearson r | p-value | Status |
|--------|--:|:---------:|:-------:|:------:|
| **Acinetobacter** | 20 | 0.52 | 0.019 | **Significant** |
| **Escherichia** | 105 | 0.39 | <0.001 | **Significant** |
| **General** | 224 | 0.31 | <0.001 | **Significant** |
| **Pseudomonas** | 27 | 0.05 | 0.82 | NOT Significant |
| **Staphylococcus** | 72 | 0.17 | 0.15 | NOT Significant |

**Resolution:** Use the conservative (smaller N, lower correlation) values for outreach.

**Explanation:** Different runs used different data filtering:
- `comprehensive_validation.json`: Full dataset with potential duplicates
- `CLAUDE.md`: Deduplicated, curated subset

---

## Issue 3: Sklearn Model Training - **FIXED**

**File:** `scripts/dramp_activity_loader.py`

**Fix Applied:** Pipeline pattern now used for both CV scoring and predictions.

---

## Issue 4: PeptideVAE Training - **CORRECT**

The PeptideVAE training dataset module correctly implements per-fold normalization:

```python
# Normalization stats computed on TRAINING data only
train_dataset = AMPDataset(sequences=train_seqs, ...)
val_dataset = AMPDataset(
    property_mean=train_dataset.property_mean,  # Uses TRAIN stats
    property_std=train_dataset.property_std,    # Uses TRAIN stats
)
```

**No fix needed** - this was always correct.

---

## Current Validated Performance

**Authoritative Source:** `deliverables/partners/CLAUDE.md`

| Model | Performance | Reliability |
|-------|:-----------:|:-----------:|
| E. coli (Escherichia) | r=0.39, p<0.001 | **HIGH** - Significant, N=105 |
| Acinetobacter | r=0.52, p=0.019 | **MEDIUM** - Significant, but N=20 |
| General | r=0.31, p<0.001 | **HIGH** - Significant, N=224 |
| Pseudomonas | r=0.05, p=0.82 | **LOW** - NOT significant |
| Staphylococcus | r=0.17, p=0.15 | **LOW** - NOT significant |

---

## For Outreach: Use Conservative Claims

| Claim | Safe? | Notes |
|-------|:-----:|-------|
| "MIC prediction for E. coli (r=0.39, p<0.001)" | **YES** | Verified, N=105 |
| "MIC prediction for Acinetobacter (r=0.52, p=0.019)" | **YES** | But caveat: N=20 is small |
| "General MIC prediction (r=0.31)" | **YES** | Conservative but significant |
| "Works for Pseudomonas" | **NO** | p=0.82, not significant |
| "Works for Staphylococcus" | **NO** | p=0.15, not significant |

---

## Key Limitations to Document

1. **Pseudomonas and Staphylococcus models are NOT significant** - do not claim predictive power
2. **Toxicity/stability are heuristic, not ML models** - do not claim validated ML performance
3. **Small sample sizes** - Acinetobacter (N=20), Pseudomonas (N=27) limit statistical power

---

*This document reflects the corrected state as of 2026-01-27.*
*Previous version (2026-01-26) contained outdated claims about unfixed issues.*
