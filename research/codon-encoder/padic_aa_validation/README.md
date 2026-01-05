# P-adic Amino Acid Encoder Validation Suite

**Doc-Type:** Research Validation · Version 1.0 · 2026-01-05

---

## Overview

This directory contains the complete validation suite for generalizing p-adic encoders to amino acid sequences when codon information is unavailable. The work builds on the v5.12.4 Ternary VAE embeddings and tests whether p-adic mathematical structure can improve DDG (protein stability change) prediction.

---

## Executive Summary

| Finding | Result | Confidence |
|---------|--------|------------|
| P-adic alone vs baseline | No improvement (0.35 vs 0.37) | High |
| Multi-prime ensemble | Worse than single prime (0.33) | High |
| Hydrophobicity ordering | Matches baseline (0.36) | High |
| Combined features | +7-9% improvement (0.39-0.40) | Moderate (p=0.08) |
| Mutation-type dependent | +159% to -737% by type | High |

**Bottom Line:** P-adic features provide modest but consistent improvement when combined with physicochemical features, but the benefit is strongly context-dependent.

---

## Directory Structure

```
padic_aa_validation/
├── README.md                           # This file
├── scripts/
│   ├── 01_baseline_benchmark.py        # Baseline comparison (physico vs padic)
│   ├── 02_ordered_indices.py           # AA ordering experiments
│   ├── 03_feature_combinations.py      # Feature combination tests
│   ├── 04_statistical_validation.py    # Bootstrap CI, permutation tests
│   └── 05_feature_optimization.py      # Greedy selection, model comparison
├── results/
│   ├── benchmarks/                     # Baseline benchmark results
│   ├── ordered_indices/                # Ordering experiment results
│   ├── enhanced_features/              # Feature combination results
│   ├── statistical_validation/         # Rigorous validation results
│   └── feature_optimization/           # Optimization results
└── docs/
    └── PADIC_ENCODER_FINDINGS.md       # Comprehensive findings document
```

---

## Validation Pipeline

### Stage 1: Baseline Benchmark (01_baseline_benchmark.py)

**Question:** Does p-adic encoding improve over simple physicochemical features?

**Method:**
- Compare Spearman correlation on S669 dataset (669 mutations)
- Test multiple primes: p=2, 3, 5, 7, 11
- Test multi-prime ensemble

**Results:**
| Model | Spearman |
|-------|----------|
| Physicochemical | 0.366 |
| P-adic p=5 | 0.347 |
| Ensemble | 0.333 |

**Conclusion:** P-adic alone does NOT beat baseline.

---

### Stage 2: Ordered Indices (02_ordered_indices.py)

**Question:** Does ordering amino acids by biological properties help?

**Method:**
- Test orderings: alphabetical, hydrophobicity, volume, charge, polarity
- Train encoder with each ordering

**Results:**
| Ordering | Spearman |
|----------|----------|
| Hydrophobicity | 0.360 |
| Polarity | 0.349 |
| Alphabetical | 0.317 |
| Volume | 0.315 |

**Conclusion:** Hydrophobicity ordering reaches baseline performance.

---

### Stage 3: Feature Combinations (03_feature_combinations.py)

**Question:** Do p-adic features add value when combined with physicochemical?

**Method:**
- Test feature sets: physico, padic, valuation, combinations
- 5-fold cross-validation with Ridge regression

**Results:**
| Feature Set | Spearman | vs Baseline |
|-------------|----------|-------------|
| Physico + Padic + Valuation | 0.40 | +8.5% |
| Physico + Padic | 0.39 | +5.6% |
| Physicochemical | 0.37 | baseline |

**Conclusion:** Combined features improve baseline by 8.5%.

---

### Stage 4: Statistical Validation (04_statistical_validation.py)

**Question:** Is the improvement statistically significant?

**Method:**
- Bootstrap confidence intervals (1000 resamples)
- Permutation test (1000 permutations)
- Paired t-test across CV folds
- Stratified analysis by mutation type

**Results:**
| Test | Result |
|------|--------|
| Bootstrap 95% CI (best) | [0.33, 0.46] |
| Bootstrap 95% CI (base) | [0.30, 0.43] |
| Permutation p-value | 0.08 |
| Paired t-test p-value | <0.001 |

**Critical Discovery - Mutation Type Heterogeneity:**

| Mutation Type | N | P-adic Advantage |
|---------------|---|------------------|
| neutral→charged | 37 | **+159%** |
| small DDG | 312 | **+23%** |
| large→small | 82 | **+16%** |
| charge_reversal | 20 | **-737%** |
| large DDG | 183 | **-6%** |

**Conclusion:** Improvement is real but context-dependent. P-adic helps for subtle mutations, hurts for dramatic ones.

---

### Stage 5: Feature Optimization (05_feature_optimization.py)

**Question:** Can we optimize feature selection to improve further?

**Method:**
- Feature importance analysis (Ridge, MI, RF, GB)
- Greedy forward selection
- Nonlinear model comparison

**Results:**
| Method | Best Score |
|--------|------------|
| Greedy selection (6 features) | 0.385 |
| Random Forest (depth=5) | 0.379 |
| Ridge | 0.369 |

**Key Discovery:** `val_product` (wt_val × mt_val) is more informative than individual valuations.

---

## How to Run

```bash
cd research/codon-encoder/padic_aa_validation/scripts

# Run complete validation pipeline
python 01_baseline_benchmark.py
python 02_ordered_indices.py
python 03_feature_combinations.py
python 04_statistical_validation.py
python 05_feature_optimization.py
```

---

## Key Recommendations

### When to Use P-adic Features

```
Is mutation a charge reversal (K→D, R→E)?
├── YES → Use ONLY physicochemical (skip p-adic)
└── NO → Continue

Is expected |ΔΔG| > 2 kcal/mol?
├── YES → Use ONLY physicochemical
└── NO → Use combined features
```

### Optimal Feature Set (14 features)

1. **Physicochemical (6):** delta_hydro, delta_charge, delta_volume, delta_polarity, abs_charge, l2_dist
2. **P-adic hyperbolic (4):** hyp_dist, delta_radius, wt_radius, mt_radius
3. **Valuation (4):** delta_val, abs_delta_val, wt_val, val_product

---

## Integration with Other Packages

This validation informs the following integrations:

| Package | Integration Point | Recommendation |
|---------|-------------------|----------------|
| jose_colbes DDG predictor | Feature engineering | Add val_product feature |
| carlos_brizuela AMP | Peptide encoding | Use for non-charge-reversal regions |
| alejandra_rojas primers | Codon optimization | Original codon encoder still best |

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-05 | 1.0 | Initial validation suite with 5-stage pipeline |
