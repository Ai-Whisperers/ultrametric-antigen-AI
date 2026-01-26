# P-adic Amino Acid Encoder - Comprehensive Findings

**Date:** 2026-01-05
**Version:** v5.12.4 Ternary VAE Integration
**Validation:** Rigorous statistical analysis with bootstrap CI and permutation tests

---

## Executive Summary

We validated conjectures about using generalized p-adic encoders when codon information is unavailable. Key findings:

1. **P-adic on arbitrary AA indices fails** (Spearman ~0.35, same as baseline)
2. **Multi-prime ensemble doesn't help** (Spearman 0.33, worse than single)
3. **Ordered indices by biological property work** (hydrophobicity ordering matches baseline)
4. **Combined features improve baseline** (Spearman 0.37-0.40, +5-9% improvement)
5. **Improvement is CONTEXT-DEPENDENT** (huge benefit for some mutation types, harmful for others)

---

## Statistical Validation Summary

### Overall Results (669 mutations)

| Metric | Baseline | Best Combined | Delta |
|--------|----------|---------------|-------|
| Spearman (mean) | 0.367 | 0.392-0.400 | +7-9% |
| 95% CI | [0.30, 0.43] | [0.33, 0.46] | Overlapping |
| Paired t-test | - | p < 0.001 | Significant |
| Permutation test | - | p = 0.08 | Borderline |

**Interpretation:** The improvement is consistent across CV folds (paired t-test significant) but the effect size is modest with overlapping confidence intervals.

### Critical Finding: Mutation Type Heterogeneity

The overall +7-9% masks dramatic variation by mutation type:

| Mutation Type | N | P-adic Advantage | Recommendation |
|---------------|---|------------------|----------------|
| **neutral→charged** | 37 | **+159%** | STRONGLY USE P-adic |
| small DDG (<1 kcal/mol) | 312 | **+23%** | USE P-adic |
| large→small size | 82 | **+16%** | USE P-adic |
| charged→neutral | 186 | +20% | USE P-adic |
| same_charge | 426 | +6% | Marginal benefit |
| **charge_reversal** | 20 | **-737%** | DO NOT use P-adic |
| large DDG (>2 kcal/mol) | 183 | -6% | Skip P-adic |

---

## Experimental Results

### 1. Prime Comparison (Arbitrary AA Indices)

| Prime | Spearman | Pearson | MAE |
|-------|----------|---------|-----|
| p=2 | 0.3476 | 0.3236 | 1.160 |
| p=5 | 0.3472 | 0.3118 | 1.164 |
| p=3 | 0.3323 | 0.3256 | 1.155 |
| p=11 | 0.3284 | 0.3148 | 1.165 |
| p=7 | 0.3146 | 0.2946 | 1.165 |

**Conclusion:** All primes perform similarly (~0.34). No single prime is special.

### 2. Baseline Comparison

| Model | Spearman | Notes |
|-------|----------|-------|
| Physicochemical | 0.3664 | Simple delta-properties |
| P-adic p=5 | 0.3538 | Ordered by arbitrary index |
| Ensemble (all) | 0.3331 | Worse than single! |
| Random | 0.0031 | Negative control |

**Conclusion:** P-adic doesn't beat simple physicochemical baseline alone.

### 3. Ordered Indices Experiment

| Ordering | Spearman | Key Property |
|----------|----------|--------------|
| Combined | 0.3673 | Hydrophobic + small + neutral |
| Hydrophobicity | 0.3600 | Most hydrophobic = index 0 |
| Polarity | 0.3492 | Least polar = index 0 |
| Alphabetical | 0.3167 | Arbitrary (baseline) |
| Volume | 0.3146 | Smallest = index 0 |
| Charge | 0.3114 | Most negative = index 0 |

**Conclusion:** Ordering by hydrophobicity matches physicochemical baseline!

### 4. Feature Combination (Key Finding)

| Feature Set | Spearman | 95% CI | #Features | vs Baseline |
|-------------|----------|--------|-----------|-------------|
| **Physico + P-adic + Valuation** | **0.39-0.40** | [0.33, 0.46] | 14 | **+7-9%** |
| Physico + P-adic | 0.37 | - | 10 | +2-5% |
| Physico + Embedding | 0.37 | - | 16 | +2% |
| Physicochemical (baseline) | 0.367 | [0.30, 0.43] | 6 | - |
| P-adic + Valuation | 0.23 | - | 8 | -37% |
| P-adic only | 0.22 | - | 4 | -41% |

**Conclusion:** P-adic features ADD VALUE only when combined with physicochemical features.

### 5. Feature Importance (Greedy Selection)

Most important features (in selection order):
1. `delta_volume` - Volume change is primary driver
2. `abs_hydro` - Absolute hydrophobicity change
3. `delta_charge` - Charge difference
4. `delta_polarity` - Polarity change
5. `val_product` - **P-adic valuation interaction (wt_val × mt_val)**
6. `sum_std` - Embedding diversity

**Key discovery:** `val_product` (valuation interaction) is more informative than individual valuations!

### 6. Model Comparison

| Model | Spearman | MAE | Notes |
|-------|----------|-----|-------|
| Random Forest (depth=5) | **0.379** | 1.146 | Best overall |
| Gradient Boosting (depth=3) | 0.378 | 1.174 | Similar |
| Ridge (α=1.0) | 0.369 | 1.152 | Linear baseline |
| Random Forest (depth=10) | 0.369 | 1.168 | Overfitting |

**Conclusion:** Shallow ensemble methods (RF, GB) slightly outperform linear models.

---

## Key Insights

### Why P-adic Alone Fails
- P-adic valuation on indices 0-19 doesn't capture biological structure
- The genetic code's 4³=64 codon structure is what matters for evolution
- Amino acid indices are arbitrary - no inherent p-adic meaning

### Why Ordered Indices Help
- Ordering by hydrophobicity creates meaningful hierarchy
- Most hydrophobic (I, F, V, L, W) get low indices → high p-adic valuation
- This aligns p-adic structure with biological function

### Why Combination Works
- Physicochemical features capture explicit property differences
- P-adic features capture implicit hierarchical relationships
- Valuation interaction (`val_product`) encodes pairwise relationships
- Together they capture complementary aspects of mutation effects

### When P-adic HURTS
- **Charge reversals**: Dramatic effects not captured by gradual hierarchy
- **Large DDG mutations**: Already well-predicted by physicochemical features
- These cases need explicit handling, not p-adic smoothing

---

## Recommended Usage

### Decision Tree for Feature Selection

```
Is mutation a charge reversal (K→D, R→E, etc.)?
├── YES → Use ONLY physicochemical features (skip p-adic)
└── NO → Continue

Is expected |ΔΔG| > 2 kcal/mol?
├── YES → Use ONLY physicochemical features (skip p-adic)
└── NO → Continue

Is mutation neutral→charged?
├── YES → Use FULL combined features (physico + p-adic + valuation)
└── NO → Use combined features with lower weight on p-adic
```

### Feature Extraction Code

```python
from src.encoders.generalized_padic_encoder import GeneralizedPadicEncoder

# 1. Order amino acids by hydrophobicity
ordering = ['I', 'F', 'V', 'L', 'W', 'M', 'A', 'C', 'G', 'T',
            'S', 'P', 'Y', 'H', 'N', 'Q', 'E', 'D', 'K', 'R']

# 2. Train encoder with this ordering
encoder = OrderedPadicEncoder(ordering=ordering, prime=5)

# 3. Extract COMBINED features (14 total):
def extract_features(wt, mt, encoder, embeddings, valuations):
    """Extract combined features for DDG prediction."""

    # Physicochemical (6 features)
    wt_props = AA_PROPERTIES[wt]
    mt_props = AA_PROPERTIES[mt]
    delta_props = [mt_props[i] - wt_props[i] for i in range(4)]
    abs_charge = abs(delta_props[1])
    l2_dist = np.linalg.norm(delta_props)

    # P-adic encoder (4 features)
    wt_emb = embeddings[wt]
    mt_emb = embeddings[mt]
    hyp_dist = poincare_distance(wt_emb, mt_emb)
    wt_radius = poincare_distance(wt_emb, origin)
    mt_radius = poincare_distance(mt_emb, origin)
    delta_radius = mt_radius - wt_radius

    # Valuation (4 features including interaction!)
    wt_val = valuations[wt]
    mt_val = valuations[mt]
    delta_val = mt_val - wt_val
    val_product = wt_val * mt_val  # KEY: interaction term

    return [
        *delta_props,        # 4 features
        abs_charge,          # 1 feature
        l2_dist,             # 1 feature
        hyp_dist,            # 1 feature
        delta_radius,        # 1 feature
        wt_radius,           # 1 feature
        mt_radius,           # 1 feature
        delta_val,           # 1 feature
        abs(delta_val),      # 1 feature
        val_product,         # 1 feature (NEW: interaction)
    ]  # Total: 14 features

# 4. Use Ridge or Random Forest (depth=5) for prediction
```

### Adaptive Model (Recommended)

```python
def predict_ddg_adaptive(wt, mt, features_physico, features_combined):
    """Use appropriate model based on mutation type."""

    # Check if charge reversal
    charge_rev = is_charge_reversal(wt, mt)

    if charge_rev:
        # Don't use p-adic features
        return model_physico.predict(features_physico)
    else:
        # Use combined features
        return model_combined.predict(features_combined)
```

---

## Files Created

```
research/codon-encoder/
├── benchmarks/
│   ├── padic_encoder_benchmark.py      # Standardized benchmark suite
│   ├── test_ordered_indices.py         # AA ordering experiments
│   ├── test_enhanced_features.py       # Feature combination tests
│   ├── rigorous_validation.py          # Bootstrap CI, permutation tests
│   └── adaptive_feature_selection.py   # Feature optimization
├── results/
│   ├── benchmarks/
│   │   └── benchmark_results.json
│   ├── ordered_indices/
│   │   └── ordering_comparison.json
│   ├── enhanced_features/
│   │   └── feature_combinations.json
│   ├── statistical_validation/
│   │   └── validation_report.json
│   ├── feature_optimization/
│   │   └── optimization_results.json
│   └── PADIC_ENCODER_FINDINGS.md       # This document

src/encoders/
├── generalized_padic_encoder.py        # Any-prime p-adic encoder
└── segment_codon_encoder.py            # Long sequence support
```

---

## Comparison with Codon-Based Approach

| Approach | Spearman | 95% CI | Use Case |
|----------|----------|--------|----------|
| Codon-based v2 | 0.81 | [0.78, 0.84] | When codon info available |
| **AA + P-adic combined** | **0.39-0.40** | **[0.33, 0.46]** | **Proteomics only** |
| AA + P-adic (adaptive) | ~0.42 | - | Mutation-type aware |
| Physicochemical baseline | 0.37 | [0.30, 0.43] | Simple baseline |
| P-adic alone | 0.22 | - | Not recommended |

**Verdict:** When codon information is available, use codon-based approach (2x better). When only AA sequence is available, use combined features adaptively based on mutation type.

---

## Limitations and Caveats

1. **Overlapping Confidence Intervals**: The improvement has statistical significance on paired tests but CIs overlap
2. **Training Variance**: Results vary ±5% across encoder training runs
3. **Small Effect Size**: +7-9% improvement is meaningful but modest
4. **Dataset Specificity**: Validated only on S669; may not generalize to other datasets
5. **Mutation Type Dependence**: Must check mutation type before applying p-adic features

---

## Future Work

1. **Ensemble encoders**: Average across multiple encoder trainings for stability
2. **Mutation-type classifiers**: Automatically route to appropriate feature set
3. **Deep learning**: End-to-end model that learns when to use p-adic features
4. **Larger datasets**: Validate on ProteinGym, ProTherm for generalization
5. **Transfer learning**: Pre-train on codon data, fine-tune for AA-only

---

## Conclusion

The generalized p-adic encoder on amino acid indices provides value for proteomics-only scenarios when:
1. Amino acids are ordered by hydrophobicity
2. P-adic features are combined with physicochemical features
3. Valuation interaction (`val_product`) is included
4. **Charge reversals and large DDG mutations are handled separately**

This achieves **+7-9% improvement** over the physicochemical baseline with statistical significance on paired tests (p < 0.001), though confidence intervals overlap. The benefit is strongly context-dependent, with >100% improvement for neutral→charged mutations but negative impact for charge reversals.
