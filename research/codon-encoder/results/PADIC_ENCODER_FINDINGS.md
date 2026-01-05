# P-adic Amino Acid Encoder - Comprehensive Findings

**Date:** 2026-01-05
**Version:** v5.12.4 Ternary VAE Integration

---

## Executive Summary

We validated conjectures about using generalized p-adic encoders when codon information is unavailable. Key findings:

1. **P-adic on arbitrary AA indices fails** (Spearman ~0.35, same as baseline)
2. **Multi-prime ensemble doesn't help** (Spearman 0.33, worse than single)
3. **Ordered indices by biological property work** (hydrophobicity ordering matches baseline)
4. **Combined features SURPASS baseline by 8.5%** (Spearman 0.40 vs 0.37)

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

**Conclusion:** P-adic doesn't beat simple physicochemical baseline.

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

| Feature Set | Spearman | #Features | vs Baseline |
|-------------|----------|-----------|-------------|
| **Physico + P-adic + Valuation** | **0.4003** | 14 | **+8.5%** |
| Physico + P-adic | 0.3895 | 10 | +5.6% |
| Physico + Valuation | 0.3771 | 10 | +2.2% |
| Physicochemical (baseline) | 0.3690 | 6 | - |
| P-adic + Valuation | 0.2313 | 8 | -37% |
| P-adic only | 0.2175 | 4 | -41% |
| Valuation only | -0.0106 | 4 | -103% |

**Conclusion:** P-adic features ADD VALUE when combined with physicochemical features.

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
- Valuation features encode position in the hydrophobicity hierarchy
- Together they capture complementary aspects of mutation effects

---

## Recommended Usage

### For Proteomics-Only Scenarios (No Codon Info)

```python
from src.encoders.generalized_padic_encoder import GeneralizedPadicEncoder

# 1. Order amino acids by hydrophobicity
ordering = ['I', 'F', 'V', 'L', 'W', 'M', 'A', 'C', 'G', 'T',
            'S', 'P', 'Y', 'H', 'N', 'Q', 'E', 'D', 'K', 'R']

# 2. Train encoder with this ordering
encoder = OrderedPadicEncoder(ordering=ordering, prime=5)

# 3. Extract COMBINED features:
#    - Physicochemical delta (6 features)
#    - Hyperbolic distance + radii (4 features)
#    - P-adic valuation (4 features)
#    Total: 14 features

# 4. Use Ridge regression for prediction
```

### Feature Extraction

```python
def extract_features(wt, mt, encoder, embeddings, valuations):
    """Extract combined features for DDG prediction."""

    # Physicochemical
    wt_props = AA_PROPERTIES[wt]
    mt_props = AA_PROPERTIES[mt]
    delta_props = [mt_props[i] - wt_props[i] for i in range(4)]
    abs_charge = abs(delta_props[1])
    l2_dist = np.linalg.norm(delta_props)

    # P-adic encoder
    wt_emb = embeddings[wt]
    mt_emb = embeddings[mt]
    hyp_dist = poincare_distance(wt_emb, mt_emb)
    wt_radius = poincare_distance(wt_emb, origin)
    mt_radius = poincare_distance(mt_emb, origin)
    delta_radius = mt_radius - wt_radius

    # Valuation
    wt_val = valuations[wt]
    mt_val = valuations[mt]
    delta_val = mt_val - wt_val
    abs_delta_val = abs(delta_val)

    return [
        *delta_props,        # 4 features
        abs_charge,          # 1 feature
        l2_dist,             # 1 feature
        hyp_dist,            # 1 feature
        delta_radius,        # 1 feature
        wt_radius,           # 1 feature
        mt_radius,           # 1 feature
        wt_val,              # 1 feature
        mt_val,              # 1 feature
        delta_val,           # 1 feature
        abs_delta_val,       # 1 feature
    ]  # Total: 14 features
```

---

## Files Created

```
research/codon-encoder/
├── benchmarks/
│   ├── padic_encoder_benchmark.py    # Standardized benchmark suite
│   ├── test_ordered_indices.py       # AA ordering experiments
│   └── test_enhanced_features.py     # Feature combination tests
├── results/
│   ├── benchmarks/
│   │   └── benchmark_results.json    # Full benchmark results
│   ├── ordered_indices/
│   │   └── ordering_comparison.json  # Ordering experiments
│   ├── enhanced_features/
│   │   └── feature_combinations.json # Feature combo results
│   └── PADIC_ENCODER_FINDINGS.md     # This document

src/encoders/
├── generalized_padic_encoder.py      # Any-prime p-adic encoder
└── segment_codon_encoder.py          # Long sequence support
```

---

## Comparison with Codon-Based Approach

| Approach | Spearman | Use Case |
|----------|----------|----------|
| Codon-based v2 | 0.81 | When codon info available |
| **AA + P-adic combined** | **0.40** | **Proteomics only** |
| Physicochemical baseline | 0.37 | Simple baseline |
| P-adic alone | 0.22 | Not recommended |

**Verdict:** When codon information is available, use codon-based approach (2x better). When only AA sequence is available, use combined features (+8.5% improvement).

---

## Future Work

1. **Multi-property ordering:** Test orderings that combine multiple properties
2. **Learned ordering:** Let the model learn optimal AA ordering
3. **Segment-based for long sequences:** Apply to full protein sequences
4. **Transfer learning:** Pre-train on large proteomics datasets

---

## Conclusion

The generalized p-adic encoder on amino acid indices provides value for proteomics-only scenarios when:
1. Amino acids are ordered by hydrophobicity
2. P-adic features are combined with physicochemical features
3. Explicit valuation features are included

This achieves **+8.5% improvement** over the physicochemical baseline, making it a useful approach when codon information is unavailable.
