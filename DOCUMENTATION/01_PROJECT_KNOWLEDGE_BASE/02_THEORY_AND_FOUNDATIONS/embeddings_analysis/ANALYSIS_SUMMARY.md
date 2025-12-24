# V5.11 Embedding Space Analysis Summary

**Analysis Date:** 2024-12-23
**Variants Compared:** v5_11, v5_11_overnight, v5_11_structural

---

## Critical Finding: P-adic Hierarchy Inversion in Extended Training

### Per-Valuation Radius Analysis

| Valuation | v5_11 | v5_11_overnight | Expected |
|-----------|-------|-----------------|----------|
| v=0 | 0.857 | 0.830 | High (root) |
| v=1 | 0.821 | 0.774 | ↓ |
| v=2 | 0.752 | 0.742 | ↓ |
| v=3 | 0.714 | 0.700 | ↓ |
| v=4 | 0.657 | **0.826** | ↓ |
| v=5 | 0.695 | **0.853** | ↓ |
| v=6 | 0.603 | **0.831** | ↓ |
| v=7 | 0.774 | **0.905** | Low (leaves) |

### Key Observation

**v5_11** maintains correct p-adic hierarchy through v=6:
- Higher valuation → smaller radius → closer to origin
- This matches the expected tree structure in hyperbolic space

**v5_11_overnight** shows INVERTED hierarchy at v≥4:
- Extended training caused high-valuation points to move OUTWARD
- Points with v=7 have LARGEST radius (0.905) instead of smallest
- This is a fundamental structural failure

---

## Metric Comparison

| Metric | v5_11 | v5_11_overnight | Winner | Significance |
|--------|-------|-----------------|--------|--------------|
| Valuation-radial corr | **-0.424** | -0.295 | v5_11 | Hierarchy preservation |
| Distance-valuation corr | **0.306** | 0.230 | v5_11 | Structural consistency |
| Separation ratio | 1.156 | **1.173** | overnight | Cluster separation |
| Hyperbolic spread | 2.762 | **2.902** | overnight | Space utilization |
| Intrinsic dim (90%) | **8** | 10 | v5_11 | Compactness |

---

## Unique Strengths

### v5_11 (Production Baseline)
- **Best p-adic hierarchy**: Monotonic decrease from v=0 to v=6
- **Strongest correlations**: Both valuation-radial and distance-valuation
- **Most compact representation**: 8 dimensions capture 90% variance
- **Early convergence**: Best at epoch 9 (stable optimization landscape)

### v5_11_overnight (Extended Exploration)
- **Better cluster separation**: Higher separation ratio (1.173)
- **Wider hyperbolic spread**: Uses more of the Poincare ball
- **More boundary points**: 12.7% at r>0.9 vs 28.9% for v5_11
- **CAUTION**: Inverted hierarchy at high valuations

### v5_11_structural
- **Architecture difference**: Uses deeper network (hidden_dim=128)
- **Partial load failure**: Results unreliable due to weight mismatch
- **Needs re-training**: Cannot be fairly compared

---

## Recommendations

### For Production Use
**v5_11 is the better choice** despite lower epoch count because:
1. Maintains correct p-adic hierarchy (critical for downstream tasks)
2. Higher structural correlations
3. More compact representation

### For v5_11_overnight
The extended training appears to have overfit, causing:
- High-valuation points to drift outward
- Breaking the fundamental tree structure

Consider:
1. Early stopping at epoch ~280 (where v5_11 peaked)
2. Adding hierarchy-preserving regularization for extended training
3. Using v5_11_overnight embeddings only where separation (not hierarchy) matters

### For v5_11_structural
1. Fix architecture loading (match n_layers to checkpoint)
2. Re-run analysis with correct weights
3. May have unique properties worth preserving

---

## Data Preserved for Future Analysis

- `comparison_results.json`: Full metrics for all variants
- Per-valuation statistics for targeted analysis
- PCA explained variance ratios for dimensionality studies

---

## Next Steps

1. Analyze proj_B embeddings (currently only proj_A analyzed)
2. Visualize embedding spaces (t-SNE/UMAP colored by valuation)
3. Test downstream task performance (classification, retrieval)
4. Investigate what caused v5_11_overnight hierarchy inversion
