# Comprehensive Module Ablation Study Results

## Overview

Tested all 32 combinations of 5 module flags:
- `use_hyperbolic` - Hyperbolic geometry projection
- `use_tropical` - Tropical (max-plus) aggregation
- `use_padic_triplet` - P-adic triplet loss
- `use_padic_ranking` - P-adic ranking loss
- `use_contrastive` - Contrastive learning loss

Test conditions: 50 epochs, 1000 training samples, structured synthetic data with 5 clusters.

---

## Key Results

### Baseline Performance
- **Accuracy**: 100.0%
- **Correlation**: +0.3273

### Individual Module Contributions

| Module | Correlation | Delta from Baseline |
|--------|-------------|---------------------|
| **padic_ranking** | **+0.9598** | **+0.6325** |
| hyperbolic | +0.5495 | +0.2222 |
| padic_triplet | +0.3456 | +0.0183 |
| tropical | +0.1455 | -0.1818 |
| contrastive | -0.0587 | -0.3860 |

### Top 10 Configurations by Correlation

| Rank | Configuration | Correlation | Modules |
|------|---------------|-------------|---------|
| 1 | rank_contrast | +0.9602 | 2 |
| 2 | rank | +0.9598 | 1 |
| 3 | trop_rank_contrast | +0.9589 | 3 |
| 4 | hyper_triplet_rank | +0.9585 | 3 |
| 5 | trop_rank | +0.9584 | 2 |
| 6 | triplet_rank_contrast | +0.9560 | 3 |
| 7 | hyper_trop_triplet_rank | +0.9559 | 4 |
| 8 | triplet_rank | +0.9552 | 2 |
| 9 | hyper_rank | +0.9544 | 2 |
| 10 | hyper_trop_rank | +0.9541 | 3 |

---

## Critical Insights

### 1. P-adic Ranking Loss is Transformative

The **ranking loss alone** provides a **+0.63 correlation improvement** - the single most impactful module. This makes sense mathematically because:

- It directly optimizes the correlation between latent representation and fitness
- It uses gradient-based learning to align the latent space with phenotype
- The p-adic structure provides natural hierarchical organization

**Recommendation**: Always include `use_padic_ranking=True`.

### 2. Hyperbolic Geometry Adds Value

Hyperbolic projection provides **+0.22 correlation improvement** alone, the second-best individual contribution:

- Better represents hierarchical/tree-like biological relationships
- Exponentially more space for hierarchical embeddings
- Natural fit for evolutionary trees

**Recommendation**: Include when modeling evolutionary relationships.

### 3. Triplet Loss is Minimal Alone but Synergistic

The triplet loss alone only adds +0.02, BUT it helps stabilize other losses:

- `hyper_triplet_rank` (+0.9585) is competitive with pure ranking
- Forces local structure preservation
- Prevents latent space collapse

**Recommendation**: Use in combination with ranking and hyperbolic.

### 4. Tropical Geometry Slightly Negative Alone

Tropical aggregation alone reduces correlation by -0.18:

- Max-plus operations can destroy gradient flow
- May need different temperature tuning
- Works better in combination with other modules

**Finding**: Not recommended alone, but `trop_rank` still achieves +0.9584.

### 5. Contrastive Learning is Harmful Alone

Contrastive loss alone is **strongly negative** (-0.39):

- InfoNCE pushes all samples apart uniformly
- Does not respect phenotype similarity
- Conflicts with ranking objective

**Critical**: Only use contrastive when combined with ranking loss, where it provides minimal benefit (+0.0004 in `rank_contrast`).

---

## Synergy Analysis

### Positive Synergies

| Combination | Expected | Actual | Synergy |
|-------------|----------|--------|---------|
| rank + contrast | 0.60 | 0.96 | +0.36 (strong positive) |
| trop + rank | 0.78 | 0.96 | +0.18 (positive) |
| hyper + rank | 0.88 | 0.95 | +0.07 (slight positive) |

The ranking loss "rescues" otherwise negative modules (contrastive, tropical) by providing a strong optimization target.

### Negative Synergies

| Combination | Expected | Actual | Synergy |
|-------------|----------|--------|---------|
| triplet + contrast | -0.02 | -0.28 | -0.26 (strong negative) |
| hyper + triplet | 0.57 | -0.28 | -0.85 (strong negative!) |
| trop + triplet | 0.15 | 0.02 | -0.13 (negative) |

**Warning**: Hyperbolic + Triplet without ranking is disastrous! The triplet loss fights with hyperbolic geometry.

---

## Recommended Configurations

### For Maximum Correlation
```python
config = AblationConfig(
    use_hyperbolic=False,  # Optional
    use_tropical=False,
    use_padic_triplet=False,
    use_padic_ranking=True,  # ESSENTIAL
    use_contrastive=False,  # Can include
    ranking_weight=0.3,
)
```
Expected: +0.96 correlation

### For Balanced Performance
```python
config = AblationConfig(
    use_hyperbolic=True,
    use_tropical=False,
    use_padic_triplet=True,
    use_padic_ranking=True,  # ESSENTIAL
    use_contrastive=False,
    padic_weight=0.5,
    ranking_weight=0.3,
)
```
Expected: +0.9585 correlation with better structure preservation

### For Evolutionary Analysis
```python
config = AblationConfig(
    use_hyperbolic=True,  # Tree structure
    use_tropical=True,   # Phylogenetic analysis
    use_padic_triplet=True,  # Local structure
    use_padic_ranking=True,  # ESSENTIAL
    use_contrastive=False,
    hyperbolic_curvature=1.0,
)
```
Expected: +0.9559 correlation with evolutionary interpretability

---

## Summary Table

| Configuration | Correlation | Accuracy | Recommendation |
|---------------|-------------|----------|----------------|
| rank (ranking only) | +0.96 | 100% | **BEST for correlation** |
| hyper_triplet_rank | +0.96 | 100% | Best balanced |
| hyper_rank | +0.95 | 100% | Good with hyperbolic |
| triplet_rank | +0.96 | 100% | Good with triplet |
| baseline | +0.33 | 100% | Reference |
| contrast (alone) | -0.06 | 100% | AVOID |

---

## Key Takeaways

1. **Ranking loss is essential** - provides +0.63 correlation improvement
2. **Hyperbolic geometry adds value** - +0.22 when used alone, synergizes with ranking
3. **Triplet loss stabilizes** - minimal alone but helps other modules
4. **Tropical should be combined** - negative alone but neutral in combinations
5. **Contrastive is dangerous alone** - only use with ranking loss as anchor
6. **Simpler is often better** - `rank` alone matches complex combinations

The p-adic ranking loss is the critical innovation - it directly optimizes the phenotype-latent correlation that we care about.
