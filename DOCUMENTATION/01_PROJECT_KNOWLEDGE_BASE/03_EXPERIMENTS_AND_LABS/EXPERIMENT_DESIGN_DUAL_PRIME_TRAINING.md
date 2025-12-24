# Experiment Design: Dual-Prime (2-adic + 3-adic) Training

**Status:** Design Document - Requires Analysis Before Implementation
**Date:** 2025-12-16
**Prerequisite:** Review DISCOVERY_ARCHITECTURE_PRIME_CAPACITY.md

---

## Hypothesis

The current architecture (latent_dim=16) has **6 = 2×3** dimensions of "slack" that could encode both binary and ternary prime structure. Currently only 3-adic is activated because training only includes 3-adic loss.

**Prediction:** Adding 2-adic loss will activate the unused binary capacity without changing architecture.

---

## Current Training Configuration

```python
# V5.11.3 Structural (v1.1.0)
loss = (
    recon_loss
    + kl_weight * kl_loss
    + radial_weight * radial_hierarchy_loss_3adic
    + rank_weight * global_rank_loss_3adic
)
```

**Observed:**
- 3-adic exponent c ≈ 0.183 ≈ 1/6
- 2-adic exponent b ≈ 0 (capacity unused)

---

## Proposed Experiment

### Phase 1: 2-adic Valuation Function

```python
def v2_exact(n: int) -> int:
    """Compute exact 2-adic valuation."""
    if n == 0:
        return MAX_VAL  # Convention for zero
    v = 0
    while n % 2 == 0:
        n //= 2
        v += 1
    return v
```

### Phase 2: Dual-Prime Loss

```python
def dual_prime_radial_loss(z_hyp, indices, lambda_2=0.5):
    """
    Combined 2-adic and 3-adic radial hierarchy loss.

    Args:
        z_hyp: Hyperbolic embeddings [B, D]
        indices: Operation indices [B]
        lambda_2: Weight for 2-adic component (0=pure 3-adic, 1=equal weight)
    """
    radii = torch.norm(z_hyp, dim=-1)

    # 3-adic component (existing)
    v3 = compute_3adic_valuation(indices)
    loss_3 = radial_hierarchy_loss(radii, v3)

    # 2-adic component (new)
    v2 = compute_2adic_valuation(indices)
    loss_2 = radial_hierarchy_loss(radii, v2)

    # Combined loss
    return (1 - lambda_2) * loss_3 + lambda_2 * loss_2
```

### Phase 3: Training Sweep

| Experiment | lambda_2 | Expected Result |
|:-----------|:---------|:----------------|
| Baseline | 0.0 | c_3 ≈ 1/6, c_2 ≈ 0 (current) |
| Light 2-adic | 0.1 | c_3 ≈ 1/6, c_2 > 0 (small) |
| Balanced | 0.5 | c_3 ≈ c_2 ≈ 1/12? |
| Heavy 2-adic | 0.9 | c_3 < c_2 (role reversal) |

---

## Analysis Requirements (Before Implementation)

### 1. Theoretical Questions

- Does joint 2+3 optimization have a unique minimum?
- Can both hierarchies be satisfied simultaneously in 16D?
- What is the optimal lambda_2 for GUE emergence?

### 2. Capacity Analysis

The "slack" of 6 dimensions factors as 2×3. Questions:
- Do the 2 and 3 components use orthogonal subspaces?
- Is there interference between v2 and v3 gradients?
- Does the architecture naturally separate them?

### 3. Metric Considerations

Current metric is Euclidean in Poincaré ball. For dual-prime:
- Should we use product metric d = d_2 × d_3?
- Or sum metric d = d_2 + d_3?
- Or ultrametric max(d_2, d_3)?

---

## Success Criteria

### Primary

1. Both c_2 > 0.05 AND c_3 > 0.05 (both primes active)
2. Total R² for radius prediction ≥ 0.95 (maintains quality)
3. Ultrametric violations remain 0

### Secondary (GUE Test)

1. Graph Laplacian KS_GUE < KS_Poisson (shift toward GUE)
2. Eigenvalue spacing distribution closer to Wigner surmise
3. Pair correlation matches Montgomery conjecture

---

## Implementation Notes

### Option A: Sequential Training

```
1. Train Phase 1: lambda_2=0 (pure 3-adic) until convergence
2. Train Phase 2: lambda_2=0.5 (dual) from Phase 1 checkpoint
3. Compare Phase 1 vs Phase 2 spectra
```

### Option B: Curriculum Learning

```
1. Start: lambda_2=0
2. Gradually increase: lambda_2 → 0.5 over N epochs
3. Monitor for mode collapse or hierarchy degradation
```

### Option C: Multi-Head Architecture

```python
class DualPrimeProjection(nn.Module):
    def __init__(self):
        self.proj_2adic = RadialProjection(...)  # 2-adic head
        self.proj_3adic = RadialProjection(...)  # 3-adic head
        self.combiner = nn.Linear(32, 16)        # Merge to 16D
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|:-----|:-----------|:-----------|
| Gradient conflict (v2 vs v3) | Medium | Use separate projection heads |
| Loss of 3-adic quality | Medium | Monitor radial_corr throughout |
| No GUE emergence | High | This tests the hypothesis, negative result is informative |
| Capacity insufficient | Low | 6 dims should fit 2×3, but verify |

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|:------|:---------|:-------------|
| Theoretical analysis | 2-3 days | None |
| Implementation | 1 day | Analysis complete |
| Training sweep | 2-3 days | GPU availability |
| Spectral analysis | 1 day | Training complete |
| Documentation | 0.5 day | Analysis complete |

**Total:** ~1 week of focused work

---

## Related Documents

- `DISCOVERY_RADIAL_EXPONENT.md` - The 1/6 formula
- `DISCOVERY_ARCHITECTURE_PRIME_CAPACITY.md` - The 2×3 capacity hypothesis
- `riemann_hypothesis_sandbox/07_adelic_analysis.py` - Current multi-prime analysis
- `riemann_hypothesis_sandbox/09_binary_ternary_decomposition.py` - 2×3 decomposition tests

---

**Document Status:** Ready for review
**Next Action:** Theoretical analysis of gradient dynamics in dual-prime optimization
