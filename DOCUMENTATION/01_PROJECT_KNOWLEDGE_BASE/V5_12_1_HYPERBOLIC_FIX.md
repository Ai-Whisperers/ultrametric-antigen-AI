# V5.12.1/V5.12.2 Hyperbolic Integration Fix

**Doc-Type:** Technical Implementation Guide - Version 2.0 - Updated 2025-12-29 - AI Whisperers

---

## Problem Statement

V5.12 achieves excellent hierarchy (-0.8288) but poor richness (0.001 vs target 0.005+). Root cause: hyperbolic geometry is applied inconsistently - it's a "side branch" rather than integral to the system.

**User's Conjecture (VERIFIED):** "The hyperbolic I/O is still not shared fully across all components of the ternary system."

---

## Critical Findings

### Original Architecture Gap (V5.12)

```
Input -> Encoder -> mu (Euclidean) ---------> Decoder -> Output
                        |
                   [Projection]
                        |
                   z_hyp (Poincare)
                        |
                   [Some losses use poincare_distance]
                   [Others use torch.norm - WRONG!]

Problem: Decoder NEVER sees z_hyp, AND losses use inconsistent geometry
```

### V5.12.2 Fixed Architecture

```
Input -> Encoder -> mu (Euclidean)
                        |
                   [Projection]
                        |
                   z_hyp (Poincare)
                        |
              +----+----+----+----+
              |    |    |    |    |
         Controller Losses Decoder Metrics
              |    |    |    |
           [ALL USE poincare_distance]

Full hyperbolic I/O across ALL three-body system components!
```

### Components: V5.12 vs V5.12.2

| Component | V5.12 (Broken) | V5.12.2 (Fixed) |
|-----------|----------------|-----------------|
| Decoder input | mu (Euclidean) | log_map_zero(z_hyp) |
| Controller radius | torch.norm(z_hyp) | poincare_distance(z_hyp, origin) |
| RichHierarchyLoss | z.norm(dim=-1) | poincare_distance(z, origin) |
| RadialHierarchyLoss | torch.norm() | poincare_distance() |
| GlobalRankLoss | torch.norm() | poincare_distance() |
| Metrics (train script) | .norm(dim=-1) | poincare_distance() |
| PAdicGeodesicLoss | poincare_distance() | Already correct |

---

## V5.12.2 Fix Checklist (ALL COMPLETED)

### Phase 1: Decoder Input (V5.12.1)

- [x] Modify `TernaryVAEV5_11_PartialFreeze.forward()` in `ternary_vae_optionc.py`
- [x] Change: `logits_A = self.decoder_A(mu_A)`
- [x] To: `logits_A = self.decoder_A(log_map_zero(z_A_hyp, c))`
- [x] Add import for `log_map_zero` from `src.geometry.poincare`
- [x] Unfreeze decoder (was frozen from v5.5 checkpoint)

### Phase 2: Metrics Computation (V5.12.1)

- [x] Update `compute_quick_metrics()` in training script
- [x] Use `poincare_distance(z_B, origin, c)` instead of `.norm(dim=-1)`

### Phase 3: Controller (V5.12.2 - CRITICAL FIX)

- [x] Fix `radius_A = torch.norm(z_A_hyp)` -> `poincare_distance(z_A_hyp, origin, c)`
- [x] Fix `radius_B = torch.norm(z_B_hyp)` -> `poincare_distance(z_B_hyp, origin, c)`
- [x] Import `poincare_distance` in ternary_vae_optionc.py

### Phase 4: Loss Functions (V5.12.2)

- [x] RichHierarchyLoss: Added curvature param, use poincare_distance
- [x] RadialHierarchyLoss: Added curvature param, use poincare_distance
- [x] GlobalRankLoss: Added curvature param, use poincare_distance
- [x] Training script passes curvature to all losses

---

## Mathematical Justification

### Why log_map_zero Works

```
z_hyp in Poincare ball B^16, ||z_hyp|| < 0.95

log_map_zero(z_hyp) = arctanh(sqrt(c) * ||z_hyp||) * direction
                    = arctanh(sqrt(c) * radius) * direction

arctanh stretches:
  [0, 0.5] -> [0, 0.55]
  [0, 0.9] -> [0, 1.47]
  [0, 0.95] -> [0, 1.83]
  [0, 0.99] -> [0, 2.65]

Result: Unbounded tangent space vector preserving direction
```

### Information Capacity

For 19,683 operations in 16D:
- 15 angular dimensions (S^15 sphere) - vast discriminative power
- 1 radial dimension - hierarchy encoding
- More than sufficient for reconstruction

### Gradient Flow

```
Before: Loss -> mu -> Encoder (z_hyp is dead branch for reconstruction)
After:  Loss -> log_map(z_hyp) -> z_hyp -> Projection -> mu -> Encoder
        Now hyperbolic structure receives gradient pressure
```

---

## Implementation Details

### Decoder Input Change

```python
# Before (ternary_vae_optionc.py:238)
logits_A = self.decoder_A(mu_A)

# After
from src.geometry.poincare import log_map_zero
c = self.hyperbolic_projection.get_curvature()
z_tangent = log_map_zero(z_A_hyp, c=c)
logits_A = self.decoder_A(z_tangent)
```

### Metrics Change

```python
# Before (train_v5_12.py:226-227)
all_radii_A.append(z_A.norm(dim=-1).cpu().numpy())
all_radii_B.append(z_B.norm(dim=-1).cpu().numpy())

# After
from src.geometry.poincare import poincare_distance
origin = torch.zeros_like(z_A)
c = model.hyperbolic_projection.get_curvature() if hasattr(model, 'hyperbolic_projection') else 1.0
all_radii_A.append(poincare_distance(z_A, origin, c).cpu().numpy())
all_radii_B.append(poincare_distance(z_B, origin, c).cpu().numpy())
```

---

## Expected Outcomes

### Metrics Targets

| Metric | V5.12 | V5.12.1 Target |
|--------|-------|----------------|
| Coverage | 100% | 100% (may drop initially during adaptation) |
| Hierarchy_B | -0.8288 | -0.80 to -0.8321 |
| Richness | 0.001 | >0.005 |
| r_v9 | 0.078 | 0.10-0.15 |

### Why Richness Should Improve

1. Decoder receives hyperbolic representation -> reconstruction pressure flows to z_hyp
2. Model cannot collapse to minimal shells without losing reconstruction
3. Gradient through log_map_zero maintains geometric structure
4. Consistent geometric framework across reconstruction and hierarchy

---

## Risk Mitigation

### Coverage Drop Risk

The frozen checkpoint decoder was trained on mu, not log_map(z_hyp). Mitigation:
- Allow longer training for adaptation
- Start with lower hierarchy weight to prioritize coverage
- Use warmup epochs before full hierarchy pressure

### Training Instability Risk

arctanh can produce large values near boundary. Mitigation:
- max_radius=0.95 keeps us away from singularity
- Gradient clipping already in place (max_grad_norm=1.0)
- Monitor for NaN/Inf during training

---

## Files to Modify

1. `src/models/ternary_vae_optionc.py` - Decoder input
2. `scripts/training/train_v5_12.py` -> copy to `train_v5_12_1.py` - Metrics
3. `configs/v5_12.yaml` -> copy to `configs/v5_12_1.yaml` - Version bump
4. `src/core/metrics.py` - ComprehensiveMetrics (optional, for consistency)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-29 | Initial v5.12.1 fix documentation |
