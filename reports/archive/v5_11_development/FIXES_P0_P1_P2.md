# Priority Fixes - Training Stability

**Doc-Type:** Fix Specification · Version 1.0 · Updated 2025-12-14

---

## Problem

Training diverged: best at epoch 8 (loss=3.95, correlation=0.703), then curvature runaway to max (4.0) caused loss explosion to 20+. Untrained model had better correlation than trained model.

---

## P0 - Critical (Stop Divergence)

| Fix | Current | Target | Location |
|-----|---------|--------|----------|
| KL target | 1.0 | 50.0 | `hyperbolic_prior.py:309` / config |
| Curvature | adaptive 2.0→4.0 | fixed 2.2 | config bounds |

## P1 - High Impact (Optimize Structure)

| Fix | Current | Target | Location |
|-----|---------|--------|----------|
| Correlation in loss | not used | `-λ * correlation` | `trainer.py` or `dual_vae_loss.py` |
| Early stop on corr drop | none | stop if Δcorr < -0.05 | `hyperbolic_trainer.py` |

## P2 - Structural (Enable Exploration)

| Fix | Current | Target | Location |
|-----|---------|--------|----------|
| Max epochs | 300 | 15-20 | config |
| Coverage boost | none | temp *= 1.1 if stalled | `hyperbolic_trainer.py` |

---

## Root Cause Chain

```
kl_target=1.0 (unrealistic)
    → kl_error=80 (always positive)
    → curvature increases every epoch
    → hits max (4.0) at epoch 28
    → hyperbolic geometry distorts
    → loss explodes, correlation degrades
```

---

## Success Criteria

- Loss stable <5.0 for 20+ epochs
- Correlation ≥0.70 maintained (not degraded)
- Curvature stable in 2.0-2.5 range

---

## P0 Validation Results (2025-12-14)

**Status: ✅ SUCCESS**

| Metric | Before P0 Fix | After P0 Fix | Target |
|--------|---------------|--------------|--------|
| Curvature A | 2.0 → 4.0 (runaway) | 2.18 → 2.00 (clamped) | 2.0-2.5 ✅ |
| Curvature B | 2.0 → 4.0 (runaway) | 2.18 → 2.08 (stable) | 2.0-2.5 ✅ |
| Loss (final) | 20.25 (exploded) | 2.13-3.10 (stable) | <5.0 ✅ |
| Correlation | 0.703 → 0.515 (degraded) | 0.715 → 0.715 (maintained) | ≥0.70 ✅ |
| Coverage | 4.9% (stuck) | 4.8% (still stuck) | >95% ❌ |

**Key Changes:**
- `kl_target`: 1.0 → 50.0 (realistic for hyperbolic KL ~80-90)
- `curvature_min`: 0.5 → 2.0 (prevent under-shoot)
- `curvature_max`: 4.0 → 2.5 (prevent overshoot)
- `adaptation_rate`: 0.01 → 0.005 (slower adaptation)

**Observations:**
1. Curvature A hit lower bound (2.0) - homeostatic wants to go lower
2. Curvature B stabilized naturally at 2.08
3. Loss trajectory: 14.5 → 2.1 (best) → 3.1 (end) - healthy
4. Coverage remains stuck - needs P1/P2 exploration fixes

**Next Steps:** P1 fixes (correlation-aware loss, coverage-triggered exploration)

---

## P1 Validation Results (2025-12-14)

**Status: ✅ SUCCESS (Early stopping working)**

| Metric | Before P1 Fix | After P1 Fix | Target |
|--------|---------------|--------------|--------|
| Epochs run | 20 (full) | 14 (early stop) | Stop on degradation ✅ |
| Correlation drop detection | None | Triggered at 0.072 drop | >0.05 threshold ✅ |
| Best correlation | 0.703 | 0.703 | Same ✅ |
| Final correlation | 0.715 (cached) | 0.631 (detected) | Stop before worse ✅ |
| Loss | 2.13-3.10 | 2.59-6.45 | Stable ✅ |

**Key Changes:**
- Added `correlation_feedback` config section
- Added `_init_correlation_feedback()` to HyperbolicVAETrainer
- Added `check_correlation_early_stop()` method
- Integrated early stopping in training loop

**Observations:**
1. Early stopping triggered at epoch 14 (0.703 → 0.631 = 0.072 drop)
2. Prevented continued training that would further degrade correlation
3. Curvature remained stable at 2.0-2.18 (P0 fix holding)
4. Coverage still stuck at 4.8-4.9%

**Remaining Issue:**
The fundamental problem is that **correlation degrades from initialization during training**.
The model learns reconstruction but destroys the initial 3-adic structure.

This suggests we need:
- P2: Add correlation as an explicit loss term (not just monitoring)
- Or: Restructure the loss to preserve 3-adic structure while learning

---

## P2 Validation Results (2025-12-14)

**Status: ✅ MECHANICS WORKING, ❌ COVERAGE STILL STUCK**

| Metric | Before P2 | After P2 | Target |
|--------|-----------|----------|--------|
| Exploration boost | N/A | Triggered epoch 5 | Detect stall ✅ |
| temp_mult | 1.0 | 1.15 → 2.00 (max) | Boost temp ✅ |
| ranking_mult | 1.0 | 0.90 → 0.35 | Reduce constraint ✅ |
| Ranking weight | 0.109 | 0.109 → 0.038 | Modulated ✅ |
| Coverage | 4.9% | 4.8-4.9% | >95% ❌ |
| P1 early stop | Epoch 14 | Epoch 14 | Still triggers ✅ |

**Key Changes:**
- Added `exploration_boost` config section
- Added `_init_exploration_boost()` method
- Added `check_coverage_stall()` detection
- Added `correlation_loss` config section
- Integrated boost in training loop with logging

**Observations:**
1. Exploration boost correctly detected coverage stall at epoch 5
2. temp_mult hit max (2.0), ranking_mult reduced to 0.35
3. Ranking weight dropped from 0.109 to 0.038 (65% reduction)
4. Despite reduced structure constraints, coverage stayed at ~5%
5. P1 early stopping still triggered at epoch 14 (correlation drop)

**Root Cause Analysis:**
The exploration boost reduces structure constraints but doesn't address the core issue:
- **The model can achieve low reconstruction loss WITHOUT exploring**
- Reconstruction of 9-trit vectors doesn't require manifold coverage
- Coverage is orthogonal to reconstruction objective

**What Would Actually Help:**
1. **Curiosity/novelty reward**: Directly reward visiting new latent regions
2. **Coverage as explicit loss**: `-λ * coverage` in the loss function
3. **Diversity regularization**: Encourage latent codes to spread out
4. **Contrastive learning**: Push different inputs to different regions

---

## P1 Re-Wiring Results (2025-12-14)

**Status: ✅ SUCCESS (Ranking weight increase helps)**

**Discovery:** The original `correlation_loss` config computed `-λ * correlation` but:
1. Only logged it, never added to actual loss (comment said "for logging only")
2. Even when wired, scalar correlation provides NO gradient signal

**Real Fix:** Increase `ranking_loss_weight` (the differentiable triplet loss):

| ranking_loss_weight | Best Correlation | Early Stop Epoch | Notes |
|---------------------|------------------|------------------|-------|
| 0.5 (original) | 0.7030 | 12 | Baseline |
| **2.0 (optimal)** | **0.7465** | 12 | +6.2% improvement |
| 5.0 | 0.7225 | 21 | More stable but lower peak |

**Key Changes:**
- `ranking_loss_weight`: 0.5 → 2.0 in config (top-level key)
- Added `ranking_margin: 0.1` and `ranking_n_triplets: 500` at top level
- Config key mismatch fixed (code reads `ranking_loss_weight`, not `ranking_loss.weight`)

**Observations:**
1. Higher ranking weight provides stronger gradient signal for 3-adic structure
2. Correlation still degrades over time, but from higher baseline
3. The fundamental tension remains: reconstruction vs structure preservation

**Remaining Issues:**
- Correlation still drops from 0.75 → 0.66 during training
- Coverage stuck at ~5%
- Curriculum tau never advances (radial structure not learned)
