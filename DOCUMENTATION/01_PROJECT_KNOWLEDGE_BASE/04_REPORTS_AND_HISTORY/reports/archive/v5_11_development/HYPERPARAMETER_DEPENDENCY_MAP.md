# Hyperparameter Dependency Map

**Doc-Type:** Architecture Analysis · Version 1.0 · Updated 2025-12-14

---

## Problem: Fragmented Parameters

Key parameters are defined in multiple places with different values, and some critical parameters are **not configurable** from YAML.

---

## Curvature - Used in 11 Files (NOT SHARED)

Each component has its **own** curvature that doesn't sync with others:

| Component | File | Default | Reads From Config |
|-----------|------|---------|-------------------|
| PAdicRankingLossHyperbolic | `padic_losses.py:527` | 1.0 | `ranking_hyperbolic.curvature` |
| HomeostaticHyperbolicPrior | `hyperbolic_prior.py:272` | 1.0 | `prior.curvature` |
| HomeostaticReconLoss | `hyperbolic_recon.py:49` | 1.0 | `recon.curvature` |
| HyperbolicCentroidLoss | `hyperbolic_recon.py:398` | 1.0 | `centroid.curvature` |
| Metrics evaluation | `metrics/hyperbolic.py:96` | 2.0 | `hyperbolic_trainer.curvature` |
| StateNet | `ternary_vae_v5_10.py` | 2.0 | model config |

**Issue**: When HomeostaticHyperbolicPrior adapts curvature from 2.0→4.0, the other components still use their original values. Geometry becomes inconsistent.

---

## kl_target - HARDCODED, NOT CONFIGURABLE

| Location | Value | Problem |
|----------|-------|---------|
| `hyperbolic_prior.py:282` | 1.0 | Default parameter, not in config |
| `hyperbolic_prior.py:309` | 1.0 | `self.kl_target = kl_target` |
| Config YAML | **MISSING** | Cannot override from config |

**P0 Fix**: Add `kl_target` to config and wire it through.

---

## Homeostatic Bounds - HARDCODED, NOT CONFIGURABLE

| Parameter | Location | Default | In Config? |
|-----------|----------|---------|------------|
| `sigma_min` | `hyperbolic_prior.py:277` | 0.3 | NO |
| `sigma_max` | `hyperbolic_prior.py:278` | 2.0 | NO |
| `curvature_min` | `hyperbolic_prior.py:278` | 0.5 | NO |
| `curvature_max` | `hyperbolic_prior.py:279` | 4.0 | NO |
| `adaptation_rate` | `hyperbolic_prior.py:280` | 0.01 | NO |
| `ema_alpha` | `hyperbolic_prior.py:281` | 0.1 | NO |
| `target_radius` | `hyperbolic_prior.py:283` | 0.5 | NO |

**P0 Fix**: Wire these through config.

---

## Config → Component Wiring Gap

In `hyperbolic_trainer.py:144-153`, HomeostaticHyperbolicPrior is created:

```python
self.hyperbolic_prior_A = HomeostaticHyperbolicPrior(
    latent_dim=prior_config.get('latent_dim', 16),
    curvature=prior_config.get('curvature', 2.0),
    prior_sigma=prior_config.get('prior_sigma', 1.0),
    max_norm=prior_config.get('max_norm', 0.95)
    # MISSING: kl_target, sigma_min, sigma_max, curvature_min, curvature_max,
    #          adaptation_rate, ema_alpha, target_radius
)
```

**Only 4 of 12 parameters are configurable!**

---

## Fix Locations

### P0 - Critical (Config + Wiring)

| Fix | Config Location | Code Location |
|-----|-----------------|---------------|
| Add `kl_target: 50.0` | `padic_losses.hyperbolic_v10.prior` | `hyperbolic_trainer.py:144` |
| Add `curvature_min: 2.0` | `padic_losses.hyperbolic_v10.prior` | `hyperbolic_trainer.py:144` |
| Add `curvature_max: 2.5` | `padic_losses.hyperbolic_v10.prior` | `hyperbolic_trainer.py:144` |

### P1 - High Impact

| Fix | Location |
|-----|----------|
| Share curvature across all components | Create `SharedGeometry` class |
| Add correlation to loss | `hyperbolic_trainer.py` or `dual_vae_loss.py` |

### P2 - Structural

| Fix | Location |
|-----|----------|
| Unify all curvature references | Global geometry config section |
| Coverage-triggered exploration | `hyperbolic_trainer.py` |

---

## Recommended Config Structure

```yaml
# SHARED GEOMETRY (new section)
geometry:
  curvature: 2.2              # Single source of truth
  curvature_min: 2.0          # P0: Lock to stable range
  curvature_max: 2.5          # P0: Prevent runaway
  max_norm: 0.95

# HOMEOSTATIC ADAPTATION (new section)
homeostatic:
  enabled: false              # P0: Disable until stable
  kl_target: 50.0             # P0: Realistic target
  sigma_min: 0.8
  sigma_max: 1.2
  adaptation_rate: 0.001      # Slower
  ema_alpha: 0.05             # More smoothing
  target_radius: 0.5
```

---

## Files Requiring Changes

| Priority | File | Changes |
|----------|------|---------|
| P0 | `configs/ternary_v5_10.yaml` | Add kl_target, curvature bounds |
| P0 | `src/training/hyperbolic_trainer.py:144-153` | Wire new config params |
| P0 | `src/losses/hyperbolic_prior.py` | Accept all params from config |
| P1 | `src/training/hyperbolic_trainer.py` | Add correlation to loss |
| P2 | All files using curvature | Reference shared geometry |

---

## Validation After Fix

```python
# All these should use the SAME curvature:
assert hyperbolic_prior_A.curvature == hyperbolic_prior_B.curvature
assert hyperbolic_prior_A.curvature == ranking_loss_hyp.curvature
assert hyperbolic_prior_A.curvature == centroid_loss.curvature
assert hyperbolic_prior_A.curvature == metrics_curvature
```

Currently this assertion would **FAIL** because each component has independent curvature.
