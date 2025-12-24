# Training Diagnostic Report - 2025-12-14

**Doc-Type:** Diagnostic Analysis · Version 1.0 · Updated 2025-12-14

---

## Summary

300-epoch training diverged. Best performance at **epoch 8**, then monotonic degradation. Root cause: **homeostatic curvature runaway** due to misconfigured KL target.

---

## Key Observations

### Best Performance Window: Epochs 0-8

| Epoch | Loss | Curvature A | Curvature B | Sigma | Correlation |
|-------|------|-------------|-------------|-------|-------------|
| 0 | 14.55 | 2.008 | 2.008 | 1.000 | **0.703** |
| 8 | **3.95** | 2.240 | 2.318 | 0.995 | 0.703 |
| 28 | 8.51 | 3.135 | **4.000** (maxed) | 1.000 | cached |
| 299 | 20.25 | 4.000 | 4.000 | 0.660 | **0.515** |

### Critical Finding: Correlation Best at Epoch 0

The **untrained model** produced better 3-adic correlation (0.703) than any trained epoch. Training **destroyed** the initial structure rather than improving it.

---

## Root Cause Analysis

### Bug 1: Homeostatic KL Target Misconfiguration

**Location**: `src/losses/hyperbolic_prior.py:309`

```python
kl_target: float = 1.0  # DEFAULT
```

**Problem**: Actual hyperbolic KL runs at **80-90 nats**, but target is 1.0.

```python
kl_error = self.kl_ema - self.kl_target  # 80 - 1 = 79 (always huge positive)
curvature_delta = self.adaptation_rate * kl_error * 0.1  # Always positive
```

**Result**: Curvature monotonically increases until hitting max (4.0).

### Bug 2: No Curvature Decrease Path

The homeostatic logic assumes KL will sometimes be *below* target. With Hyperbolic KL at 80-90, this never happens. Curvature only increases, never decreases.

### Bug 3: Coverage Never Improves

Coverage stuck at 4.9-5.0% for all 300 epochs. The exploration mechanisms are not activating:
- Ranking weight stays at 0.109 (low)
- Curriculum tau stays at 0.0 (never advances)
- No novelty/curiosity signal

---

## Optimal Conditions (Inferred from Epoch 8)

| Parameter | Optimal Value | Final Value | Issue |
|-----------|---------------|-------------|-------|
| Curvature | 2.2-2.3 | 4.0 | Maxed out |
| Sigma | 0.995-1.0 | 0.66 | Drifted down |
| Loss | 3.95 | 20.25 | Diverged |
| KL (actual) | ~80-90 | ~7 | Collapsed late |
| Correlation | 0.703 | 0.515 | Degraded |

**Insight**: The system performs best with:
- **Moderate curvature**: ~2.2 (not the max of 4.0)
- **Sigma near 1.0**: Balanced exploration/exploitation
- **Early epochs**: Before homeostatic runaway

---

## Architecture Issues

### 1. Homeostatic Adaptation is Unstable

The homeostatic mechanism lacks negative feedback. Once curvature starts increasing, nothing stops it.

**Fix options**:
- Set `kl_target` to realistic value (~50-80)
- Add curvature decay term
- Use bidirectional adaptation with momentum

### 2. Correlation Degrades During Training

The initial random weights produce better structure than training. This suggests:
- The loss function doesn't optimize for 3-adic structure
- Training destroys initialization structure
- Need correlation-aware loss term

### 3. Coverage is Disconnected from Training

Coverage plateau detection exists but doesn't trigger any response. The system can't escape the 5% local minimum.

**Fix options**:
- Coverage-based temperature boost
- Curiosity reward for visiting new regions
- Force exploration when coverage stalls

---

## Recommended Fixes (Priority Order)

### P0: Fix Homeostatic KL Target

```yaml
# configs/ternary_v5_10.yaml
hyperbolic_v10:
  prior:
    kl_target: 50.0  # Was 1.0, realistic target
    curvature_decay: 0.001  # New: prevent runaway
```

### P1: Add Curvature Bounds Enforcement

```python
# After curvature update, enforce stay within "good" range
optimal_curvature_range = (2.0, 2.5)  # Based on epoch 8
if self.adaptive_curvature > optimal_curvature_range[1]:
    self.adaptive_curvature *= 0.99  # Soft decay back
```

### P2: Correlation-Aware Early Stopping

Stop training if correlation degrades from best:
```python
if current_correlation < best_correlation - 0.05:
    # Revert to best checkpoint or reduce learning rate
```

### P3: Coverage-Triggered Exploration

```python
if coverage_delta < 0.1 for N epochs:
    temperature *= 1.1  # Boost exploration
    ranking_weight *= 0.9  # Reduce structure constraint
```

---

## Experiment Suggestions

### Experiment 1: Fixed Curvature

Disable homeostatic curvature adaptation, fix at 2.2:
```yaml
curvature: 2.2
curvature_min: 2.2
curvature_max: 2.2
```

### Experiment 2: Shorter Training

Train only 10-15 epochs, stop before divergence.

### Experiment 3: Correlation-Weighted Loss

Add term that directly optimizes for 3-adic correlation:
```python
loss = base_loss - 0.1 * ranking_correlation
```

---

## Files to Modify

| File | Change |
|------|--------|
| `configs/ternary_v5_10.yaml` | kl_target: 50.0 |
| `src/losses/hyperbolic_prior.py` | Add curvature decay, fix adaptation logic |
| `src/training/hyperbolic_trainer.py` | Add correlation-aware early stopping |

---

## Conclusion

The architecture has potential (epoch 8 showed loss can drop to 3.95), but homeostatic adaptation is unstable. The most impactful fix is correcting the KL target from 1.0 to ~50.0, which would prevent curvature runaway. Secondary fixes should address coverage stagnation and correlation degradation.

**Key insight**: The untrained model had better correlation than the trained model. This means our loss function is optimizing for the *wrong thing*—reconstruction accuracy instead of structural preservation.
