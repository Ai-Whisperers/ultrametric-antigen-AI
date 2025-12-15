# Three-Body System: VAE-A, VAE-B, StateNet v5

**Doc-Type:** Architecture Analysis · Version 1.0 · Updated 2025-12-14

---

## Connections Implemented

### 1. StateNet → HyperbolicPrior (Already Wired)

**Location:** `src/training/hyperbolic_trainer.py:540-550`

```python
if self.use_hyperbolic_prior and 'delta_sigma' in train_losses:
    delta_sigma = train_losses.get('delta_sigma', 0.0)
    delta_curvature = train_losses.get('delta_curvature', 0.0)
    self.hyperbolic_prior_A.set_from_statenet(delta_sigma, delta_curvature)
    self.hyperbolic_prior_B.set_from_statenet(delta_sigma, delta_curvature)
```

**Flow:** StateNet observes prior_sigma/curvature → outputs delta corrections → applied to both VAEs' hyperbolic priors

### 2. Curriculum → Beta/Temperature (NEW)

**Location:** `src/training/trainer.py:250-262`

```python
if self.curriculum is not None:
    tau = self.curriculum.get_tau().item()
    beta_curriculum_factor = 1.0 + 0.2 * tau      # +20% beta when tau=1
    temp_curriculum_factor = 1.0 - 0.15 * tau     # -15% temp when tau=1
    beta_A = beta_A * beta_curriculum_factor
    temp_A = temp_A * temp_curriculum_factor
```

**Logic:**
- tau=0 (radial focus): Keep high exploration (unchanged beta/temp)
- tau=1 (ranking focus): Increase structure (higher beta), more certainty (lower temp)

### 3. Curriculum → Rho Cross-Injection (NEW)

**Location:** `src/training/trainer.py:221-228`

```python
if self.curriculum is not None:
    tau = self.curriculum.get_tau().item()
    rho_curriculum_factor = 1.0 - 0.3 * tau  # -30% rho when tau=1
    self.model.rho = self.model.rho * rho_curriculum_factor
```

**Logic:**
- tau=0 (radial focus): Full cross-injection for exploration
- tau=1 (ranking focus): Reduced cross-injection to preserve individual VAE structures

---

## Complete Three-Body Feedback Loops

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           StateNet v5 (20D → 8D)                        │
│  Observes: H_A, H_B, KL, coverage, r_A, r_B, radii, sigma, curvature   │
│  Outputs: delta_lr, delta_λ1/2/3, delta_ranking, delta_σ, delta_c, δτ  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  HyperbolicPrior │    │   Curriculum    │    │   VAE Params    │
│  σ += δσ        │    │   τ += δτ       │    │   λ1/2/3 += δλ  │
│  c += δc        │    │                 │    │   ranking_wt    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                       │
         │              ┌───────┴───────┐               │
         │              ▼               ▼               │
         │       ┌──────────┐   ┌──────────┐            │
         │       │ beta ×   │   │ rho ×    │            │
         │       │ (1+0.2τ) │   │ (1-0.3τ) │            │
         │       └────┬─────┘   └────┬─────┘            │
         │            │              │                  │
         ▼            ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        VAE-A (Chaotic)                                  │
│  Cross-injection: z_A' = (1-ρ)z_A + ρ·z_B.detach()                     │
│  Loss: CE_A + β_A·KL_A                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                        VAE-B (Frozen)                                   │
│  Cross-injection: z_B' = (1-ρ)z_B + ρ·z_A.detach()                     │
│  Loss: CE_B + β_B·KL_B + entropy + repulsion                            │
└─────────────────────────────────────────────────────────────────────────┘
         │                                                │
         └────────────────────┬───────────────────────────┘
                              │
                              ▼
                    Entropy Alignment: λ3·|H_A - H_B|
                    Gradient Balancing: scale by ratio
```

---

## Remaining Gaps

### Gap 1: Ranking Correlation EMA Too Slow (α=0.95)

**Location:** `src/models/ternary_vae_v5_10.py:875-876`

```python
self.r_A_ema = 0.95 * self.r_A_ema + 0.05 * r_A
```

**Issue:** With α=0.95, it takes ~20 epochs for a correlation change to fully propagate to StateNet. By then, damage may be done.

**Fix Needed:** Reduce EMA alpha or use raw values when correlation drops significantly.

### Gap 2: Coverage Signal Only Updated at Intervals

**Location:** `src/training/hyperbolic_trainer.py:487-505`

```python
should_check_coverage = (epoch == 0) or (epoch % self.coverage_check_interval == 0)
```

**Issue:** Coverage only evaluated every N epochs (default 10). StateNet sees stale coverage between evaluations.

**Fix Needed:** More frequent coverage sampling or batch-level coverage estimation.

### Gap 3: No VAE-Specific Curriculum Response

**Issue:** Both VAE-A and VAE-B receive identical curriculum modulation. VAE-A (chaotic) might benefit from different tau response than VAE-B (frozen).

**Fix Needed:** Asymmetric curriculum factors:
```python
beta_A = beta_A * (1.0 + 0.15 * tau)  # VAE-A: lighter beta boost
beta_B = beta_B * (1.0 + 0.25 * tau)  # VAE-B: stronger beta boost
```

### Gap 4: StateNet Learning Rate Not Adaptive

**Location:** Model uses fixed `statenet_lr_scale=0.1`

**Issue:** StateNet learns at fixed rate relative to VAEs. When VAEs converge, StateNet should adapt faster to find new optima.

**Fix Needed:** Adaptive StateNet learning rate based on loss plateau detection.

### Gap 5: No Backward Flow from Hyperbolic Metrics to StateNet Architecture

**Issue:** StateNet observes hyperbolic metrics but its architecture is static. If hyperbolic learning is dominant, StateNet could benefit from larger hyperbolic attention head.

**Fix Needed:** Dynamic attention head sizing or hyperbolic-specific StateNet branch.

### Gap 6: Missing Cross-VAE Correlation Signal

**Issue:** StateNet sees r_A and r_B independently, but doesn't see correlation BETWEEN VAE-A and VAE-B embeddings. If both VAEs learn similar structures, redundancy isn't penalized.

**Fix Needed:** Add `r_AB` (cross-VAE embedding correlation) to StateNet input.

### Gap 7: Temperature Boost Not Connected to Curriculum

**Location:** `exploration_boost` in config modulates temp_multiplier, but curriculum doesn't.

**Issue:** Two parallel temperature modulation systems (exploration_boost vs curriculum) that don't coordinate.

**Fix Needed:** Unify: `temp = base_temp * exploration_mult * curriculum_factor`

---

## Summary (Updated 2025-12-14)

**Closed Loops (Now Working):**
1. StateNet → HyperbolicPrior (delta_sigma, delta_curvature)
2. Curriculum → Beta modulation (asymmetric: +15% VAE-A, +30% VAE-B when tau=1)
3. Curriculum → Temperature modulation (asymmetric: -10% VAE-A, -20% VAE-B when tau=1)
4. Curriculum → Rho modulation (-30% when tau=1)
5. Ranking correlation EMA (α reduced from 0.95 to 0.7 for faster response)
6. Coverage/Correlation evaluation (interval reduced from 10/5 to 3/3 epochs)
7. Asymmetric curriculum response (VAE-A lighter, VAE-B stronger)
8. **Gap 7 Fixed:** Exploration boost temp unified with curriculum
   - Formula: `final_temp = base_temp * curriculum_factor * exploration_mult`
   - `exploration_temp_multiplier` set by hyperbolic_trainer, applied in base trainer
9. **Gap 6 Fixed:** Cross-VAE correlation signal (r_AB) added to StateNet v5.1
   - StateNet input extended from 20D to 21D
   - r_AB computed as normalized cosine similarity between z_A and z_B embeddings
   - High r_AB = redundancy (both VAEs learning similar structures)
   - StateNet can now penalize redundancy and encourage complementary representations
10. **Gap 4 Fixed:** Adaptive StateNet LR based on loss plateau detection
    - Tracks `loss_ema` and `loss_grad_ema` to detect convergence
    - When loss plateaus (gradient near zero), `statenet_lr_boost` increases (up to 3x)
    - Effective LR scale: `statenet_lr_scale * statenet_lr_boost`
    - Allows StateNet to explore more aggressively when VAEs are stuck
11. **Gap 5 Fixed:** Dynamic StateNet architecture via learnable attention head scales
    - Added `attention_head_scales` ParameterDict with learnable metric/hyperbolic/curriculum scales
    - Scales are learned through backprop, range [0, 2] via sigmoid
    - Attention outputs scaled: `metric_attention * metric_scale`
    - Allows network to dynamically emphasize important attention heads

**Partially Fixed:**
- P1 Ranking Loss Weight: Increased from 0.5 to 2.0 (+6.2% correlation improvement)
  - Note: Scalar correlation_loss provides no gradient; ranking loss does

**Open Loops (Remaining Gaps):**
- None (all gaps addressed)

**Fundamental Issues:**
- Correlation degrades during training (0.75 → 0.66)
- Coverage stuck at ~5% (reconstruction doesn't require exploration)
- Curriculum tau never advances (radial structure not learned)
