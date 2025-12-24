## What TensorBoard can unlock in your pipeline (beyond basic logging)

### 1) Manifold observability (highest value)

You currently log scalars that **can be misleading**. What matters is observing the **space itself**.

• **3D latent embedding over time** (PCA / UMAP, animated by epoch)
Color by:
– 3-adic depth
– shared prefix
– VAE-A vs VAE-B
– reconstruction error

This directly validates whether the tree structure emerges or collapses.

• **Radial density in the Poincaré ball**
– radius histogram per epoch
– radial vs angular variance

Answers:
– do leaves move to the boundary?
– is there geometric saturation?
– does StateNet actually regulate dynamics?

• **Topological distortion metric**

```
distortion = d_latent(a,b) / d_3adic(a,b)
```

Log mean, variance, p95, p99 over epochs.
This measures **structural fidelity**, not just accuracy.

• **3-adic cluster stability**
For each depth k:
– centroid drift
– intra-cluster radius
– inter-cluster separation

Detects hierarchy leakage or collapse.

---

### 2) High-signal plots experts will respect

• **Phase-space trajectories**
Axes:
– entropy
– KL
Color:
– coverage

Shows system dynamics, phase transitions, stability.

• **Bifurcation plots**
Vary ρ, temperature, or λ₃ and log:
– final coverage
– distortion
– collapse regimes

Turns the model into a **dynamical system**, not a black box.

• **Dual-VAE tension metrics**
– grad_norm_A / grad_norm_B
– KL_A vs KL_B
– recon_A vs recon_B

Verifies real complementarity vs dominance.

---

### 3) Training-time control (not post-mortem)

• **Early-warning signals**
– latent variance spikes
– radial collapse
– gradient ratio inversion

Trigger automatic actions:
– partial freeze
– temperature bump
– soft reset

This enables **self-regulating training**.

• **Embedding trajectory replay**
Compare manifold evolution across runs.
This is **geometric debugging**.

---

### 4) Inference-time observability (often ignored)

• **Coverage vs sampling temperature**
– novelty
– invalidity rate

Provides a real control knob.

• **Local neighborhood probing**
– sample ε-balls in latent space
– decode
– measure functional diversity

Tests manifold smoothness and robustness.

• **Uncertainty as geometry**
– decoder variance
– bit-flip sensitivity per LUT position

Uncertainty becomes a **local geometric property**, not noise.

---

### 5) Priority order (minimal set, maximal insight)

1. Animated 3D latent embedding
2. Topological distortion vs epoch
3. Radial density / curvature stats
4. Phase-space trajectories
5. Inference neighborhood probes

---

**Bottom line:**
TensorBoard can become a **geometric microscope** that shows structure emerging, not just losses decreasing.

---

## Appendix: System Architecture Reference (for Feasibility Verification)

This section documents the actual codebase architecture so visualization claims above can be validated against the real implementation.

---

### A) Canonical Model: `DualNeuralVAEV5_10`

**Location:** `src/models/ternary_vae_v5_10.py`
**Version:** v5.10 (Pure Hyperbolic Geometry)
**Total Parameters:** ~168,770

#### A.1) Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DualNeuralVAEV5_10                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: x ∈ {-1, 0, 1}^9  (ternary operation)                         │
│                                                                         │
│   ┌─────────────────────┐      ┌─────────────────────┐                 │
│   │   TernaryEncoderA   │      │   TernaryEncoderB   │                 │
│   │   [9]→256→128→64→32 │      │   [9]→256→128→64→32 │                 │
│   │         ↓           │      │         ↓           │                 │
│   │   [μ_A, logvar_A]   │      │   [μ_B, logvar_B]   │                 │
│   └──────────┬──────────┘      └──────────┬──────────┘                 │
│              │                            │                             │
│              ↓                            ↓                             │
│   ┌──────────────────────────────────────────────────┐                 │
│   │        Reparameterization: z = μ + σ·ε           │                 │
│   │        z_A ∈ ℝ^16,  z_B ∈ ℝ^16                   │                 │
│   └──────────────────────────────────────────────────┘                 │
│              │                            │                             │
│              ↓                            ↓                             │
│   ┌──────────────────────────────────────────────────┐                 │
│   │           Cross-Injection (ρ scheduled)          │                 │
│   │   z_A' = (1-ρ)·z_A + ρ·z_B.detach()             │                 │
│   │   z_B' = (1-ρ)·z_B + ρ·z_A.detach()             │                 │
│   └──────────────────────────────────────────────────┘                 │
│              │                            │                             │
│              ↓                            ↓                             │
│   ┌─────────────────────┐      ┌─────────────────────┐                 │
│   │   TernaryDecoderA   │      │   TernaryDecoderB   │                 │
│   │   [16]→32→64→[27]   │      │ [16]→128→ResBlock   │                 │
│   │   (simple, chaotic) │      │   →128→[27]         │                 │
│   │                     │      │   (residual, frozen)│                 │
│   └──────────┬──────────┘      └──────────┬──────────┘                 │
│              │                            │                             │
│              ↓                            ↓                             │
│   logits_A ∈ ℝ^(9×3)           logits_B ∈ ℝ^(9×3)                      │
│   (after reshape & /temp)      (after reshape & /temp)                 │
│                                                                         │
│   ┌─────────────────────────────────────────────────────┐              │
│   │                   StateNetV4                         │              │
│   │   18D state → [LN→Tanh→64→ReLU→12] → [64→Tanh→7D]  │              │
│   │   + metric_attention (10→18)                         │              │
│   │   + hyperbolic_attention (6→18)                      │              │
│   └─────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

#### A.2) Key Model Outputs (Available for Visualization)

```python
outputs = model.forward(x, temp_A, temp_B, beta_A, beta_B)

# Directly accessible tensors:
outputs['logits_A']      # (batch, 9, 3) - decoder output A
outputs['logits_B']      # (batch, 9, 3) - decoder output B
outputs['mu_A']          # (batch, 16) - latent mean A
outputs['mu_B']          # (batch, 16) - latent mean B
outputs['logvar_A']      # (batch, 16) - latent log-variance A
outputs['logvar_B']      # (batch, 16) - latent log-variance B
outputs['z_A']           # (batch, 16) - sampled latent A
outputs['z_B']           # (batch, 16) - sampled latent B
outputs['H_A']           # scalar - entropy of VAE-A output
outputs['H_B']           # scalar - entropy of VAE-B output
outputs['rho']           # scalar - current permeability
outputs['lambda3']       # scalar - entropy alignment weight

# Model buffer state (registered buffers):
model.grad_norm_A_ema    # EMA of gradient norm A
model.grad_norm_B_ema    # EMA of gradient norm B
model.r_A_ema            # EMA of ranking correlation A
model.r_B_ema            # EMA of ranking correlation B
model.mean_radius_A      # Mean Poincaré ball radius A
model.mean_radius_B      # Mean Poincaré ball radius B
model.prior_sigma        # Wrapped normal spread
model.curvature          # Poincaré ball curvature
```

#### A.3) StateNetV4 Input Vector (18D)

```
idx  name                 range        purpose
────────────────────────────────────────────────────────
0    H_A                  [0,3]        VAE-A output entropy
1    H_B                  [0,3]        VAE-B output entropy
2    kl_A                 [0,∞)        VAE-A KL divergence
3    kl_B                 [0,∞)        VAE-B KL divergence
4    grad_ratio           [0,∞)        ||∇_A|| / ||∇_B||
5    rho                  [0.1,0.7]    cross-injection permeability
6    lambda1              [0,1]        repulsion weight
7    lambda2              [0,1]        entropy weight
8    lambda3              [0,1]        entropy alignment weight
9    coverage_A_norm      [0,1]        unique ops A / 19683
10   coverage_B_norm      [0,1]        unique ops B / 19683
11   missing_ops_norm     [0,1]        missing ops / 19683
12   r_A                  [-1,1]       ranking correlation A
13   r_B                  [-1,1]       ranking correlation B
14   mean_radius_A        [0,0.95]     avg ||z_A|| in Poincaré ball
15   mean_radius_B        [0,0.95]     avg ||z_B|| in Poincaré ball
16   prior_sigma          [0.3,2.0]    wrapped normal spread
17   curvature            [0.5,4.0]    Poincaré curvature
```

#### A.4) StateNetV4 Output Corrections (7D)

```
idx  name                 scale                 effect
────────────────────────────────────────────────────────
0    delta_lr             ×0.1                  LR multiplier
1    delta_lambda1        ×0.02                 repulsion adjustment
2    delta_lambda2        ×0.02                 entropy adjustment
3    delta_lambda3        ×0.02                 alignment adjustment
4    delta_ranking_weight ×0.3                  ranking loss weight
5    delta_sigma          ×0.05                 prior sigma adjustment
6    delta_curvature      ×0.02                 curvature adjustment
```

---

### B) Loss Components

**Location:** `src/losses/`

#### B.1) Loss Module Hierarchy

```
DualVAELoss (Joint Aggregator)
├── ReconstructionLoss          # CE(logits, x)
├── KLDivergenceLoss            # KL(q||p) with free bits
├── EntropyRegularization       # -Σ p·log(p) for output diversity
├── RepulsionLoss               # RBF-based latent spread
├── PAdicMetricLoss*            # MSE(||z_i-z_j||, d_3(i,j))
├── PAdicRankingLoss*           # Triplet margin loss
├── PAdicRankingLossV2*         # Hard negative mining
├── PAdicRankingLossHyperbolic* # Poincaré distance ranking
├── PAdicNormLoss*              # MSB/LSB hierarchy
├── HomeostaticHyperbolicPrior* # Wrapped normal on Poincaré
├── HomeostaticReconLoss*       # Radius-weighted reconstruction
└── HyperbolicCentroidLoss*     # Fréchet mean clustering
    (* = optional, config-driven)
```

#### B.2) Mathematical Formulations

**Reconstruction Loss:**
```
L_recon = (1/N) Σᵢ CE(softmax(logits_i), x_i)
        = CrossEntropy over 9 positions × 3 classes
```

**KL Divergence (with free bits):**
```
L_kl = (1/N) Σᵢ Σ_d max(λ_free, -0.5·(1 + logvar - μ² - exp(logvar)))
```

**Entropy Regularization:**
```
H = -Σ_o p(o)·log(p(o) + ε)
where p(o) = softmax(mean_batch(logits))
L_entropy = -H  (maximize entropy)
```

**Repulsion Loss (RBF kernel):**
```
L_repulsion = (1/|pairs|) Σ_{i≠j} exp(-||z_i - z_j||² / σ²)
```

**P-adic Metric Loss:**
```
d_3(i,j) = 3^(-v_3(|i-j|))
where v_3(n) = max{k : 3^k | n}

L_padic = MSE(||z_i - z_j||₂, C·d_3(i,j))
```

**P-adic Ranking Loss (Triplet):**
```
For triplets (a,b,c) where d_3(a,b) < d_3(a,c):
L_ranking = max(0, ||z_a - z_b|| - ||z_a - z_c|| + margin)
```

**Hyperbolic Prior KL:**
```
d_p(x,y) = (1/√c)·arcosh(1 + 2c·||x-y||²/((1-c||x||²)(1-c||y||²)))

L_prior = KL(q(z|x) || Wrapped_Normal(0, σ))
        computed via parallel transport on Poincaré manifold
```

**Conformal Factor (Poincaré):**
```
λ_x = 2/(1 - c·||x||²)
```

---

### C) Training Infrastructure

**Location:** `src/training/`

#### C.1) Trainer Hierarchy

```
TernaryVAETrainer (Base)
    └── train_epoch(train_loader, epoch, total_epochs)
        ├── forward pass
        ├── loss computation
        ├── backward pass
        ├── optimizer step
        └── monitoring

HyperbolicVAETrainer (Wrapper)
    └── train_epoch(...)
        ├── base_trainer.train_epoch(...)
        ├── hyperbolic loss additions
        ├── continuous feedback loop
        └── homeostatic adaptation
```

#### C.2) Scheduler Components

**TemperatureScheduler:**
```python
temp(t) = temp_start + (temp_end - temp_start) × (t / T)
        + [optional cyclic modulation]
        + [Phase 4 boost if t > phase_4_start]
```

**BetaScheduler:**
```python
β(t) = {
  0                                           if t < warmup_epochs
  β_start + (β_end - β_start)·(t-warmup)/T    otherwise
}
```

**4-Phase Training Schedule:**
```
Phase 1 (0-40):     ρ = ρ_min (0.1)       Expansion
Phase 2 (40-120):   ρ ∈ [0.1, 0.3]        Consolidation
Phase 3 (120-250):  ρ ∈ [0.3, 0.7]*       Resonant coupling
Phase 4 (250+):     ρ = ρ_max (0.7)       Ultra-exploration

* Phase 3→4 requires gradient balance: 0.8 < grad_ratio < 1.2
```

#### C.3) TrainingMonitor Capabilities

**Currently logged (batch-level):**
```
loss_total, loss_recon_A, loss_recon_B
loss_kl_A, loss_kl_B
loss_entropy_B, loss_repulsion_B
loss_entropy_align
loss_padic_* (if enabled)
loss_hyperbolic_* (if v5.10)
grad_ratio
learning_rate
```

**Currently logged (epoch-level):**
```
coverage_A, coverage_B, coverage_pct
missing_operations
H_A, H_B (entropies)
rho, lambda1, lambda2, lambda3
temp_A, temp_B, beta_A, beta_B
r_A_hyp, r_B_hyp (ranking correlations)
mean_radius_A, mean_radius_B
prior_sigma, curvature
```

---

### D) Data Space

**Location:** `src/data/`

#### D.1) Operation Space Definition

```
Domain:     All f: {0,1,2}³ → {0,1,2}  (ternary operations)
Size:       3^9 = 19,683 unique operations
Encoding:   x ∈ {-1, 0, 1}^9  (offset by -1 from {0,1,2})

Index mapping:
  index i ∈ [0, 19682] → base-3 digits → 9D vector
  Example: i=0 → (0,0,0,0,0,0,0,0,0) → (-1,-1,-1,-1,-1,-1,-1,-1,-1)
```

#### D.2) 3-adic Metric

```
v_3(n) = max{k ∈ ℕ : 3^k divides n}

d_3(i,j) = 3^(-v_3(|i-j|))

Properties:
  - d_3(i,i) = 0
  - d_3 is an ultrametric: d(a,c) ≤ max(d(a,b), d(b,c))
  - Creates hierarchical tree structure
  - Depth k subtree: operations sharing k least significant trits
```

**Example 3-adic distances:**
```
|i-j| = 1:    v_3 = 0,  d_3 = 1.000
|i-j| = 3:    v_3 = 1,  d_3 = 0.333
|i-j| = 9:    v_3 = 2,  d_3 = 0.111
|i-j| = 27:   v_3 = 3,  d_3 = 0.037
|i-j| = 81:   v_3 = 4,  d_3 = 0.012
```

---

### E) Hyperbolic Geometry Implementation

**Location:** `src/losses/hyperbolic_prior.py`, `src/metrics/hyperbolic.py`

#### E.1) Poincaré Ball Model

```
Ball:       B^n_c = {x ∈ ℝ^n : c·||x||² < 1}
Curvature:  c > 0 (default: 2.0)
Max norm:   0.95 (prevents numerical instability at boundary)

Projection: z_proj = z / (1 + ||z||) × max_norm
```

#### E.2) Geodesic Distance

```python
def poincare_distance(x, y, c=2.0):
    norm_x_sq = (x**2).sum(-1)
    norm_y_sq = (y**2).sum(-1)
    diff_sq = ((x - y)**2).sum(-1)

    numerator = 2 * c * diff_sq
    denominator = (1 - c*norm_x_sq) * (1 - c*norm_y_sq)

    return (1/sqrt(c)) * arcosh(1 + numerator/denominator)
```

#### E.3) Exponential Map (from origin)

```
exp_0(v) = tanh(√c·||v||) × v / (√c·||v||)
```

---

### F) Metrics Implementation

**Location:** `src/metrics/hyperbolic.py`

#### F.1) Ranking Correlation Evaluation

```python
def compute_ranking_correlation_hyperbolic(
    model, device,
    n_samples=5000,
    n_triplets=1000,
    curvature=2.0,
    max_norm=0.95
) -> Tuple[float, float, float, float, float, float]:
    """
    Returns:
        corr_A_hyp:    VAE-A hyperbolic ranking correlation
        corr_B_hyp:    VAE-B hyperbolic ranking correlation
        corr_A_euc:    VAE-A Euclidean ranking correlation
        corr_B_euc:    VAE-B Euclidean ranking correlation
        mean_radius_A: Average ||z_A|| in Poincaré ball
        mean_radius_B: Average ||z_B|| in Poincaré ball
    """
```

**Algorithm:**
1. Sample n_samples random operations
2. Forward through model → z_A, z_B
3. Project to Poincaré: `z_hyp = z / (1 + ||z||) × max_norm`
4. Sample n_triplets (anchor, positive, negative)
5. For each triplet:
   - Ground truth: `is_closer_3adic = d_3(a,b) < d_3(a,c)`
   - Prediction: `is_closer_hyp = d_poincare(z_a,z_b) < d_poincare(z_a,z_c)`
   - Agreement: `is_closer_3adic == is_closer_hyp`
6. correlation = agreements / n_triplets

---

### G) Configuration Schema

**Location:** `src/training/config_schema.py`

#### G.1) Key Hyperparameters

| Parameter | Location | Default | Range | Purpose |
|-----------|----------|---------|-------|---------|
| `latent_dim` | ModelConfig | 16 | [8,64] | Latent code dimension |
| `rho_min` | ModelConfig | 0.1 | [0,1] | Min permeability |
| `rho_max` | ModelConfig | 0.7 | [0,1] | Max permeability |
| `lambda3_base` | ModelConfig | 0.3 | [0,1] | Entropy alignment base |
| `lambda3_amplitude` | ModelConfig | 0.15 | [0,0.5] | Entropy oscillation |
| `eps_kl` | ModelConfig | 0.0005 | [0,0.01] | KL collapse threshold |
| `use_statenet` | ModelConfig | True | bool | Enable StateNet v4 |
| `beta_start/end` | VAEConfig | 0.3/0.8 | [0,1] | KL weight schedule |
| `temp_start/end` | VAEConfig | 1.0/0.3 | [0.1,2] | Temperature annealing |
| `curvature` | HyperbolicConfig | 2.0 | [0.5,4] | Poincaré curvature |
| `prior_sigma` | HyperbolicConfig | 1.0 | [0.3,2] | Wrapped normal spread |
| `max_norm` | HyperbolicConfig | 0.95 | [0.8,0.99] | Ball radius constraint |

---

### H) Feasibility Analysis for Proposed Visualizations

Based on the architecture above, here is the feasibility assessment for each visualization in sections 1-5:

#### H.1) Manifold Observability (Section 1)

| Visualization | Feasibility | Data Source | Notes |
|---------------|-------------|-------------|-------|
| 3D latent embedding (PCA/UMAP) | ✅ FEASIBLE | `outputs['z_A']`, `outputs['z_B']` (batch, 16) | Must reduce 16D→3D; can color by index (→3-adic depth via v_3(i)) |
| Color by 3-adic depth | ✅ FEASIBLE | `v_3(index)` computation exists in `padic_losses.py` | Already implemented |
| Color by shared prefix | ✅ FEASIBLE | Derive from operation index base-3 representation | Simple base-3 slicing |
| Color by VAE-A vs VAE-B | ✅ FEASIBLE | Separate forward passes, stack embeddings | Direct |
| Color by reconstruction error | ✅ FEASIBLE | `loss_recon_A`, `loss_recon_B` per sample | Need to modify loss to return per-sample |
| Radial density histogram | ✅ FEASIBLE | `||z||` from `outputs['z_A/B']` | Trivial |
| Radial vs angular variance | ✅ FEASIBLE | Convert to polar, compute variance | Needs implementation |
| Topological distortion metric | ✅ FEASIBLE | `d_poincare(a,b) / d_3adic(a,b)` | Poincaré distance implemented |
| 3-adic cluster stability | ⚠️ PARTIAL | Centroid drift needs epoch-to-epoch tracking | Need to store centroids per epoch |

#### H.2) High-Signal Plots (Section 2)

| Visualization | Feasibility | Data Source | Notes |
|---------------|-------------|-------------|-------|
| Phase-space: entropy vs KL | ✅ FEASIBLE | `H_A`, `H_B`, `loss_kl_A`, `loss_kl_B` | Already logged |
| Color by coverage | ✅ FEASIBLE | `coverage_A`, `coverage_B` | Already logged |
| Bifurcation plots (ρ, temp, λ3) | ⚠️ REQUIRES RUNS | Need multiple training runs varying params | Analysis script needed |
| grad_norm_A / grad_norm_B | ✅ FEASIBLE | `model.grad_norm_A_ema`, `model.grad_norm_B_ema` | Already tracked |
| KL_A vs KL_B | ✅ FEASIBLE | `outputs` dict | Already logged |
| recon_A vs recon_B | ✅ FEASIBLE | `outputs` dict | Already logged |

#### H.3) Training-Time Control (Section 3)

| Visualization | Feasibility | Data Source | Notes |
|---------------|-------------|-------------|-------|
| Latent variance spikes | ✅ FEASIBLE | `outputs['logvar_A/B']` → exp(logvar).mean() | Simple |
| Radial collapse detection | ✅ FEASIBLE | `model.mean_radius_A/B` | Already tracked in v5.10 |
| Gradient ratio inversion | ✅ FEASIBLE | `outputs['grad_ratio']` or loss dict | Already tracked |
| Automatic triggers | ⚠️ NEEDS HOOKS | Callback system not in current trainer | Would need trainer modification |
| Embedding trajectory replay | ⚠️ STORAGE-HEAVY | Would need to save z_A, z_B per epoch | Checkpoint extension needed |

#### H.4) Inference-Time Observability (Section 4)

| Visualization | Feasibility | Data Source | Notes |
|---------------|-------------|-------------|-------|
| Coverage vs sampling temp | ✅ FEASIBLE | Sweep temp parameter at inference | Simple loop |
| Novelty rate | ⚠️ DEFINITION NEEDED | Need to define "novel" vs training set | Requires design decision |
| Invalidity rate | ❓ UNCLEAR | No validity metric defined for ternary ops | Domain-specific |
| ε-ball neighborhood probing | ✅ FEASIBLE | Sample z±ε, decode, compare | Straightforward |
| Decoder variance | ⚠️ SINGLE-PASS | Current decoder deterministic | Would need dropout/ensemble |
| Bit-flip sensitivity | ✅ FEASIBLE | Perturb z, measure output change | Simple |

#### H.5) Missing Elements Not Addressed in Proposal

| Element | Status | Impact |
|---------|--------|--------|
| Hyperparameter documentation | ❌ MISSING | Can't verify "optimal" settings |
| Baseline comparisons | ❌ MISSING | No reference for "good" metrics |
| Statistical significance | ❌ MISSING | No confidence intervals proposed |
| Computational cost estimates | ❌ MISSING | Unknown overhead per visualization |
| Update frequency recommendations | ❌ MISSING | Every batch? epoch? N epochs? |

---

### I) Recommended Implementation Priority

Based on feasibility and insight value:

**Tier 1 (Immediate, High Value):**
1. 3D latent embedding colored by 3-adic depth (already have z_A/z_B)
2. Radial density histogram (trivial from ||z||)
3. Phase-space: entropy vs KL vs coverage (already logged)
4. Topological distortion over epochs (have all components)

**Tier 2 (Moderate Effort):**
5. ε-ball neighborhood probing at inference
6. Gradient ratio time series with phase markers
7. Per-sample reconstruction error coloring

**Tier 3 (Requires Infrastructure):**
8. Embedding trajectory replay (checkpoint extension)
9. Automatic triggers via callback hooks
10. Bifurcation analysis (multi-run orchestration)

---

### J) Code Entry Points for Implementation

```python
# Access latent embeddings for visualization
from src import DualNeuralVAEV5_10
model = DualNeuralVAEV5_10(...)
outputs = model(x, temp_A, temp_B, beta_A, beta_B)
z_A = outputs['z_A']  # (batch, 16)
z_B = outputs['z_B']  # (batch, 16)

# Project to Poincaré ball
from src.metrics.hyperbolic import project_to_poincare, poincare_distance
z_A_hyp = project_to_poincare(z_A, max_norm=0.95)

# Compute 3-adic valuation
from src.losses.padic_losses import compute_3adic_valuation
# Not directly exposed - would need: v_3(|i-j|) implementation

# Access training metrics
from src.training.monitor import TrainingMonitor
monitor = TrainingMonitor(log_dir="runs/...")
# TensorBoard logs at: runs/ternary_vae_{timestamp}/

# Ranking correlation evaluation
from src.metrics.hyperbolic import compute_ranking_correlation_hyperbolic
corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc, r_A, r_B = \
    compute_ranking_correlation_hyperbolic(model, device)
```

---

**Document Version:** 1.0
**Architecture Reference:** Ternary VAE v5.10 (Pure Hyperbolic Geometry)
**Codebase Audit Date:** 2025-12-12
