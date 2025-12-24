# Ternary VAE Architecture Documentation

**Version:** 5.11 (Unified Hyperbolic Geometry with Frozen Coverage)
**Last Updated:** 2025-12-14
**Status:** Production-ready

---

## Overview

The Ternary VAE v5.11 represents a paradigm shift from pure dual-pathway training to a **Frozen Encoder** architecture. It solves the "radial hierarchy inversion" observed in v5.10 by:

1.  **Freezing the v5.5 Encoder:** Preserving 100% operation coverage without risk of catastrophic forgetting.
2.  **Trainable Hyperbolic Projection:** Learning the geometric embedding (Poincare ball) on top of the fixed Euclidean features.
3.  **Differentiable Controller:** Dynamically adjusting loss weights using StateNet v4.

**Key Innovation:** Decoupling **coverage** (solved by v5.5) from **geometry** (solved by v5.11), allowing for precise hierarchical embedding without destabilizing the latent space.

---

## System Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Script                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │      HyperbolicVAETrainer          │
         │  (Orchestrates hyperbolic training) │
         └────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
  ┌──────────┐   ┌──────────────┐   ┌──────────────┐
  │Frozen    │   │ Hyperbolic   │   │ StateNet     │
  │Encoder   │   │ Projection   │   │ Controller   │
  │(v5.5)    │   │ (Trainable)  │   │ (Trainable)  │
  └──────────┘   └──────────────┘   └──────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   TernaryVAEV5_11     │
              │ (Frozen + Projected)  │
              └───────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
         ┌────────┐              ┌────────┐
         │z_euc   │              │ z_hyp  │
         │(Fixed) │              │(Learned│
         │        │              │ Struct)│
         └────────┘              └────────┘
```

---

## Codebase Statistics

| Module     | Files  | Lines     | Purpose                                           |
| ---------- | ------ | --------- | ------------------------------------------------- |
| models/    | 5      | 2,312     | VAE architectures (v5.6, v5.7, v5.10, appetitive) |
| training/  | 6      | 2,517     | Training orchestration + monitoring               |
| losses/    | 7      | 3,531     | Loss computations (23 distinct components)        |
| metrics/   | 2      | 232       | Evaluation metrics (hyperbolic geometry)          |
| data/      | 3      | 163       | Dataset generation and loading                    |
| utils/     | 3      | 500       | Utility functions and helpers                     |
| artifacts/ | 2      | 149       | Checkpoint management                             |
| **Total**  | **29** | **9,435** | Production-ready codebase                         |

---

## Model Lineage

```
ternary_vae_v5_6.py (Base Dual VAE - 538 lines)
├── Dual encoder/decoder architecture
├── Stop-gradient cross-injection
├── Adaptive gradient balance
└── StateNet v1 (12D → 4D)

      ↓ extends

ternary_vae_v5_7.py (Metric Attention - 625 lines)
├── Inherits all v5.6 features
├── StateNet v3 (14D → 5D, adds r_A, r_B, delta_ranking)
└── Dynamic ranking weight modulation

      ↓ extends

ternary_vae_v5_10.py (Pure Hyperbolic - 822 lines)
├── Inherits all v5.7 features
├── StateNet v4 (18D → 7D, adds hyperbolic state)
├── Pure hyperbolic geometry (no Euclidean contamination)
└── Homeostatic parameter adaptation
```

---

## Module Responsibilities

### 1. Training Module (`src/training/`) - 2,517 lines

**Purpose:** Orchestrate training process with clean separation of concerns.

#### HyperbolicVAETrainer (`hyperbolic_trainer.py` - 495 lines)

**Responsibility:** v5.10 hyperbolic training orchestration

**Key Features:**

- Pure hyperbolic loss computation (prior, recon, centroid)
- Continuous feedback for ranking weight adaptation
- Homeostatic parameter modulation (prior_sigma, curvature)
- Evaluation interval optimization (coverage every 5 epochs, correlation every 20)

**Key Methods:**

- `train_epoch(train_loader, val_loader, epoch)`: Execute one training epoch
- `_compute_hyperbolic_losses()`: Compute all hyperbolic loss components
- `_update_homeostatic_params()`: Adapt prior_sigma and curvature

#### TernaryVAETrainer (`trainer.py` - 440 lines)

**Responsibility:** Base training loop orchestration

**Key Methods:**

- `__init__(model, config, device)`: Initialize trainer with dependencies
- `train_epoch(train_loader)`: Execute one training epoch
- `validate(val_loader)`: Run validation

**Dependencies:**

- TemperatureScheduler, BetaScheduler, LearningRateScheduler
- TrainingMonitor for logging
- CheckpointManager for persistence
- DualVAELoss for base loss computation

#### TrainingMonitor (`monitor.py` - 694 lines)

**Responsibility:** Unified logging and metrics tracking

**Key Methods:**

- `log_epoch_summary()`: Comprehensive epoch metrics
- `log_hyperbolic_epoch()`: v5.10 hyperbolic-specific metrics
- `log_batch()`: Batch-level TensorBoard logging
- `evaluate_coverage()`: Compute operation coverage
- `_log()`: Unified console + file logging

**Tracks:**

- Hyperbolic correlation (r_A_hyp, r_B_hyp)
- Euclidean correlation (r_A_euc, r_B_euc)
- Mean radii (boundary vs origin positioning)
- Homeostatic metrics (prior_sigma, curvature)
- Coverage and entropy history

#### Schedulers (`schedulers.py` - 211 lines)

**Classes:**

1. **TemperatureScheduler**: Linear annealing with cyclic modulation
2. **BetaScheduler**: KL warmup to prevent posterior collapse
3. **LearningRateScheduler**: Step-based learning rate scheduling

---

### 2. Loss Module (`src/losses/`) - 3,531 lines

**Purpose:** All loss computation separated from model architecture.

#### Hyperbolic Losses (v5.10 - NEW)

**HyperbolicPrior** (`hyperbolic_prior.py` - 387 lines)

- Wrapped Normal distribution on Poincare ball
- KL computed in tangent space with change-of-measure correction
- Prior mass concentrates at origin (tree root)

**HomeostaticHyperbolicPrior**

- Extends HyperbolicPrior with adaptive sigma/curvature
- Prevents collapse (sigma too small) or explosion (sigma too big)

**HyperbolicReconLoss** (`hyperbolic_recon.py` - 546 lines)

- Geodesic distance-based reconstruction
- Radius-weighted cross-entropy (points near origin matter more)
- Natural curriculum: learns tree structure before leaves

**HomeostaticReconLoss**

- Adaptive reconstruction with curriculum learning
- radius_power adapts based on training progress

**HyperbolicCentroidLoss**

- Frechet mean (hyperbolic centroid) for each 3-adic prefix cluster
- Multi-level: root → branches → leaves (max_level=4 → 81 clusters)

#### P-adic Losses (`padic_losses.py` - 1,047 lines)

**PAdicRankingLossHyperbolic**

- Triplet loss using Poincare distance
- Radial weight for hierarchy enforcement
- Hard negative mining

**PAdicMetricLoss** (DISABLED in v5.10)

- Euclidean metric alignment (conflicts with hyperbolic structure)

**PAdicNormLoss** (DISABLED in v5.10)

- Radial hierarchy now handled by radial_weight

#### Base Losses (`dual_vae_loss.py` - 524 lines)

- **ReconstructionLoss**: Cross-entropy for ternary operations
- **KLDivergenceLoss**: KL divergence with free bits
- **EntropyRegularization**: Output diversity
- **RepulsionLoss**: RBF kernel-based latent repulsion
- **DualVAELoss**: Unified loss computation

---

### 3. Metrics Module (`src/metrics/`) - 232 lines

**Purpose:** Hyperbolic evaluation metrics for 3-adic ranking correlation.

#### Hyperbolic Metrics (`hyperbolic.py` - 209 lines)

**Key Functions:**

- `project_to_poincare(z, max_norm)`: Project latents to Poincare ball
- `poincare_distance(u, v, c)`: Compute hyperbolic distance
- `compute_3adic_valuation(x)`: 3-adic valuation for ultrametric
- `compute_ranking_correlation_hyperbolic(z_A, z_B, x, curvature)`: Main evaluation metric

**Returns:**

- `corr_A_hyp`, `corr_B_hyp`: Hyperbolic correlation (target: r > 0.99)
- `corr_A_euc`, `corr_B_euc`: Euclidean correlation (comparison baseline)
- `mean_radius_A`, `mean_radius_B`: Position in Poincare ball

---

### 4. Models Module (`src/models/`) - 2,312 lines

**Purpose:** Neural network architecture only.

#### DualNeuralVAEV5_10 (`ternary_vae_v5_10.py` - 822 lines)

**Key Components:**

1. **StateNet v4** (18D input → 7D output)

   - Inherits v5.7: H_A, H_B, KL_A, KL_B, grad_ratio, rho, lambda3, coverage_A, coverage_B, r_A, r_B, delta_ranking
   - v5.10 adds: mean_radius_A, mean_radius_B, prior_sigma, curvature
   - Outputs: lr_correction, lambda1-3 corrections, ranking_weight, delta_sigma, delta_curvature

2. **VAE-A (Chaotic Regime)**

   - Explores boundary of Poincare ball
   - Higher temperature, variable beta
   - Target: high coverage through exploration

3. **VAE-B (Stable Regime)**
   - Anchors near origin of Poincare ball
   - Lower temperature, stabilizing influence
   - Target: consistent tree structure

**Architecture Details:**

- Total parameters: ~170,000
- VAE-A: ~50,000 params
- VAE-B: ~118,000 params (residual blocks)
- StateNet v4: ~1,500 params

#### DualNeuralVAEV5_7 (`ternary_vae_v5_7.py` - 625 lines)

- StateNet v3 with metric attention
- Dynamic ranking weight modulation
- Backward compatible with v5.6 checkpoints

#### DualNeuralVAEV5 (`ternary_vae_v5_6.py` - 538 lines)

- Base dual VAE architecture
- Stop-gradient cross-injection
- Adaptive gradient balance

---

### 5. Data Module (`src/data/`) - 163 lines

**Purpose:** Data generation and loading.

#### Generation (`generation.py` - 62 lines)

- `generate_all_ternary_operations()`: Generate all 19,683 operations
- `count_ternary_operations()`: Return total count (3^9)
- `generate_ternary_operation_by_index(idx)`: Generate specific operation

#### Dataset (`dataset.py` - 79 lines)

- `TernaryOperationDataset`: PyTorch dataset wrapper
- Validates shape and value ranges ({-1, 0, 1})

---

### 6. Artifacts Module (`src/artifacts/`) - 149 lines

**Purpose:** Checkpoint lifecycle management.

#### CheckpointManager (`checkpoint_manager.py` - 136 lines)

- `save_checkpoint()`: Saves latest.pt, best.pt, epoch_N.pt
- `load_checkpoint()`: Restore state with metadata
- `list_checkpoints()`: Enumerate available checkpoints

---

## Training Flow (v5.10)

### Initialization

```python
# 1. Load configuration
config = yaml.safe_load('configs/ternary_v5_10.yaml')

# 2. Create monitor (unified logging)
monitor = TrainingMonitor(
    eval_num_samples=1000,
    tensorboard_dir='runs',
    log_dir='logs',
    log_to_file=True
)

# 3. Generate data
operations = generate_all_ternary_operations()
dataset = TernaryOperationDataset(operations)
train_loader, val_loader = create_data_loaders(dataset, config)

# 4. Initialize model (v5.10 with StateNet v4)
model = DualNeuralVAEV5_10(
    input_dim=9, latent_dim=16,
    use_statenet=True,
    statenet_hyp_sigma_scale=0.05,
    statenet_hyp_curvature_scale=0.02
)

# 5. Initialize trainers
base_trainer = TernaryVAETrainer(model, config, device)
trainer = HyperbolicVAETrainer(base_trainer, model, device, config, monitor)

# 6. Train
for epoch in range(300):
    losses = trainer.train_epoch(train_loader, val_loader, epoch)
    monitor.log_epoch_summary(epoch, ...)
```

### Training Loop (per epoch)

```python
for epoch in range(total_epochs):
    # 1. Forward pass with scheduled parameters
    outputs = model(batch, temp_A, temp_B, beta_A, beta_B)

    # 2. Compute hyperbolic losses
    hyp_kl_A = hyperbolic_prior(mu_A, logvar_A, z_A)
    hyp_kl_B = hyperbolic_prior(mu_B, logvar_B, z_B)
    centroid_loss = centroid_loss_fn(z_A, z_B, x)
    ranking_loss = ranking_loss_hyperbolic(z_A, z_B, x)

    # 3. Backward and optimize
    total_loss.backward()
    model.update_gradient_norms()
    optimizer.step()

    # 4. Homeostatic adaptation
    if config['homeostatic']:
        prior.adapt(mean_radius_A, mean_radius_B, ...)

    # 5. Evaluate (optimized intervals)
    if epoch % 5 == 0:  # Coverage check
        cov_A, cov_B = evaluate_coverage(...)
    if epoch % 20 == 0:  # Correlation check
        corr_A_hyp, corr_B_hyp = compute_ranking_correlation_hyperbolic(...)

    # 6. Log and checkpoint
    monitor.log_epoch_summary(epoch, ...)
```

---

## Configuration (v5.10)

```yaml
config_version: "5.10"

model:
  input_dim: 9
  latent_dim: 16
  use_statenet: true
  statenet_hyp_sigma_scale: 0.05
  statenet_hyp_curvature_scale: 0.02

padic_losses:
  # DISABLED: Euclidean contamination
  enable_metric_loss: false
  enable_norm_loss: false

  # ENABLED: Pure hyperbolic
  enable_ranking_loss_hyperbolic: true
  ranking_hyperbolic:
    curvature: 2.0
    radial_weight: 0.4
    max_norm: 0.95

  hyperbolic_v10:
    use_hyperbolic_prior: true
    prior:
      homeostatic: true
      curvature: 2.0
      prior_sigma: 1.0
    use_hyperbolic_recon: true
    use_centroid_loss: true

# Evaluation intervals (optimized)
coverage_check_interval: 5
eval_interval: 20
eval_num_samples: 1000
```

---

## Target Metrics

| Metric                 | Target   | Description                               |
| ---------------------- | -------- | ----------------------------------------- |
| Hyperbolic Correlation | r > 0.99 | 3-adic ranking preserved in Poincare ball |
| Coverage               | > 99.7%  | Operations discovered                     |
| Mean Radius A          | 0.7-0.9  | VAE-A explores boundary                   |
| Mean Radius B          | 0.3-0.5  | VAE-B anchors near origin                 |

---

## File Reference

### Core Files (v5.10)

| File                                   | Lines | Purpose                                |
| -------------------------------------- | ----- | -------------------------------------- |
| `src/models/ternary_vae_v5_10.py`      | 822   | Pure hyperbolic model with StateNet v4 |
| `src/training/hyperbolic_trainer.py`   | 495   | Hyperbolic training orchestration      |
| `src/losses/hyperbolic_prior.py`       | 387   | Wrapped normal prior on Poincare ball  |
| `src/losses/hyperbolic_recon.py`       | 546   | Geodesic reconstruction loss           |
| `src/metrics/hyperbolic.py`            | 209   | Hyperbolic correlation metrics         |
| `scripts/train/train_ternary_v5_10.py` | 202   | Training entry point                   |
| `configs/ternary_v5_10.yaml`           | 285   | v5.10 configuration                    |

### Supporting Files

| File                          | Lines | Purpose               |
| ----------------------------- | ----- | --------------------- |
| `src/training/trainer.py`     | 440   | Base trainer          |
| `src/training/monitor.py`     | 694   | Unified logging       |
| `src/losses/padic_losses.py`  | 1,047 | P-adic ranking losses |
| `src/losses/dual_vae_loss.py` | 524   | Base VAE losses       |

---

## See Also

- **API Reference:** `docs/API_REFERENCE.md`
- **Mathematical Foundations:** `docs/theory/MATHEMATICAL_FOUNDATIONS.md`
- **Installation:** `docs/INSTALLATION_AND_USAGE.md`
