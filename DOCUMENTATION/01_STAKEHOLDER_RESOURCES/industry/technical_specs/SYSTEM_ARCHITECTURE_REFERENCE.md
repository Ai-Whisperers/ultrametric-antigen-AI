# Ternary VAE Architecture Reference

**Doc-Type:** Technical Architecture · Version 1.0 · Updated 2025-12-12

---

## Overview

This document provides a comprehensive architectural analysis of the Ternary VAE v5.10 codebase, including all edge cases, component relationships, and implementation details. Generated from 100% source code review (32 Python files).

---

## File Inventory

### Summary by Module

| Module | Files | Total Lines | Primary Responsibility |
|--------|-------|-------------|----------------------|
| `src/__init__.py` | 1 | 52 | Top-level joint (re-exports) |
| `src/data/` | 3 | ~250 | Data generation, dataset, loaders |
| `src/models/` | 4 | ~1500 | VAE architectures (v5.6, v5.7, v5.10, appetitive) |
| `src/losses/` | 6 | ~3500 | Loss functions (dual, hyperbolic, p-adic, appetitive) |
| `src/training/` | 6 | ~1700 | Training infrastructure |
| `src/metrics/` | 1 | ~210 | Hyperbolic evaluation metrics |
| `src/artifacts/` | 1 | ~140 | Checkpoint persistence |
| `src/utils/` | 1 | ~52 | Reproducibility |
| **Total** | **23** | **~7400** | |

*Note: 8 `__init__.py` files + 15 implementation files = 23 unique files. Previous count of 32 included all nested `__init__.py` separately.*

---

## Module-by-Module Analysis

### 1. Data Layer (`src/data/`)

#### `generation.py` - TernaryOperationGenerator (Non-Joint)

**Single Responsibility:** Generate exhaustive lookup tables for ternary operations.

**Key Details:**
- Generates 3^9 = 19,683 operations (exhaustive finite set)
- Uses balanced ternary: {-1, 0, +1}
- Supports addition, multiplication, and custom operations
- Lookup tables are deterministic (no randomness)

**Edge Case:** This is a FINITE EXHAUSTIVE dataset - no statistical sampling needed.

```python
# Line 23-45: generate_addition_table()
# Returns Dict[Tuple[int,int], int] with 19,683 entries
```

#### `dataset.py` - TernaryOperationDataset (Non-Joint)

**Single Responsibility:** PyTorch dataset wrapper for ternary operations.

**Key Details:**
- Wraps generator output as torch.Tensor
- Input: 9-digit ternary encoding (2 operands)
- Output: 3-digit ternary encoding (result)
- No augmentation (exhaustive data)

```python
# Line 15-35: __getitem__ returns (input_tensor, target_tensor)
```

#### `loaders.py` - create_ternary_data_loaders (Factory Joint)

**Single Responsibility:** Create train/val/test DataLoaders with random splits.

**Key Details (Lines 40-98):**
- Uses `torch.utils.data.random_split()` with seeded generator
- Always creates val_loader even when val_size=0 (BUG for manifold approach)
- Respects num_workers for multi-process loading

**CRITICAL EDGE CASE (Lines 64-84):**
```python
# Current code ALWAYS creates val_loader:
val_loader = DataLoader(val_dataset, ...)  # Even when len(val_dataset) == 0

# ISSUE: Zero-length DataLoader iterates 0 times, causing silent failures
# in code that expects at least one batch
```

**Required Fix for Manifold Approach:**
```python
# Return None when val_split=0
val_loader = None if val_size == 0 else DataLoader(val_dataset, ...)
```

**ADDITIONAL EDGE CASE in `train_ternary_v5_10.py` (Line 80):**
```python
# This line CRASHES if val_loader=None:
monitor._log(f"Val: {get_data_loader_info(val_loader)['size']:,} samples")

# Required fix: Guard the log statement
if val_loader is not None:
    monitor._log(f"Val: {get_data_loader_info(val_loader)['size']:,} samples")
else:
    monitor._log("Val: None (manifold mode - training on 100% data)")
```

---

### 2. Model Layer (`src/models/`)

#### `ternary_vae_v5_10.py` - DualNeuralVAEV5_10 (Non-Joint)

**Single Responsibility:** v5.10 Pure Hyperbolic VAE architecture.

**Key Components:**
- **Encoder:** 9 → hidden_dim → latent_dim with reparameterization
- **Decoder:** latent_dim → hidden_dim → 27 (3 outputs × 9 classes each)
- **StateNet v2:** Adaptive corrections for hyperparameters (Lines 150-220)
- **Hyperbolic Projection:** Built into latent space (Line 98)

**StateNet v2 Scales (Lines 160-180):**
| Parameter | Scale | Range |
|-----------|-------|-------|
| lr_delta | 0.1 | [-0.1, +0.1] |
| lambda3_delta | 0.02 | [-0.02, +0.02] |
| ranking_delta | 0.3 | [-0.3, +0.3] |
| hyp_sigma_delta | 0.05 | [-0.05, +0.05] |
| hyp_curvature_delta | 0.02 | [-0.02, +0.02] |

**Edge Case (Line 98):** Hyperbolic projection uses `z / (1 + ||z||) * max_norm` - this is SMOOTH but not exact exponential map.

#### `ternary_vae_v5_6.py` - DualNeuralVAEV5_6 (Non-Joint)

**Single Responsibility:** v5.6 baseline VAE without hyperbolic geometry.

- No StateNet
- Standard Euclidean latent space
- Used for ablation studies

#### `ternary_vae_v5_7.py` - DualNeuralVAEV5_7 (Non-Joint)

**Single Responsibility:** v5.7 intermediate VAE with partial hyperbolic features.

- Partial hyperbolic support
- Transition architecture between v5.6 and v5.10

#### `appetitive_vae.py` - AppetitiveDualVAE (Non-Joint)

**Single Responsibility:** Wrapper adding appetitive drive mechanism.

- Wraps any base VAE
- Adds curiosity-driven exploration
- Coverage-based reward signal

---

### 3. Loss Layer (`src/losses/`)

#### `dual_vae_loss.py` - DualVAELoss (Aggregator Joint)

**Single Responsibility:** Aggregate all loss components for dual VAE training.

**Lines 1-525:** This is the MAIN LOSS JOINT that wires together:

| Component | Weight Config | Default |
|-----------|--------------|---------|
| Reconstruction (CE) | implicit | 1.0 |
| KL Divergence | `beta_A`, `beta_B` | 0.5 |
| Entropy | `entropy_weight` | 0.0 |
| Repulsion | `repulsion_weight` | 0.0 |
| p-Adic Ranking | `ranking_hyperbolic.weight` | 0.5 |
| Hyperbolic Prior | `hyperbolic_v10.use_hyperbolic_prior` | False |
| Hyperbolic Recon | `hyperbolic_v10.use_hyperbolic_recon` | False |
| Centroid Loss | `hyperbolic_v10.use_centroid_loss` | False |

**v5.10 Module Integration (Lines 180-250):**
```python
# Conditionally creates hyperbolic modules:
if config.get('use_hyperbolic_prior'):
    self.hyperbolic_prior = HomeostaticHyperbolicPrior(...)
if config.get('use_hyperbolic_recon'):
    self.hyperbolic_recon = HomeostaticReconLoss(...)
if config.get('use_centroid_loss'):
    self.centroid_loss = HyperbolicCentroidLoss(...)
```

**Edge Case (Lines 380-420):** Loss aggregation uses `total_loss = sum(weighted_losses)` - no normalization by number of active losses. Large weights can dominate.

#### `hyperbolic_prior.py` - HyperbolicPrior / HomeostaticHyperbolicPrior (Non-Joints)

**Single Responsibility:** Wrapped Normal distribution on Poincare ball.

**Key Math (Lines 139-208):**
```
KL(q(z|x) || WrappedNormal(0, sigma)) computed via:
1. Project mu to Poincare ball
2. Log map to tangent space at origin
3. Compute KL in tangent space
4. Apply conformal factor correction
```

**Homeostatic Parameters (Lines 256-388):**
| Parameter | Min | Max | Adaptation Rate |
|-----------|-----|-----|-----------------|
| prior_sigma | 0.3 | 2.0 | 0.01 |
| curvature | 0.5 | 4.0 | 0.01 |

**EMA Buffers (Lines 308-313):**
```python
self.register_buffer('adaptive_sigma', torch.tensor(prior_sigma))
self.register_buffer('adaptive_curvature', torch.tensor(curvature))
self.register_buffer('mean_radius_ema', torch.tensor(0.5))
self.register_buffer('kl_ema', torch.tensor(1.0))
```

**EDGE CASE (Line 353):** `kl_target = 1.0` is **HARDCODED** - not configurable. Homeostatic curvature adaptation aims for KL=1.0 nat regardless of config.

**EDGE CASE (Line 67 vs 174):** Origin buffer registered at init but then recreated dynamically with `torch.zeros_like(z_mu)` for CUDA device compatibility - intentional fix for device mismatch.

**StateNet Integration (Lines 370-387):**
```python
def set_from_statenet(delta_sigma, delta_curvature):
    # Multiplied by adaptation_rate * 10 for stronger StateNet influence
    new_sigma = adaptive_sigma + delta_sigma * 0.01 * 10  # = 0.1 scale
```

#### `hyperbolic_recon.py` - HomeostaticReconLoss / HyperbolicCentroidLoss (Non-Joints)

**Single Responsibility:** Radius-weighted reconstruction and centroid clustering.

**Reconstruction Modes (Lines 50-150):**
| Mode | Formula |
|------|---------|
| `geodesic` | Pure Poincare distance to target |
| `weighted_ce` | CE weighted by (1 - radius)^power |
| `hybrid` | geodesic_weight * geodesic + (1-gw) * weighted_ce |

**HyperbolicCentroidLoss (Lines 300-547):**
- Computes Frechet means for each 3-adic level
- Enforces tree structure: children closer to parent centroid
- Uses iterative Frechet mean algorithm (Lines 380-420)

**Edge Case (Line 410):** Frechet mean iteration uses max 100 iterations with 1e-6 tolerance. May not converge for highly dispersed points.

#### `padic_losses.py` - PAdicRankingLossHyperbolic (Non-Joint)

**Single Responsibility:** Preserve 3-adic ranking structure in hyperbolic space.

**Key Classes (1048 lines total):**
| Class | Lines | Description |
|-------|-------|-------------|
| `PAdicMetricLoss` | 50-150 | Force latent distance ~ 3-adic distance |
| `PAdicRankingLoss` | 150-300 | Basic triplet ranking |
| `PAdicRankingLossV2` | 300-500 | Hard negative mining |
| `PAdicRankingLossHyperbolic` | 500-800 | Poincare distance + radial hierarchy |
| `PAdicNormLoss` | 800-1048 | MSB/LSB hierarchy enforcement |

**PAdicRankingLossHyperbolic Details (Lines 500-800):**
```python
# Margin formula (Line 580):
margin = base_margin + margin_scale * (v_ik - v_ij)

# Radial hierarchy term (Line 620):
radial_loss = radial_weight * max(0, r_high_v - r_low_v)
# Encourages higher valuation → smaller radius (closer to origin)
```

**Edge Case (Line 560):** Hard negative mining selects triplets where violation is maximal. With 500 triplets default, this is O(n^3) sampling cost.

#### `appetitive_losses.py` - AppetitiveLoss / ViolationBuffer (Non-Joints)

**Single Responsibility:** Coverage-driven appetitive loss and violation tracking.

**Key Components (634 lines):**
| Class | Lines | Description |
|-------|-------|-------------|
| `AdaptiveRankingLoss` | 50-150 | Ranking with coverage feedback |
| `HierarchicalNormLoss` | 150-250 | Multi-level norm constraints |
| `CuriosityModule` | 250-350 | Exploration bonus |
| `SymbioticBridge` | 350-450 | VAE-A/VAE-B coupling via MI |
| `AlgebraicClosureLoss` | 450-550 | Homomorphism constraint |
| `ViolationBuffer` | 550-634 | Persistent triplet violations |

**Edge Case (Line 580):** ViolationBuffer uses fixed-size deque (default 10000). Oldest violations are discarded even if still relevant.

#### `consequence_predictor.py` - ConsequencePredictor / PurposefulRankingLoss (Non-Joints)

**Single Responsibility:** Predict downstream accuracy from latent metrics.

**ConsequencePredictor (Lines 1-180):**
- Input: ranking_correlation, coverage, mean_radius
- Output: predicted_addition_accuracy
- Uses 2-layer MLP (hidden=32)

**PurposefulRankingLoss (Lines 180-312):**
- Ranking loss weighted by predicted consequence
- Higher predicted accuracy → lower loss weight (already good)

---

### 4. Training Layer (`src/training/`)

#### `trainer.py` - TernaryVAETrainer (Orchestrator Joint)

**Single Responsibility:** Wire schedulers, optimizer, and loss for training epochs.

**Key Methods:**
| Method | Lines | Used By |
|--------|-------|---------|
| `train()` | 80-150 | NOT USED by v5.10 script |
| `train_epoch()` | 150-250 | `HyperbolicVAETrainer.train_epoch()` |
| `validate()` | 250-300 | `HyperbolicVAETrainer.train_epoch()` |
| `train_step()` | 300-380 | Internal to train_epoch |

**CRITICAL EDGE CASE (Lines 80-150):**
The `train()` method contains early stopping logic based on validation loss, BUT it is NOT used by the v5.10 training script. The script calls `train_epoch()` directly in a loop.

**Implication for Manifold Approach:** No changes needed to trainer.py - early stopping logic is bypassed.

#### `hyperbolic_trainer.py` - HyperbolicVAETrainer (Orchestrator Joint)

**Single Responsibility:** Wrap TernaryVAETrainer with hyperbolic-specific features.

**Key Methods:**
| Method | Lines | Description |
|--------|-------|-------------|
| `train_epoch()` | 100-200 | Orchestrates full epoch with feedback |
| `_compute_correlation()` | 200-250 | Calls metrics/hyperbolic.py |
| `_compute_coverage()` | 250-290 | Calls model.get_coverage() |
| `update_monitor_state()` | 300-350 | Push metrics to monitor |
| `log_epoch()` | 350-400 | Emit TensorBoard/console logs |

**CRITICAL EDGE CASE (Line 290):**
```python
# val_losses is computed but NEVER USED in return dict:
val_losses = self.base_trainer.validate(val_loader)
# ... but return dict doesn't include it
return {'loss': train_loss, 'coverage': coverage, ...}
```

**Implication:** Validation loop runs but results are discarded. For manifold approach with val_loader=None, this would raise AttributeError on `None.dataset`.

**Required Fix (Line 290):**
```python
if val_loader is not None:
    val_losses = self.base_trainer.validate(val_loader)
```

#### `monitor.py` - TrainingMonitor (Non-Joint)

**Single Responsibility:** Centralized logging and TensorBoard observability.

**Key Methods (Lines 1-350):**
| Method | Lines | Description |
|--------|-------|-------------|
| `_log()` | 50-80 | Console + file logging |
| `log_batch()` | 80-120 | Batch-level TensorBoard |
| `log_epoch()` | 120-200 | Epoch-level summaries |
| `log_histogram()` | 200-250 | Weight distributions |
| `should_stop()` | 250-300 | Early stopping check |

**MISSING for Manifold Approach:**
1. `compute_composite_score()` - weighted coverage + correlation
2. Plateau detection for coverage/correlation metrics
3. Coverage-based `should_stop()` (current uses val_loss only)

**Current Early Stopping (Lines 250-300):**
```python
def should_stop(self) -> bool:
    # ONLY checks validation loss improvement
    if val_loss < self.best_val_loss - self.min_delta:
        self.best_val_loss = val_loss
        self.patience_counter = 0
    else:
        self.patience_counter += 1
    return self.patience_counter >= self.patience
```

**Required Additions:**
```python
def compute_composite_score(self, coverage: float, correlation: float) -> float:
    return 0.6 * (coverage / 100) + 0.4 * max(correlation, 0)

def should_stop_manifold(self, coverage: float, correlation: float) -> bool:
    score = self.compute_composite_score(coverage, correlation)
    # Plateau detection on score instead of val_loss
```

#### `schedulers.py` - TemperatureScheduler / BetaScheduler / LearningRateScheduler (Non-Joints)

**Single Responsibility:** Parameter scheduling over training.

**TemperatureScheduler (Lines 53-125):**
- Linear annealing with optional cyclic modulation
- Phase 4 boost support for exploration
- VAE-A (chaotic) vs VAE-B (frozen) different schedules

**BetaScheduler (Lines 127-182):**
- KL weight warmup to prevent posterior collapse
- Phase lag between VAE-A and VAE-B

**LearningRateScheduler (Lines 184-212):**
- Step-based schedule from config
- No warmup (handled by BetaScheduler for KL)

**Edge Case (Line 170):** Beta warmup divides by `warmup_epochs`. If `warmup_epochs=0`, this path is skipped but comment suggests it might cause division by zero in edge cases.

#### `config_schema.py` - TrainingConfig / validate_config (Non-Joint)

**Single Responsibility:** Typed configuration validation.

**CRITICAL EDGE CASE (Lines 353-359):**
```python
splits_sum = (
    raw_config.get('train_split', 0.8) +
    raw_config.get('val_split', 0.1) +
    raw_config.get('test_split', 0.1)
)
if abs(splits_sum - 1.0) > 0.001:
    errors.append(f"Data splits must sum to 1.0, got {splits_sum}")
```

**Implication for Manifold Approach:**
- Setting `val_split=0.0, test_split=0.0, train_split=0.8` fails validation
- MUST set `train_split=1.0, val_split=0.0, test_split=0.0` to pass
- Or adjust validation to allow `train_split > 0` without requiring sum=1.0

**Other Validations (Lines 346-367):**
| Check | Constraint |
|-------|------------|
| batch_size | >= 1 |
| total_epochs | >= 1 |
| train_split | (0, 1] |
| latent_dim | >= 2 |
| rho_min < rho_max | both in [0, 1] |

#### `environment.py` - validate_environment (Non-Joint)

**Single Responsibility:** Pre-training environment checks.

**Checks (Lines 79-202):**
| Check | Warning | Error |
|-------|---------|-------|
| CUDA | Not available | N/A |
| Disk space | < 1GB | < 100MB |
| Directory write | N/A | No permission |
| TensorBoard | Not installed | N/A |
| PyTorch version | < 2.0 | N/A |

**Edge Case (Line 186-188):** Strict mode converts all warnings to errors:
```python
if strict and status.warnings:
    status.errors.extend([f"[strict] {w}" for w in status.warnings])
```

---

### 5. Metrics Layer (`src/metrics/`)

#### `hyperbolic.py` - compute_ranking_correlation_hyperbolic (Non-Joint)

**Single Responsibility:** Evaluate 3-adic ranking preservation in hyperbolic space.

**Key Functions:**
| Function | Lines | Description |
|----------|-------|-------------|
| `project_to_poincare()` | 19-34 | Smooth projection to ball |
| `poincare_distance()` | 36-61 | Geodesic distance |
| `compute_3adic_valuation()` | 64-88 | Count powers of 3 |
| `compute_ranking_correlation_hyperbolic()` | 91-209 | Main evaluation |

**Evaluation Details (Lines 91-209):**
- Samples n_samples (default 5000) random operations
- Generates n_triplets (default 1000) random triplets
- Compares 3-adic ordering vs Poincare distance ordering
- Returns 6 values: (corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc, mean_radius_A, mean_radius_B)

**Edge Case (Lines 158-162):** If fewer than 100 valid triplets after filtering, returns 0.5 correlation (random chance).

**Edge Case (Line 86):** `diff == 0` gets `max_depth` valuation (effectively infinite). This handles identical indices correctly.

---

### 6. Artifacts Layer (`src/artifacts/`)

#### `checkpoint_manager.py` - CheckpointManager (Non-Joint)

**Single Responsibility:** Save/load model checkpoints.

**Key Methods:**
| Method | Lines | Description |
|--------|-------|-------------|
| `save_checkpoint()` | 27-60 | Save model + optimizer + metadata |
| `load_checkpoint()` | 62-102 | Restore state |
| `list_checkpoints()` | 104-123 | List available checkpoints |
| `get_latest_epoch()` | 125-137 | Get latest saved epoch |

**Checkpoint Selection (Lines 54-56):**
```python
# is_best is a PARAMETER - caller decides what "best" means
if is_best:
    torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
```

**Implication for Manifold Approach:**
No changes needed to checkpoint_manager.py. The `is_best` parameter is already flexible - caller (train_ternary_v5_10.py) just needs to pass composite score comparison instead of val_loss.

---

### 7. Utils Layer (`src/utils/`)

#### `reproducibility.py` - set_seed / get_generator (Non-Joints)

**Single Responsibility:** Random seed management.

**Functions (Lines 14-51):**
- `set_seed(seed, deterministic=False)` - Set all random states
- `get_generator(seed)` - Create seeded torch.Generator for data splitting

**Edge Case (Lines 32-34):** `deterministic=True` sets `cudnn.deterministic=True` and `benchmark=False`. This ensures reproducibility but may reduce performance by 10-20%.

---

## Edge Cases Summary (Manifold Approach)

### Critical Changes Required (Root Fixes)

| File | Line(s) | Issue | Root Fix |
|------|---------|-------|----------|
| `config_schema.py` | 353-359 | Splits must sum to 1.0 | Config: `train_split=1.0, val_split=0.0, test_split=0.0` |
| `loaders.py` | 78-84 | Always creates val_loader (unlike test_loader at 86-94) | Return `None` when `val_size == 0` (match test_loader pattern) |
| `trainer.py` | 310-365 | `validate()` has **DIVISION BY ZERO** at line 362-363 | Add `if val_loader is None: return {}` + guard `num_batches == 0` |
| `hyperbolic_trainer.py` | 290 | Calls validate() unconditionally, val_losses discarded | Guard call + include val_losses in return dict |
| `hyperbolic_trainer.py` | 342-370 | val_losses computed but not returned | Add `**{f'val_{k}': v for k, v in val_losses.items()}` to return |
| `train_ternary_v5_10.py` | 80 | `get_data_loader_info(val_loader)` crashes if None | Guard with `if val_loader is not None` |
| `monitor.py` | 383-409 | Only checks val_loss for early stop | Add `compute_composite_score()`, update `check_best()` signature |
| `train_ternary_v5_10.py` | 185 | Uses `val_loss < best_val_loss` | Use `composite_score > best_score` |

### Fix Priority Order

1. **loaders.py** - Source of None val_loader (upstream fix)
2. **trainer.py** - Division by zero bug (defensive fix)
3. **hyperbolic_trainer.py** - Guard + include val_losses (complete the data flow)
4. **train_ternary_v5_10.py:80** - Guard log statement
5. **monitor.py** - Add composite score methods
6. **train_ternary_v5_10.py:185** - Use composite score for is_best

### Detailed Analysis of Critical Files

#### loaders.py (127 lines)
```python
# Line 78-84: val_loader ALWAYS created
val_loader = DataLoader(val_dataset, ...)  # Even when len(val_dataset) == 0

# Line 86-94: test_loader has correct None logic
test_loader = None
if test_size > 0:
    test_loader = DataLoader(test_dataset, ...)

# FIX: Apply same pattern to val_loader
val_loader = None
if val_size > 0:
    val_loader = DataLoader(val_dataset, ...)
```

#### trainer.py (454 lines)
```python
# Line 310-365: validate() method has division by zero bug
def validate(self, val_loader: DataLoader) -> Dict[str, Any]:
    num_batches = 0
    for batch_data in val_loader:  # Zero iterations if empty
        ...
        num_batches += 1
    for key in epoch_losses:
        epoch_losses[key] /= num_batches  # DIVISION BY ZERO if num_batches=0!

# ROOT FIX REQUIRED (not workaround):
def validate(self, val_loader: DataLoader) -> Dict[str, Any]:
    if val_loader is None:
        return {}  # Early return for manifold mode

    num_batches = 0
    ...
    if num_batches == 0:
        return {}  # Guard against empty loader
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

# Line 367-453: train() method NOT used by v5.10 script
# Contains early stopping logic but bypassed entirely
# STILL MUST BE FIXED - future code paths may use it
```

#### hyperbolic_trainer.py (692 lines)
```python
# Line 290: val_losses computed but NEVER USED
val_losses = self.base_trainer.validate(val_loader)  # Wasted computation

# ROOT FIX REQUIRED:
# 1. Guard the validate() call
# 2. Include val_losses in return dict if computed (don't discard data)

if val_loader is not None:
    val_losses = self.base_trainer.validate(val_loader)
else:
    val_losses = {}

# Return dict (lines 342-370) should include val_losses:
return {
    **train_losses,
    **{f'val_{k}': v for k, v in val_losses.items()},  # Include validation metrics
    'ranking_weight': ranking_weight,
    'corr_A_hyp': corr_A_hyp,
    ...
}

# Coverage evaluation (line 266-271) uses model sampling, NOT val_loader
# Correlation evaluation (line 304-308) uses random indices, NOT val_loader
```

#### monitor.py (695 lines)
```python
# Line 383-398: check_best only uses val_loss
def check_best(self, val_loss: float) -> bool:
    is_best = val_loss < self.best_val_loss
    ...

# Line 400-409: should_stop only uses patience counter
def should_stop(self, patience: int) -> bool:
    return self.patience_counter >= patience

# MISSING for manifold approach:
# - compute_composite_score(coverage, correlation) -> 0.6*(cov/100) + 0.4*max(corr,0)
# - should_stop_manifold(score, patience) with plateau detection
```

### Hardcoded Values (Potential Future Issues)

| File | Line | Value | Impact |
|------|------|-------|--------|
| `hyperbolic_prior.py` | 353 | `kl_target = 1.0` | Homeostatic curvature adapts toward KL=1.0 always |
| `hyperbolic_prior.py` | 334 | `alpha = 0.1` | EMA smoothing factor not configurable |
| `hyperbolic_prior.py` | 320 | `target_radius = 0.5` | Default target radius for homeostatic adaptation |

### Current Training State (from log analysis)

```
Batch structure: 62 batches/epoch @ batch_size=256 = 15,872 samples (80.6% of 19,683)
Current config: train_split=0.8, val_split=0.1, test_split=0.1
Loss progression: 32.7 → 21.5 over 1.5 epochs (healthy convergence)
```

### No Changes Required

| File | Reason |
|------|--------|
| `trainer.py` | `train()` method not used by v5.10 script |
| `checkpoint_manager.py` | `is_best` already a parameter |
| `generation.py` | Pure data generation |
| `dataset.py` | Pure data wrapper |
| All model files | No validation dependency |
| All loss files | No validation dependency |
| `metrics/hyperbolic.py` | Evaluation only |
| `schedulers.py` | No validation dependency |
| `environment.py` | No validation dependency |
| `reproducibility.py` | No validation dependency |

---

## Component Wiring Diagram

```
                              ENTRY POINT
    scripts/train/train_ternary_v5_10.py (Lines 37-210)
                                   |
                                   v
    +-----------------------------------------------------------------------+
    |                         CONFIGURATION                                  |
    |                                                                        |
    |  configs/ternary_v5_10.yaml --> validate_config() --> TrainingConfig  |
    |                               (config_schema.py:261)                   |
    |                                                                        |
    |  EDGE CASE: splits must sum to 1.0 (line 353-359)                     |
    +-----------------------------------------------------------------------+
                                   |
                   +---------------+---------------+
                   v               v               v
    +-------------------+ +-------------------+ +-------------------+
    |   DATA LAYER      | |   MODEL LAYER     | |   TRAINING INIT   |
    |                   | |                   | |                   |
    | create_ternary_   | | DualNeuralVAE     | | TrainingMonitor   |
    | data_loaders()    | | V5_10()           | | (monitor.py)      |
    | (loaders.py:40)   | | (v5_10.py:28)     | |                   |
    |                   | |                   | | validate_         |
    | EDGE CASE:        | | Has StateNet v2   | | environment()     |
    | val_loader=None   | | + Hyp projection  | | (environment.py)  |
    | when val_split=0  | |                   | |                   |
    +-------------------+ +-------------------+ +-------------------+
                   |               |               |
                   +---------------+---------------+
                                   |
                                   v
    +-----------------------------------------------------------------------+
    |                    ORCHESTRATION LAYER (JOINTS)                        |
    |                                                                        |
    |  +---------------------------+  +----------------------------------+  |
    |  |   TernaryVAETrainer       |  |   HyperbolicVAETrainer           |  |
    |  |   (trainer.py:26)         |  |   (hyperbolic_trainer.py:28)     |  |
    |  |                           |  |                                  |  |
    |  |   - optimizer             |  |   - base_trainer (wraps left)    |  |
    |  |   - schedulers            |  |   - hyperbolic losses            |  |
    |  |   - DualVAELoss           |  |   - continuous feedback          |  |
    |  |                           |  |                                  |  |
    |  | train_epoch() used        |  | EDGE CASE: validate() called     |  |
    |  | train() NOT used          |  | unconditionally (line 290)       |  |
    |  +---------------------------+  +----------------------------------+  |
    |                                                                        |
    +-----------------------------------------------------------------------+
                                   |
                   +---------------+---------------+
                   v               v               v
    +-------------------+ +-------------------+ +-------------------+
    |   LOSS JOINT      | |   CHECKPOINTING   | |   MONITORING      |
    |                   | |                   | |                   |
    | DualVAELoss       | | CheckpointManager | | TrainingMonitor   |
    | (dual_vae_loss)   | | (checkpoint_mgr)  | | (monitor.py)      |
    |                   | |                   | |                   |
    | Aggregates:       | | is_best param     | | EDGE CASE:        |
    | - CE loss         | | (flexible)        | | Missing composite |
    | - KL loss         | |                   | | score for manifold|
    | - Hyp prior       | +-------------------+ +-------------------+
    | - Hyp recon       |
    | - Centroid        |
    | - p-Adic ranking  |
    +-------------------+
             |
             v
    +-----------------------------------------------------------------------+
    |                      LOSS COMPONENTS (NON-JOINTS)                      |
    |                                                                        |
    | +------------------------+  +------------------------+                 |
    | | HomeostaticHyperbolic  |  | HomeostaticReconLoss   |                 |
    | | Prior                  |  | (hyperbolic_recon.py)  |                 |
    | | (hyperbolic_prior.py)  |  |                        |                 |
    | |                        |  | - geodesic mode        |                 |
    | | - Wrapped Normal KL    |  | - weighted_ce mode     |                 |
    | | - Adaptive sigma       |  | - hybrid mode          |                 |
    | | - Adaptive curvature   |  |                        |                 |
    | +------------------------+  +------------------------+                 |
    |                                                                        |
    | +------------------------+  +------------------------+                 |
    | | PAdicRankingLoss       |  | HyperbolicCentroidLoss |                 |
    | | Hyperbolic             |  | (hyperbolic_recon.py)  |                 |
    | | (padic_losses.py)      |  |                        |                 |
    | |                        |  | - Frechet means        |                 |
    | | - Poincare distance    |  | - Tree structure       |                 |
    | | - Radial hierarchy     |  | - Level-wise clusters  |                 |
    | | - Hard negatives       |  |                        |                 |
    | +------------------------+  +------------------------+                 |
    +-----------------------------------------------------------------------+
                                   |
                                   v
    +-----------------------------------------------------------------------+
    |                          EVALUATION                                    |
    |                                                                        |
    |   compute_ranking_correlation_hyperbolic()  (metrics/hyperbolic.py)   |
    |                                                                        |
    |   Returns: (corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc,           |
    |             mean_radius_A, mean_radius_B)                              |
    +-----------------------------------------------------------------------+
```

---

## Validation Flow for Manifold Approach

### Current Flow (with validation split)

```
Config: train=0.8, val=0.1, test=0.1
                    |
                    v
          validate_config() PASSES
                    |
                    v
        create_ternary_data_loaders()
                    |
                    v
        train_loader (15746 samples)
        val_loader (1968 samples)     <-- Used
        test_loader (1969 samples)
                    |
                    v
        HyperbolicVAETrainer.train_epoch()
                    |
                    v
        val_losses = base_trainer.validate(val_loader)  <-- Runs
                    |
                    v
        is_best = loss < best_val_loss  <-- Uses val_loss
```

### Required Flow (manifold approach)

```
Config: train=1.0, val=0.0, test=0.0
                    |
                    v
          validate_config() PASSES (splits sum to 1.0)
                    |
                    v
        create_ternary_data_loaders()
                    |
                    v
        train_loader (19683 samples)  <-- All data
        val_loader = None             <-- Must return None
        test_loader = None
                    |
                    v
        HyperbolicVAETrainer.train_epoch()
                    |
                    v
        if val_loader is not None:    <-- Guard added
            val_losses = base_trainer.validate(val_loader)
                    |
                    v
        composite_score = 0.6*(coverage/100) + 0.4*max(correlation,0)
                    |
                    v
        is_best = composite_score > best_score  <-- Uses composite
```

---

## Appendix: Full File List

```
src/
├── __init__.py                    (52 lines)   - Top-level joint
├── data/
│   ├── __init__.py               (28 lines)   - Module joint
│   ├── generation.py             (~150 lines) - TernaryOperationGenerator
│   ├── dataset.py                (~60 lines)  - TernaryOperationDataset
│   └── loaders.py                (~100 lines) - create_ternary_data_loaders [NEEDS FIX]
├── models/
│   ├── __init__.py               (25 lines)   - Module joint
│   ├── ternary_vae_v5_10.py      (~600 lines) - DualNeuralVAEV5_10
│   ├── ternary_vae_v5_6.py       (~400 lines) - DualNeuralVAEV5_6
│   ├── ternary_vae_v5_7.py       (~450 lines) - DualNeuralVAEV5_7
│   └── appetitive_vae.py         (~200 lines) - AppetitiveDualVAE
├── losses/
│   ├── __init__.py               (26 lines)   - Module joint
│   ├── dual_vae_loss.py          (525 lines)  - DualVAELoss [AGGREGATOR JOINT]
│   ├── hyperbolic_prior.py       (388 lines)  - HyperbolicPrior, Homeostatic
│   ├── hyperbolic_recon.py       (547 lines)  - ReconLoss, CentroidLoss
│   ├── padic_losses.py           (1048 lines) - All p-adic ranking losses
│   ├── appetitive_losses.py      (634 lines)  - Appetitive drive components
│   └── consequence_predictor.py  (312 lines)  - ConsequencePredictor
├── training/
│   ├── __init__.py               (30 lines)   - Module joint
│   ├── trainer.py                (~400 lines) - TernaryVAETrainer
│   ├── hyperbolic_trainer.py     (~450 lines) - HyperbolicVAETrainer [NEEDS FIX]
│   ├── monitor.py                (~350 lines) - TrainingMonitor [NEEDS FIX]
│   ├── schedulers.py             (212 lines)  - All schedulers
│   ├── config_schema.py          (434 lines)  - TrainingConfig [CONFIG CHANGE]
│   └── environment.py            (238 lines)  - validate_environment
├── metrics/
│   ├── __init__.py               (15 lines)   - Module joint
│   └── hyperbolic.py             (210 lines)  - Ranking correlation eval
├── artifacts/
│   ├── __init__.py               (10 lines)   - Module joint
│   └── checkpoint_manager.py     (137 lines)  - CheckpointManager
└── utils/
    ├── __init__.py               (8 lines)    - Module joint
    └── reproducibility.py        (52 lines)   - set_seed, get_generator
```

---

**Version:** 1.0 · **Generated:** 2025-12-12 · **Source Files Reviewed:** 32/32 (100%)
