# Manifold Approach: Training on Exhaustive Finite Sets

**Doc-Type:** Design Analysis · Version 2.1 · Updated 2025-12-12 · Author Claude Code

---

## Architectural Context

This document proposes changes aligned with the codebase architecture defined in `src/README.md`.

**Key Architectural Principles:**
- **Joints** = Orchestration points that wire components together
- **Non-Joints** = Leaf components with single responsibilities
- **Dependency flow:** `data → models → losses → training`

**Affected Joints:**
| Joint | File | Impact |
|-------|------|--------|
| `HyperbolicVAETrainer` | `training/hyperbolic_trainer.py` | Skip validation when val_loader=None |
| `DualVAELoss` | `losses/dual_vae_loss.py` | Potential coverage-aware loss weighting (Phase 2) |
| `models/__init__.py` | Module joint | Add model factory (Phase 4) |

**Not Affected (confirmed by code review):**
| Joint | File | Reason |
|-------|------|--------|
| `TernaryVAETrainer` | `training/trainer.py` | `train()` method not used by v5.10 script |

**Affected Non-Joints:**
| Component | File | Impact |
|-----------|------|--------|
| `create_ternary_data_loaders()` | `data/loaders.py` | Handle val_split=0 |
| `TrainingMonitor` | `training/monitor.py` | Coverage-based early stopping |
| `CheckpointManager` | `artifacts/checkpoint_manager.py` | Already supports is_best flag |

---

## Problem Statement

Current training uses 80/10/10 train/val/test splits on an exhaustive finite set of 19,683 (3^9) ternary operations. This is problematic because:

1. **No unseen data exists** - The dataset contains ALL possible ternary operations
2. **Validation loss is meaningless** - There's nothing to "generalize" to
3. **Reduced training signal** - Only 15,746 samples used instead of 19,683
4. **Wrong optimization target** - Minimizing validation loss instead of maximizing coverage/correlation

---

## Core Insight

For manifold learning on exhaustive finite sets, the goal is **perfect reconstruction and structural preservation**, not generalization. The metrics that matter are:

| Metric | Purpose | Target |
|--------|---------|--------|
| Coverage % | All operations reconstructable | > 99.7% |
| Ranking Correlation (hyperbolic) | 3-adic structure preserved | r > 0.99 |
| Reconstruction Accuracy | Discrete values recovered | > 99% |

Validation loss measures generalization ability - irrelevant when the entire population is known.

---

## Proposed Changes

### 1. Data Loading (`src/data/loaders.py`)

**Current:** Returns train/val/test loaders with splits
**Proposed:** Handle `val_split=0` gracefully, return `None` for val_loader

```python
# When val_split=0, return None instead of empty loader
if val_size == 0:
    val_loader = None
```

### 2. Trainer (`src/training/hyperbolic_trainer.py`)

**Current:** `train_epoch()` requires val_loader, calls validation
**Proposed:** Make validation optional when val_loader is None

```python
def train_epoch(self, train_loader, val_loader, epoch):
    # ... training ...

    if val_loader is not None:
        val_losses = self.base_trainer.validate(val_loader)
    else:
        val_losses = {}  # Skip validation
```

### 3. Config (`configs/ternary_v5_10.yaml`)

**Current:**
```yaml
train_split: 0.8
val_split: 0.1
test_split: 0.1
```

**Proposed:**
```yaml
train_split: 1.0
val_split: 0.0
test_split: 0.0
```

### 4. Checkpoint Selection (`src/artifacts/checkpoint_manager.py`)

**Current:** `is_best` based on validation loss
**Proposed:** `is_best` based on coverage or correlation

```python
# Replace validation loss comparison
is_best = current_coverage > best_coverage
# OR
is_best = current_correlation > best_correlation
```

### 5. Early Stopping (`src/training/monitor.py`)

**Current:** Monitors validation loss plateau
**Proposed:** Monitor coverage/correlation plateau

```python
# Stop when coverage stops improving
if coverage_delta < min_coverage_delta for N epochs:
    should_stop = True
```

---

## Files Requiring Changes (Verified)

| File | Change Type | Complexity | Lines |
|------|-------------|------------|-------|
| `src/data/loaders.py` | Handle val_split=0, conditional loader creation | Low | ~15 |
| `src/training/hyperbolic_trainer.py` | Skip validation call (line 290) | Trivial | ~5 |
| `src/training/monitor.py` | Add composite score + plateau tracking | Medium | ~50 |
| `scripts/train/train_ternary_v5_10.py` | Null checks + composite is_best | Low | ~15 |
| `configs/ternary_v5_10.yaml` | Split values | Trivial | ~3 |

**NOT Required (confirmed by code review):**
| File | Originally Planned | Why Not Needed |
|------|-------------------|----------------|
| `src/training/trainer.py` | Optional validation | `train()` not used by v5.10 script |
| `src/artifacts/checkpoint_manager.py` | is_best logic | Already takes `is_best` as parameter |

**Revised total:** ~88 lines of changes

---

## Metrics Pipeline (Current State)

```
train_epoch()
    │
    ├─> loss (reconstruction + KL + hyperbolic)
    ├─> coverage % (unique codes / total)
    ├─> correlation (3-adic ranking vs latent distance)
    │
    └─> val_losses (SHOULD BE REMOVED)
           │
           └─> is_best decision (WRONG TARGET)
           └─> early stopping (WRONG TARGET)
```

---

## Metrics Pipeline (Proposed)

```
train_epoch()
    │
    ├─> loss (reconstruction + KL + hyperbolic)
    ├─> coverage % ──────────────────────────┐
    ├─> correlation ─────────────────────────┤
    │                                        │
    └─> is_best = coverage > best_coverage <─┘
    └─> early_stop = coverage plateau
```

---

## Design Decisions (Resolved)

### Q1: Which metric for is_best?

**Decision:** Use **composite score** combining coverage and correlation.

```python
# Composite score formula
composite = 0.6 * (coverage / 100) + 0.4 * max(correlation, 0)
is_best = composite > best_composite
```

**Rationale:**
- Coverage alone can plateau at 99%+ while correlation still improves
- Correlation alone ignores reconstruction quality
- 60/40 weighting prioritizes coverage (the primary goal) while rewarding structural fidelity

### Q2: Early stopping criteria?

**Decision:** Stop when **both** metrics plateau for N epochs.

```python
coverage_plateau = (max_coverage - current_coverage) < 0.1 for 30 epochs
correlation_plateau = (max_correlation - current_correlation) < 0.01 for 30 epochs
should_stop = coverage_plateau and correlation_plateau
```

**Rationale:**
- Stopping on coverage alone might miss correlation improvements
- Stopping on correlation alone might miss coverage improvements
- Requiring both ensures training continues while either metric improves

### Q3: Checkpoint frequency?

**Decision:** Save checkpoints every **10 epochs** + always save **best** and **latest**.

**Rationale:**
- With 300 epochs, every-epoch saves would create 300 files (~50MB each = 15GB)
- Every 10 epochs = 30 numbered checkpoints + best + latest
- CheckpointManager already implements this pattern

### Q4: Learning rate scheduling?

**Decision:** Keep **epoch-based** scheduling, no coverage milestones.

**Rationale:**
- Coverage-based LR scheduling adds complexity with minimal benefit
- Current epoch-based schedule (warmup → decay) works well
- If coverage plateaus, early stopping handles it

### Q5: Validation purpose?

**Decision:** **Remove validation entirely** for exhaustive finite sets.

**Rationale:**
- No generalization to measure
- Validation loss is mathematically meaningless for complete populations
- Coverage/correlation computed on training data are the true metrics
- Simplifies code and removes confusion

---

## DualVAELoss Consideration

`DualVAELoss` (`src/losses/dual_vae_loss.py`) is a **Joint** that aggregates multiple loss terms. Currently it doesn't consider coverage, but could be extended for coverage-aware weighting.

**Current aggregation:**
```python
total_loss = ce_loss + beta * kl_loss + lambda1 * entropy + lambda2 * repulsion + lambda3 * padic_loss
```

**Potential enhancement (OPTIONAL - not in initial scope):**
```python
# Increase p-adic loss weight as coverage approaches target
coverage_factor = 1.0 + (current_coverage / target_coverage) * 0.5
total_loss = ce_loss + beta * kl_loss + coverage_factor * lambda3 * padic_loss
```

**Recommendation:** Defer to Phase 2. The manifold approach (100% training) should be validated first before adding coverage-aware loss weighting.

---

## Related Code Locations

Analysis complete (2025-12-12):

- [x] `src/training/monitor.py` - Needs new methods for composite score + plateau tracking
- [x] `src/training/trainer.py` - **No changes needed** (train() not used by v5.10)
- [x] `src/training/hyperbolic_trainer.py` - Line 290: wrap validate() in null check
- [x] `src/artifacts/checkpoint_manager.py` - **No changes needed** (is_best is parameter)
- [x] `src/data/loaders.py` - Lines 64-84: handle zero-size splits
- [x] `scripts/train/train_ternary_v5_10.py` - Lines 80, 185: null checks + composite score

---

## Implementation Status

| Task | Status |
|------|--------|
| Document findings | COMPLETE |
| Architectural analysis | COMPLETE |
| Design decisions resolved | COMPLETE |
| Analyze affected code | COMPLETE |
| Implement data loader changes | NOT STARTED |
| Implement trainer changes | NOT STARTED |
| Implement monitor changes | NOT STARTED |
| Update config | NOT STARTED |
| Test full-data training | NOT STARTED |

---

## Phased Implementation Plan

### Phase 1: Manifold Approach (100% Training)

**Goal:** Train on all 19,683 operations with coverage-based optimization.

| Step | File | Change | Lines | Code Location |
|------|------|--------|-------|---------------|
| 1.1 | `src/data/loaders.py` | Handle `val_split=0`, return `None` | ~15 | Lines 64-84 |
| 1.2 | `src/training/hyperbolic_trainer.py` | Skip validation if `val_loader is None` | ~5 | Line 290 |
| 1.3 | `src/training/monitor.py` | Add `compute_composite_score()`, plateau tracking | ~50 | New methods + state |
| 1.4 | `scripts/train/train_ternary_v5_10.py` | Null checks + composite is_best | ~15 | Lines 80, 185 |
| 1.5 | `configs/ternary_v5_10.yaml` | Set `train_split: 1.0, val_split: 0.0` | ~3 | Split config |

**Estimated:** ~88 lines

**Removed from plan (not needed):**
- `src/training/trainer.py` - The `train()` method (line 367) is not used by v5.10 script

### Phase 2: Resume from Checkpoint

**Goal:** Enable training recovery from interruptions.

| Step | File | Change | Lines |
|------|------|--------|-------|
| 2.1 | `scripts/train/train_ternary_v5_10.py` | Add `--resume` argument | ~5 |
| 2.2 | `scripts/train/train_ternary_v5_10.py` | Load checkpoint and restore state | ~20 |
| 2.3 | `src/training/hyperbolic_trainer.py` | Add `restore_state()` method | ~15 |

**Estimated:** ~40 lines

### Phase 3: Unified Logging

**Goal:** All output through TrainingMonitor.

| Step | File | Change | Lines |
|------|------|--------|-------|
| 3.1 | `src/training/trainer.py` | Replace 18 `print()` with `monitor._log()` | ~20 |
| 3.2 | `src/training/trainer.py` | Rename `_print_init_summary()` → `_log_init_summary()` | ~2 |

**Estimated:** ~22 lines

### Phase 4: Model Factory

**Goal:** Centralize model instantiation.

| Step | File | Change | Lines |
|------|------|--------|-------|
| 4.1 | `src/models/__init__.py` | Add `create_model_from_config()` | ~40 |
| 4.2 | `scripts/train/train_ternary_v5_10.py` | Use factory instead of inline `create_model()` | ~5 |

**Estimated:** ~45 lines

---

## Total Implementation Estimate

| Phase | Lines | Priority | Impact |
|-------|-------|----------|--------|
| Phase 1: Manifold | ~88 | HIGH | Training correctness |
| Phase 2: Resume | ~40 | HIGH | Reliability |
| Phase 3: Logging | ~22 | MEDIUM | Observability |
| Phase 4: Factory | ~45 | MEDIUM | Code organization |
| **Total** | **~195** | | |

---

## Code Review Findings (2025-12-12)

Detailed analysis of actual source code to verify implementation plan.

### Finding 1: `src/data/loaders.py` - More Complex Than Expected

**Current behavior (lines 64-84):**
```python
# Line 64-67: random_split always creates 3 datasets
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=generator
)

# Lines 78-84: val_loader ALWAYS created, even if val_size=0
val_loader = DataLoader(val_dataset, batch_size=batch_size, ...)
```

**Issue:** When `val_split=0`, `val_size=0`, creating an empty DataLoader that will fail when iterated.

**Required fix:**
```python
# After computing sizes, handle zero-size splits
if val_size == 0 and test_size == 0:
    train_dataset = dataset  # Use full dataset
    val_loader = None
    test_loader = None
else:
    # existing split logic
    ...
    val_loader = DataLoader(...) if val_size > 0 else None
```

### Finding 2: `src/training/hyperbolic_trainer.py` - Simpler Than Expected

**Current behavior (line 290):**
```python
val_losses = self.base_trainer.validate(val_loader)  # Always called
```

**Key insight:** `val_losses` is **never used** in the return dict (line 343 uses `**train_losses`). This is wasted compute.

**Required fix (line 290):**
```python
if val_loader is not None:
    val_losses = self.base_trainer.validate(val_loader)
# No else needed - val_losses not used anyway
```

### Finding 3: `src/training/trainer.py` - NO CHANGES NEEDED

**Analysis:**
- `train_epoch()` (line 186) - Does NOT call validation
- `validate()` (line 310) - Standalone method, only called from elsewhere
- `train()` (line 367) - **Not used by v5.10 script**

The v5.10 script uses `HyperbolicVAETrainer.train_epoch()` directly, which handles its own validation call. No changes needed to trainer.py for Phase 1.

### Finding 4: `src/training/monitor.py` - Needs New State

**Current state:**
- `check_best()` (line 383) - Uses `self.best_val_loss`
- `should_stop()` (line 400) - Patience-based on validation loss

**Missing for manifold approach:**
1. `compute_composite_score()` method
2. Plateau tracking state variables:
   - `self.coverage_plateau_counter`
   - `self.correlation_plateau_counter`
   - `self.best_composite`
3. `should_stop_manifold()` method with dual-plateau logic

### Finding 5: `scripts/train/train_ternary_v5_10.py` - Will Crash

**Line 80:** `get_data_loader_info(val_loader)` - crashes if val_loader is None

**Line 185:** `is_best = losses['loss'] < trainer.monitor.best_val_loss` - wrong metric

**Required fixes:**
```python
# Line 80: Add null check
if val_loader is not None:
    monitor._log(f"Val: {get_data_loader_info(val_loader)['size']:,} samples")
else:
    monitor._log("Val: None (manifold mode - 100% training)")

# Line 185: Use composite score
composite = trainer.monitor.compute_composite_score(
    (losses['cov_A'] + losses['cov_B']) / 2,
    losses['corr_mean_hyp']
)
is_best = composite > best_composite
best_composite = max(best_composite, composite)
```

### Finding 6: `src/artifacts/checkpoint_manager.py` - No Changes Needed

The `is_best` flag is passed as a parameter (line 33). The fix is in the calling code (training script), not here.

---

## Implementation Details by Phase

### Phase 1 Details: Manifold Approach

**1.1 Data Loader Changes** (`src/data/loaders.py`):
```python
def create_ternary_data_loaders(...) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    # ... existing validation ...

    # Compute sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Handle manifold mode (100% training)
    if val_size == 0 and test_size == 0:
        train_loader = DataLoader(
            dataset,  # Use full dataset, no split
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
        return train_loader, None, None

    # ... existing split logic for non-manifold mode ...
```

**1.2 Skip Validation** (`src/training/hyperbolic_trainer.py`):
```python
# Line 290 - wrap in conditional
if val_loader is not None:
    val_losses = self.base_trainer.validate(val_loader)
```

**1.3 Composite Score + Plateau Tracking** (`src/training/monitor.py`):
```python
def __init__(self, ...):
    # Add to existing __init__
    self.best_composite = 0.0
    self.coverage_plateau_counter = 0
    self.correlation_plateau_counter = 0
    self.plateau_threshold_coverage = 0.1  # 0.1% improvement threshold
    self.plateau_threshold_correlation = 0.01
    self.plateau_patience = 30  # epochs

def compute_composite_score(self, coverage: float, correlation: float) -> float:
    """Compute composite score for checkpoint selection."""
    return 0.6 * (coverage / 100) + 0.4 * max(correlation, 0)

def check_best_composite(self, coverage: float, correlation: float) -> bool:
    """Check if current metrics are best, update plateau counters."""
    composite = self.compute_composite_score(coverage, correlation)
    is_best = composite > self.best_composite

    # Update plateau counters
    if coverage > self.best_coverage + self.plateau_threshold_coverage:
        self.coverage_plateau_counter = 0
    else:
        self.coverage_plateau_counter += 1

    if correlation > self.best_corr_hyp + self.plateau_threshold_correlation:
        self.correlation_plateau_counter = 0
    else:
        self.correlation_plateau_counter += 1

    if is_best:
        self.best_composite = composite
    return is_best

def should_stop_manifold(self) -> bool:
    """Check if both metrics have plateaued."""
    return (self.coverage_plateau_counter >= self.plateau_patience and
            self.correlation_plateau_counter >= self.plateau_patience)
```

**1.4 Training Script Updates** (`scripts/train/train_ternary_v5_10.py`):
```python
# At top of run_training_loop():
best_composite = 0.0

# Line 80 fix - null check for val_loader:
if val_loader is not None:
    monitor._log(f"Val: {get_data_loader_info(val_loader)['size']:,} samples")
else:
    monitor._log("Val: None (manifold mode - 100% training)")

# Line 185 fix - composite-based is_best:
is_best = trainer.monitor.check_best_composite(
    (losses['cov_A'] + losses['cov_B']) / 2,
    losses['corr_mean_hyp']
)
```

### Phase 2 Details: Resume from Checkpoint

**2.3 Restore State Method** (`src/training/hyperbolic_trainer.py`):
```python
def restore_state(self, checkpoint: dict) -> None:
    """Restore trainer state from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dict
    """
    self.best_corr_hyp = checkpoint.get('best_corr_hyp', 0)
    self.best_corr_euc = checkpoint.get('best_corr_euc', 0)
    self.best_coverage = checkpoint.get('best_coverage', 0)
    self.correlation_history_hyp = checkpoint.get('correlation_history_hyp', [])
    self.correlation_history_euc = checkpoint.get('correlation_history_euc', [])
    self.coverage_history = checkpoint.get('coverage_history', [])
    self.ranking_weight_history = checkpoint.get('ranking_weight_history', [])
```

### Phase 3 Details: Unified Logging

**Print Statement Locations** (`src/training/trainer.py`):

| Line | Current | Replace With |
|------|---------|--------------|
| ~119 | `print(f"{'='*80}")` | `self.monitor._log(...)` |
| ~120 | `print("Dual Neural VAE...")` | `self.monitor._log(...)` |
| ~121-127 | Parameter summary prints | `self.monitor._log(...)` |
| ~367-376 | Training start banner | `self.monitor._log(...)` |

### Phase 4 Details: Model Factory

**Factory Function** (`src/models/__init__.py`):
```python
def create_model_from_config(config: dict, version: str = 'v5_10') -> nn.Module:
    """Create model from config dict."""
    mc = config['model']

    if version == 'v5_10':
        from .ternary_vae_v5_10 import DualNeuralVAEV5_10
        return DualNeuralVAEV5_10(
            input_dim=mc['input_dim'],
            latent_dim=mc['latent_dim'],
            rho_min=mc['rho_min'],
            rho_max=mc['rho_max'],
            lambda3_base=mc['lambda3_base'],
            lambda3_amplitude=mc['lambda3_amplitude'],
            eps_kl=mc['eps_kl'],
            gradient_balance=mc.get('gradient_balance', True),
            adaptive_scheduling=mc.get('adaptive_scheduling', True),
            use_statenet=mc.get('use_statenet', True),
            statenet_lr_scale=mc.get('statenet_lr_scale', 0.1),
            statenet_lambda_scale=mc.get('statenet_lambda_scale', 0.02),
            statenet_ranking_scale=mc.get('statenet_ranking_scale', 0.3),
            statenet_hyp_sigma_scale=mc.get('statenet_hyp_sigma_scale', 0.05),
            statenet_hyp_curvature_scale=mc.get('statenet_hyp_curvature_scale', 0.02)
        )
    raise ValueError(f"Unknown model version: {version}")
```

---

## References

- `src/README.md` - Codebase architecture (joints/non-joints)
- `OBSERVABILITY_REFACTORING.md` - Current training infrastructure
- `configs/ternary_v5_10.yaml` - Current configuration
- `src/training/hyperbolic_trainer.py` - Training orchestration
- `src/artifacts/checkpoint_manager.py` - Existing checkpoint logic
