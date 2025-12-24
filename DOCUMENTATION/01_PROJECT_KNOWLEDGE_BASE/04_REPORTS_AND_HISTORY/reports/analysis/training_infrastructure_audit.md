# Training Infrastructure Architectural Audit

**Doc-Type:** Technical Audit Report · Version 1.0 · Updated 2025-12-12 · Author Claude Code

---

## Executive Summary

The v5.10.1 training infrastructure has accumulated architectural debt through incremental additions without proper integration. The primary issues are:

1. **Dual monitor instances** - `TernaryVAETrainer` creates its own monitor, while `train_ternary_v5_10.py` creates a second "unified" monitor
2. **Fragmented logging** - 36+ direct print statements bypass file logging
3. **No configuration validation** - YAML loaded without schema checking
4. **Inconsistent logging API** - Multiple methods for similar logging needs
5. **Missing pre-training validation** - No checks for CUDA, disk space, config validity

---

## 1. Monitor Instance Fragmentation

### Current State

```
train_ternary_v5_10.py
    |
    +-- Creates: TrainingMonitor (unified) ──────────────> logs/training_*.log
    |                                                      runs/ternary_vae_*
    +-- Creates: TernaryVAETrainer
            |
            +-- Creates: TrainingMonitor (duplicate!) ──> logs/training_*.log (SECOND FILE)
                                                          runs/ternary_vae_* (SECOND DIR)
```

### Evidence

**train_ternary_v5_10.py:531-536**
```python
monitor = TrainingMonitor(
    eval_num_samples=config.get('eval_num_samples', 1000),
    tensorboard_dir=config.get('tensorboard_dir', 'runs'),
    log_dir=args.log_dir,
    log_to_file=True
)
```

**src/training/trainer.py:90-94**
```python
self.monitor = TrainingMonitor(
    eval_num_samples=config['eval_num_samples'],
    tensorboard_dir=config.get('tensorboard_dir'),
    experiment_name=config.get('experiment_name')
)
```

### Impact

- Two log files created with different timestamps
- Two TensorBoard directories with partial metrics
- `base_trainer.monitor` tracks coverage/entropy history
- Script `monitor` tracks hyperbolic correlation history
- Checkpoint metadata inconsistent between instances

---

## 2. Logging Fragmentation

### Direct Print Statements (Bypassing File Logging)

| File | Count | Examples |
|------|-------|----------|
| trainer.py | 18 | torch.compile status, init summary, early stopping |
| appetitive_trainer.py | 18 | Same patterns as trainer.py |
| monitor.py | 0 | Uses `_log()` abstraction |
| train_ternary_v5_10.py | 0 | Uses monitor methods |

**Total: 36 print statements in core training that bypass file logging**

### Inconsistent Logging Methods

```python
# In monitor.py - 6 different logging methods:
_log()                  # Internal abstraction
log_batch()             # Batch-level metrics
log_hyperbolic_batch()  # v5.10 batch metrics
log_hyperbolic_epoch()  # v5.10 epoch metrics
log_epoch_summary()     # Comprehensive epoch summary
log_epoch()             # Legacy epoch logging
log_tensorboard()       # TensorBoard-specific
```

### Usage Split

- `TernaryVAETrainer` uses `monitor.log_epoch()`
- `PureHyperbolicTrainer` uses `monitor.log_epoch_summary()` + `monitor.log_hyperbolic_epoch()`
- Neither uses a unified interface

---

## 3. Configuration Validation (Non-Existent)

### Current Loading Pattern

```python
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
```

### Missing Validations

| Validation | Status | Risk |
|------------|--------|------|
| Required keys exist | MISSING | KeyError on missing config |
| Value types correct | MISSING | Silent type coercion bugs |
| Value ranges valid | MISSING | Invalid hyperparameters |
| Interdependency checks | MISSING | Conflicting settings |
| Config version compatibility | MISSING | Breaking changes undetected |
| File existence check | MISSING | Unhelpful FileNotFoundError |

### Example Inconsistency

```python
# Inconsistent access patterns in trainer.py:
lr=config['optimizer']['lr_start'],           # Direct access - crashes if missing
weight_decay=config['optimizer'].get('weight_decay', 0.0001)  # Safe access with default
```

---

## 4. Pre-Training Validation (Missing)

### Current Initialization Sequence

1. Argument parsing (no validation of paths)
2. Config loading (no schema validation)
3. Monitor creation (before any validation)
4. Seed setup (correct)
5. Device detection (assumes CUDA available)
6. Dataset generation (no integrity check)
7. Model initialization (no parameter validation)
8. Training start

### Missing Checks

| Check | Impact if Missing |
|-------|-------------------|
| CUDA availability | Crash on CPU-only systems |
| Disk space for checkpoints | Training fails mid-run |
| Log directory permissions | Silent logging failure |
| TensorBoard availability | Metrics not recorded |
| Config schema validity | Silent failures |
| Model parameter ranges | Training instability |
| Dataset integrity | Corrupted results |

---

## 5. Error Handling Patterns

### Current State

| Component | Pattern | Quality |
|-----------|---------|---------|
| torch.compile | try/except with fallback | Good |
| Config loading | No error handling | Poor |
| File I/O | Assumes success | Poor |
| Dataset generation | No validation | Poor |
| CUDA detection | `'cuda' if available else 'cpu'` | Minimal |
| Gradient clipping | No overflow handling | Medium |

### Example - Config Error Propagation

```python
# If config['model']['input_dim'] missing:
# 1. KeyError raised at line 616
# 2. Stack trace points to dict access, not config issue
# 3. User must debug to find missing key
```

---

## 6. Architectural Debt Summary

### Severity Matrix

| Issue | Severity | Effort to Fix | Priority |
|-------|----------|---------------|----------|
| Dual monitor instances | HIGH | Medium | P0 |
| Print statements bypass logging | MEDIUM | Low | P1 |
| No config validation | MEDIUM | Medium | P1 |
| Inconsistent logging API | MEDIUM | Medium | P2 |
| Missing error handling | HIGH | Medium | P1 |
| No pre-training validation | MEDIUM | Low | P2 |

### Root Cause

Incremental feature additions without architectural review:
- v5.6: Base trainer with monitor
- v5.7: Added StateNet, more logging
- v5.9: Added hyperbolic losses
- v5.10: Added "unified" monitor on top of existing one
- v5.10.1: Patched observability without refactoring

---

## 7. Recommended Architecture

### Target State

```
train_ternary_v5_10.py (Entry)
    |
    +-- validate_config(config_path) ─────> ConfigValidationError if invalid
    |
    +-- validate_environment() ───────────> EnvironmentError if CUDA/disk/perms fail
    |
    +-- Creates: TernaryVAETrainer(config, monitor=None)
            |
            +-- If monitor=None: Creates internal monitor
            +-- If monitor provided: Uses injected monitor
            |
            +-- Single TrainingMonitor instance
                    |
                    +-- _log() ──────────> Console + File
                    +-- log_tensorboard() ──────────> TensorBoard
                    +-- log_batch() / log_epoch() ──> Unified API
```

### Key Changes Required

1. **Monitor injection** - `TernaryVAETrainer.__init__(config, device, monitor=None)`
2. **Config validation** - Pydantic/dataclass schema before training
3. **Environment validation** - Check CUDA, disk, permissions upfront
4. **Unified logging** - Single `log()` method with level/category
5. **Remove prints** - Replace with `monitor._log()` calls
6. **Error handling** - Explicit try/except with actionable messages

---

## 8. Files Requiring Modification

| File | Changes Needed |
|------|----------------|
| `src/training/trainer.py` | Accept optional monitor, remove prints |
| `src/training/monitor.py` | Simplify API, add validation methods |
| `src/training/appetitive_trainer.py` | Same as trainer.py |
| `scripts/train/train_ternary_v5_10.py` | Add validation, pass monitor to trainer |
| `configs/ternary_v5_10.yaml` | Add schema version field |
| NEW: `src/training/config_schema.py` | Pydantic config validation |
| NEW: `src/training/environment.py` | Pre-training environment checks |

---

## 9. Metrics

### Current State

| Metric | Value |
|--------|-------|
| Monitor instances per training run | 2 |
| Print statements bypassing file log | 36 |
| Config validation checks | 0 |
| Pre-training environment checks | 1 (CUDA) |
| Logging methods in monitor.py | 19 |
| Error handling coverage | ~15% |

### Target State

| Metric | Target |
|--------|--------|
| Monitor instances per training run | 1 |
| Print statements bypassing file log | 0 |
| Config validation checks | 60+ (all keys) |
| Pre-training environment checks | 5+ |
| Logging methods in monitor.py | 5-7 (simplified) |
| Error handling coverage | 90%+ |

---

## 10. Next Steps

1. **Do not implement yet** - This report documents current state
2. **Review with stakeholder** - Validate priorities
3. **Create implementation plan** - Phased approach to avoid breaking training
4. **Test coverage** - Add tests before refactoring
5. **Incremental migration** - One component at a time

---

**Report Status:** Complete - Awaiting review before implementation
