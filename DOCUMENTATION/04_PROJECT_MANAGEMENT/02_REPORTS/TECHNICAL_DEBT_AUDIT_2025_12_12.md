# Codebase Technical Debt Audit

**Doc-Type:** Technical Debt Inventory · Version 3.0 · Updated 2025-12-12

---

## Overview

Comprehensive audit of the Ternary VAE v5.10 codebase covering performance bottlenecks, dead code, incorrect logic, wasted computation, and vectorization opportunities across all 32 src/ files.

---

## Audit Progress

### Files Read: 32/32 (COMPLETE)

| Module            | Files | Status       | Verified This Session |
| :---------------- | :---- | :----------- | :-------------------- |
| `src/__init__.py` | 1     | **VERIFIED** | Yes - Clean           |
| `src/data/`       | 4     | **VERIFIED** | Yes - 1 issue found   |
| `src/models/`     | 5     | **VERIFIED** | Yes - Clean           |
| `src/losses/`     | 7     | **VERIFIED** | Yes - 10 issues found |
| `src/training/`   | 8     | **VERIFIED** | Yes - 6 issues found  |
| `src/metrics/`    | 2     | **VERIFIED** | Yes - Clean           |
| `src/artifacts/`  | 2     | **VERIFIED** | Yes - Clean           |
| `src/utils/`      | 3     | **VERIFIED** | Yes - 2 issues found  |

---

## VERIFIED Issues (This Session)

### Category 1: Performance Bottlenecks

#### P1.1 - O(n²) 3-Adic Distance Matrix (VERIFIED)

- **File:** `src/losses/padic_losses.py`
- **Lines:** 35-48 (inside `_get_3adic_distance_matrix()`)
- **Issue:** Nested Python loops compute 19683×19683 matrix
- **Code Found:**

```python
for i in range(n):
    for j in range(i + 1, n):
        diff = abs(i - j)
        if diff == 0:
            dist = 0.0
        else:
            v = 0
            while diff % 3 == 0:
                v += 1
                diff //= 3
            dist = 3.0 ** (-v)
        distances[i, j] = dist
        distances[j, i] = dist
```

- **Impact:** Blocks training startup for several seconds
- **Fix:** Vectorize or precompute to disk

#### P1.2 - O(n²) Poincaré Distance Loop (VERIFIED)

- **File:** `src/losses/padic_losses.py`
- **Lines:** 834-838 (inside `_mine_hard_negatives_hyperbolic()`)
- **Issue:** Python loop to compute pairwise Poincaré distances
- **Code Found:**

```python
d_poincare_matrix = torch.zeros(batch_size, batch_size, device=device)
for i in range(batch_size):
    d_poincare_matrix[i] = self._poincare_distance(
        z_hyp[i:i+1].expand(batch_size, -1), z_hyp
    )
```

- **Impact:** Slows hard negative mining
- **Fix:** Use torch.cdist or vectorized Poincaré distance

#### P1.3 - Valuation Loop in \_mine_hard_negatives_hyperbolic (VERIFIED)

- **File:** `src/losses/padic_losses.py`
- **Lines:** 847-857
- **Issue:** Python loop computing valuations one at a time
- **Code Found:**

```python
v_to_all = torch.zeros(batch_size, device=device)
for j in range(batch_size):
    diff = abs(int(anchor_idx_val) - int(batch_indices[j]))
    if diff == 0:
        v_to_all[j] = 9.0
    else:
        v = 0
        while diff % 3 == 0 and diff > 0:
            v += 1
            diff //= 3
        v_to_all[j] = v
```

- **Impact:** O(batch_size²) inner loop
- **Fix:** Use `compute_3adic_valuation_batch()` which already exists

---

### Category 2: Dead/Wasted Code

#### D2.1 - Origin Buffer Registered But Recreated (VERIFIED)

- **File:** `src/losses/hyperbolic_prior.py`
- **Lines:** 67 vs 174
- **Issue:** Buffer registered at init but recreated dynamically
- **Code Found:**

```python
# Line 67: Registered but unused
self.register_buffer('origin', torch.zeros(latent_dim))

# Line 174: Actually used (for CUDA compatibility)
origin = torch.zeros_like(z_mu)
```

- **Impact:** Minor wasted memory
- **Fix:** Remove line 67

#### D2.2 - val_losses Computed But Discarded (VERIFIED)

- **File:** `src/training/hyperbolic_trainer.py`
- **Line:** 290
- **Issue:** Calls `validate()` but discards result
- **Code Found:**

```python
# Line 290: val_losses computed
val_losses = self.base_trainer.validate(val_loader)
# val_losses NEVER appears in the return dict (lines 342-370)
# Only train_losses is unpacked: return {**train_losses, ...}
```

- **Impact:** Wasted computation every epoch
- **Fix:** Either use val_losses in return dict OR guard the call for manifold approach

---

### Category 3: Incorrect Logic

#### L3.1 - Broken Operation Composition (CRITICAL - VERIFIED)

- **File:** `src/losses/appetitive_losses.py`
- **Lines:** 428-486 (`compose_operations()`)
- **Issue:** Code comments explicitly admit confusion about composition logic
- **Code Found:**

```python
# Lines 463-473 contain these comments:
# "This doesn't quite work - let me reconsider"
# "Actually for LUT composition... this isn't well-defined"
# "Let's use a simpler interpretation..."

# The implementation at lines 478-484:
for j in range(batch_size):
    select_idx = (b_lut[j, i] + 1).item()  # 0, 1, or 2
    composed_lut[j, i] = a_lut[j, select_idx * 3 + i % 3]
```

- **Impact:** `AlgebraicClosureLoss` produces meaningless gradients
- **Fix:** Rewrite composition logic or disable until fixed

#### L3.2 - Trivial Addition Accuracy Test (VERIFIED)

- **File:** `src/losses/consequence_predictor.py`
- **Lines:** 219-229 (`evaluate_addition_accuracy()`)
- **Issue:** Tests `z_A + z_0 - z_0 ≈ z_A` which is trivially true by definition
- **Code Found:**

```python
# Line 222: Trivial computation
z_predicted = z_A + z_0 - z_0  # Always equals z_A

# Lines 225-229: Error measurement that's always ~0
errors = torch.norm(z_predicted - z_A, dim=1)
threshold = 0.1
accuracy = (errors < threshold).float().mean().item()
```

- **Impact:** Test always passes ~100%, provides no signal
- **Fix:** Implement actual composition test

---

### Category 4: Hardcoded Values

#### H4.1 - Hardcoded kl_target (VERIFIED)

- **File:** `src/losses/hyperbolic_prior.py`
- **Line:** 353
- **Code:** `kl_target = 1.0  # Target KL in nats`
- **Fix:** Make configurable via constructor

#### H4.2 - Hardcoded EMA Alpha (VERIFIED)

- **File:** `src/losses/hyperbolic_prior.py`
- **Line:** 334
- **Code:** `alpha = 0.1`
- **Fix:** Make configurable via constructor

#### H4.3 - Hardcoded Target Radius (VERIFIED)

- **File:** `src/losses/hyperbolic_prior.py`
- **Line:** 320 (function parameter default)
- **Code:** `target_radius: float = 0.5`
- **Fix:** Consider making adaptive

#### H4.4 - Fixed Frechet Mean Iterations (VERIFIED)

- **File:** `src/losses/hyperbolic_recon.py`
- **Line:** 456 (parameter default) and 481 (loop)
- **Code:** `n_iter: int = 5` and `for _ in range(n_iter):`
- **Impact:** May not converge for some cluster configurations
- **Fix:** Add convergence check or make configurable

#### H4.5 - Hardcoded TODO Placeholder (VERIFIED)

- **File:** `src/training/appetitive_trainer.py`
- **Line:** 593
- **Code:** `'addition_accuracy': 0.0  # TODO: Implement addition accuracy evaluation`
- **Impact:** Metric always reports 0.0, provides no signal
- **Fix:** Implement actual addition accuracy or remove from metrics

---

### Category 5: Silent Failures / Division by Zero

#### S5.1 - Empty val_loader Silent Failure (VERIFIED)

- **File:** `src/data/loaders.py`
- **Lines:** 78-84
- **Issue:** Always creates val_loader even when val_size=0
- **Code Found:**

```python
# Lines 78-84: Always creates DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
```

- **Contrast with test_loader (lines 86-94):**

```python
# Returns None when test_size=0
test_loader = None
if test_size > 0:
    test_loader = DataLoader(...)
```

- **Impact:** Downstream code may assume batches exist
- **Fix:** Return None when val_size=0

#### S5.2 - Division by Zero in trainer.py validate() (VERIFIED)

- **File:** `src/training/trainer.py`
- **Lines:** 362-363
- **Issue:** No guard before division by num_batches
- **Code Found:**

```python
# Lines 362-363: Division without guard
for key in epoch_losses:
    epoch_losses[key] /= num_batches  # CRASH if num_batches=0
```

- **Impact:** Crashes if val_loader has 0 batches (manifold approach)
- **Fix:** Guard with `if num_batches > 0:`

#### S5.3 - Division by Zero in appetitive_trainer.py train_epoch() (VERIFIED)

- **File:** `src/training/appetitive_trainer.py`
- **Lines:** 408-410
- **Issue:** Same division bug as trainer.py
- **Code Found:**

```python
# Lines 408-410: Division without guard
for key in epoch_losses:
    if key not in ['lr_corrected', 'delta_lr', ...]:
        epoch_losses[key] /= num_batches  # CRASH if num_batches=0
```

- **Impact:** Crashes if train_loader empty
- **Fix:** Guard with `if num_batches > 0:`

#### S5.4 - Division by Zero in appetitive_trainer.py validate() (VERIFIED)

- **File:** `src/training/appetitive_trainer.py`
- **Lines:** 475-476
- **Issue:** Same division bug in validate()
- **Code Found:**

```python
# Lines 475-476: Division without guard
for key in epoch_losses:
    epoch_losses[key] /= num_batches  # CRASH if num_batches=0
```

- **Impact:** Crashes if val_loader has 0 batches
- **Fix:** Guard with `if num_batches > 0:`

#### S5.5 - Unconditional val_loader Access (VERIFIED)

- **File:** `src/training/appetitive_trainer.py`
- **Line:** 541
- **Issue:** Uses val_loader without checking if None
- **Code Found:**

```python
# Line 541: Unconditional call
val_losses = self.validate(val_loader)  # val_loader could be None
```

- **Impact:** Crashes in manifold approach where val_loader=None
- **Fix:** Guard with `if val_loader is not None:`

---

### Category 6: Vectorization Opportunities

#### V6.1 - 3-Adic Valuation Batch Uses Loop (VERIFIED)

- **File:** `src/losses/padic_losses.py`
- **Lines:** 119-124 (inside `compute_3adic_valuation_batch()`)
- **Issue:** Uses Python `for` loop with limited iterations
- **Code Found:**

```python
for _ in range(9):  # Max 9 digits in base-3 for 19683
    divisible = (remaining % 3 == 0)
    if not divisible.any():
        break
    v[divisible] += 1
    remaining[divisible] = remaining[divisible] // 3
```

- **Note:** This is partially vectorized (tensor operations inside loop)
- **Impact:** Acceptable for now, but could be fully vectorized with bit operations
- **Priority:** Low

---

### Category 7: Configuration Constraints (Not Bugs - Important for Manifold)

#### C7.1 - Splits Must Sum to 1.0 (VERIFIED - NOT A BUG)

- **File:** `src/training/config_schema.py`
- **Lines:** 353-359
- **Code Found:**

```python
splits_sum = (
    raw_config.get('train_split', 0.8) +
    raw_config.get('val_split', 0.1) +
    raw_config.get('test_split', 0.1)
)
if abs(splits_sum - 1.0) > 0.001:
    errors.append(f"Data splits must sum to 1.0, got {splits_sum}")
```

- **Impact for Manifold:** Must use `train_split=1.0, val_split=0.0, test_split=0.0`
- **Note:** This is correct validation, not a bug

---

### Category 8: Incorrect Approximations

#### A8.1 - Wrong Union Approximation (VERIFIED)

- **File:** `src/utils/metrics.py`
- **Line:** 220 (inside `CoverageTracker.update()`)
- **Issue:** Uses `max()` instead of actual set union
- **Code Found:**

```python
# Line 220: Wrong approximation
union = max(coverage_A, coverage_B)  # Approximate union
self.history['coverage_union'].append(union)
```

- **Impact:** Underestimates true union coverage (if A covers ops 1-100 and B covers ops 50-150, max=100 but union=150)
- **Fix:** Track actual set intersection and compute: `union = coverage_A + coverage_B - intersection`

---

### Category 9: Vectorization Opportunities

#### V9.1 - Python Loops for Set Conversion (VERIFIED)

- **File:** `src/utils/metrics.py`
- **Lines:** 27-29, 99-103
- **Issue:** Uses Python `for` loop to convert samples to set of tuples
- **Code Found:**

```python
# Lines 27-29 in evaluate_coverage():
for i in range(samples_rounded.shape[0]):
    lut = tuple(samples_rounded[i].cpu().tolist())
    unique_ops.add(lut)

# Lines 99-103 in compute_diversity_score():
for i in range(samples_A_rounded.shape[0]):
    ops_A.add(tuple(samples_A_rounded[i].cpu().tolist()))
for i in range(samples_B_rounded.shape[0]):
    ops_B.add(tuple(samples_B_rounded[i].cpu().tolist()))
```

- **Impact:** Slow for large sample sizes
- **Priority:** Low (not critical path)

---

### Category 10: Useful Utilities (Not Issues - Opportunities)

#### U10.1 - has_plateaued() Exists But Unused (VERIFIED)

- **File:** `src/utils/metrics.py`
- **Lines:** 260-276 (inside `CoverageTracker`)
- **Code Found:**

```python
def has_plateaued(self, patience: int = 50, min_delta: float = 0.001) -> bool:
    """Check if coverage has plateaued."""
    if len(self.history['coverage_union']) < patience:
        return False
    recent = self.history['coverage_union'][-patience:]
    improvement = (recent[-1] - recent[0]) / 19683
    return improvement < min_delta
```

- **Impact:** Could be used for manifold approach plateau detection (100% coverage reached)
- **Recommendation:** Integrate into manifold training early stopping logic

---

## Verified Clean Files (No Issues Found)

| File                                  | Lines | Notes                          |
| :------------------------------------ | :---- | :----------------------------- |
| `src/__init__.py`                     | 51    | Package init, clean            |
| `src/data/__init__.py`                | 28    | Exports only                   |
| `src/data/dataset.py`                 | 80    | TernaryOperationDataset, clean |
| `src/data/generation.py`              | 63    | Data generation, clean         |
| `src/models/__init__.py`              | 17    | Exports only                   |
| `src/models/ternary_vae_v5_6.py`      | 539   | Legacy v5.6 model, clean       |
| `src/models/ternary_vae_v5_7.py`      | 626   | StateNet v3, clean             |
| `src/models/ternary_vae_v5_10.py`     | 823   | StateNet v4 hyperbolic, clean  |
| `src/models/appetitive_vae.py`        | 312   | Wrapper, clean                 |
| `src/losses/__init__.py`              | 92    | Exports only                   |
| `src/losses/dual_vae_loss.py`         | 525   | Loss aggregator, clean         |
| `src/training/__init__.py`            | 61    | Exports only                   |
| `src/training/schedulers.py`          | 212   | Parameter scheduling, clean    |
| `src/training/environment.py`         | 238   | Environment validation, clean  |
| `src/training/monitor.py`             | 695   | Training monitor, clean        |
| `src/training/config_schema.py`       | 434   | Config validation, clean       |
| `src/metrics/__init__.py`             | 24    | Exports only                   |
| `src/metrics/hyperbolic.py`           | 210   | Poincaré metrics, clean        |
| `src/artifacts/__init__.py`           | 14    | Exports only                   |
| `src/artifacts/checkpoint_manager.py` | 137   | Checkpoint I/O, clean          |
| `src/utils/__init__.py`               | 29    | Exports only                   |
| `src/utils/reproducibility.py`        | 52    | Seed management, clean         |

---

## Priority Matrix (All Verified Issues)

| Priority | Category | Issue                         | File                     | Lines        | Effort | Impact      |
| :------- | :------- | :---------------------------- | :----------------------- | :----------- | :----- | :---------- |
| P0       | Logic    | L3.1 Broken Composition       | appetitive_losses.py     | 428-486      | High   | Critical    |
| P0       | Silent   | S5.2 Division by Zero         | trainer.py               | 362-363      | Low    | Critical    |
| P0       | Silent   | S5.3 Division by Zero         | appetitive_trainer.py    | 408-410      | Low    | Critical    |
| P0       | Silent   | S5.4 Division by Zero         | appetitive_trainer.py    | 475-476      | Low    | Critical    |
| P0       | Silent   | S5.5 Unconditional val_loader | appetitive_trainer.py    | 541          | Low    | Critical    |
| P1       | Perf     | P1.1 O(n²) Distance Matrix    | padic_losses.py          | 35-48        | Medium | High        |
| P1       | Logic    | L3.2 Trivial Addition Test    | consequence_predictor.py | 219-229      | Medium | High        |
| P1       | Logic    | A8.1 Wrong Union Approx       | utils/metrics.py         | 220          | Low    | High        |
| P2       | Perf     | P1.2 O(n²) Poincaré Loop      | padic_losses.py          | 834-838      | Medium | Medium      |
| P2       | Perf     | P1.3 Valuation Inner Loop     | padic_losses.py          | 847-857      | Low    | Medium      |
| P2       | Silent   | S5.1 Empty val_loader         | loaders.py               | 78-84        | Low    | Medium      |
| P2       | Dead     | D2.2 Wasted val_losses        | hyperbolic_trainer.py    | 290          | Low    | Medium      |
| P3       | Dead     | D2.1 Unused Origin Buffer     | hyperbolic_prior.py      | 67           | Low    | Low         |
| P3       | Config   | H4.1-4 Hardcoded Values       | hyperbolic_prior.py      | 320,334,353  | Low    | Low         |
| P3       | Config   | H4.4 Fixed Frechet Iter       | hyperbolic_recon.py      | 456,481      | Low    | Low         |
| P3       | Config   | H4.5 Hardcoded TODO           | appetitive_trainer.py    | 593          | Low    | Low         |
| P3       | Perf     | V9.1 Python Loop Sets         | utils/metrics.py         | 27-29,99-103 | Low    | Low         |
| P4       | Feature  | U10.1 Unused Plateau Check    | utils/metrics.py         | 260-276      | Low    | Opportunity |

---

## Session Checkpoint

**Last Updated:** 2025-12-12 (Current Session)
**Files Verified This Session:** 32/32 (COMPLETE)

- src/**init**.py: 1 file
- src/data/: 4 files
- src/models/: 5 files
- src/losses/: 7 files
- src/training/: 8 files
- src/metrics/: 2 files
- src/artifacts/: 2 files
- src/utils/: 3 files

**Verified Issues Summary:**

- **P0 Critical:** 5 (broken composition, 4 division-by-zero bugs)
- **P1 High:** 3 (O(n²) matrix, trivial test, wrong union)
- **P2 Medium:** 4 (loops, val_loader, wasted compute)
- **P3 Low:** 7 (buffer, hardcoded values, TODO, python loops)
- **P4 Opportunity:** 1 (unused plateau detection)

**Total:** 20 issues across 10 files (22 clean files)

**Recommended Fix Order:**

1. Fix P0 issues before any training run (crashes, wrong gradients)
2. Fix A8.1 wrong union if using coverage tracking
3. Consider P1 for manifold approach reliability
4. Schedule P2-P4 for future sprints

---

## Changelog

| Date       | Version | Description                                                                          |
| :--------- | :------ | :----------------------------------------------------------------------------------- |
| 2025-12-12 | 3.0     | COMPLETE: All 32/32 files verified. Added A8.1, V9.1, U10.1 from final 7 files.      |
| 2025-12-12 | 2.1     | Verified src/training/ (8 files): found S5.2-S5.5, D2.2, H4.5. Total 25/32 verified. |
| 2025-12-12 | 2.0     | Fresh audit: verified 17/32 files, documented 11 issues with exact line numbers      |
| 2025-12-12 | 1.1     | (Previous session) Claimed 32/32 but based on compacted context                      |
| 2025-12-12 | 1.0     | (Previous session) Initial audit                                                     |
