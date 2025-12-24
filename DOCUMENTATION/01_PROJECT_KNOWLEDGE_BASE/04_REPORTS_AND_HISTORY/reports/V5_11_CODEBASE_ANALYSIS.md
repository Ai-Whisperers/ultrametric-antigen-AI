# V5.11.3 Codebase Analysis

**Doc-Type:** Technical Analysis · Version 0.5 · Updated 2025-12-19 · Status: COMPLETE

---

## Purpose

Document the actual active vs legacy components in the ternary-vaes codebase to enable clean consolidation before v5.11.4 improvements.

**Key Finding:** The codebase has THREE parallel training approaches that share code through `__init__.py` exports, creating hidden dependency chains.

---

## 1. Three Parallel Training Approaches

### Approach A: Inline (v5.11 Production)
```
scripts/train/train.py
    → src.models.TernaryVAEV5_11
    → src.losses.PAdicGeodesicLoss, RadialHierarchyLoss, GlobalRankLoss
    → src.data.generation.generate_all_ternary_operations
    → src.core.TERNARY
```
**Status:** ACTIVE - Used for production v5.11.3 model

### Approach B: TernaryVAETrainer (v5.6-v5.10)
```
scripts/train/archive/train_ternary_v5_*.py
    → src.training.TernaryVAETrainer
        → src.losses.DualVAELoss
            → src.losses.padic_losses (PAdicMetricLoss, PAdicRankingLoss, etc.)
            → src.losses.hyperbolic_prior (HyperbolicPrior, HomeostaticHyperbolicPrior)
            → src.losses.hyperbolic_recon (HyperbolicReconLoss, etc.)
        → src.training.TemperatureScheduler, BetaScheduler
        → src.training.TrainingMonitor
        → src.artifacts.CheckpointManager
```
**Status:** LEGACY - Archived scripts, but code still maintained for backward compat

### Approach C: LossRegistry Pattern (Designed but Not Integrated)
```
[NOT USED IN ANY TRAINING SCRIPT]
    → src.losses.LossRegistry
    → src.losses.components (ReconstructionLossComponent, etc.)
        → src.losses.padic_losses (PAdicRankingLossV2, PAdicRankingLossHyperbolic)
```
**Status:** STRUCTURAL DEBT - Complete implementation, never integrated

---

## 2. Dependency Chain Analysis

### Chain 1: DualVAELoss pulls in ALL legacy losses
```python
# src/losses/dual_vae_loss.py lines 21-23
from .padic_losses import PAdicMetricLoss, PAdicRankingLoss, PAdicRankingLossV2, PAdicRankingLossHyperbolic, PAdicNormLoss
from .hyperbolic_prior import HyperbolicPrior, HomeostaticHyperbolicPrior
from .hyperbolic_recon import HyperbolicReconLoss, HomeostaticReconLoss, HyperbolicCentroidLoss
```

### Chain 2: TernaryVAETrainer requires DualVAELoss
```python
# src/training/trainer.py line 23
from ..losses import DualVAELoss, RadialStratificationLoss
```

### Chain 3: components.py wraps legacy losses
```python
# src/losses/components.py lines 19-22
from .padic_losses import (
    PAdicRankingLossV2,
    PAdicRankingLossHyperbolic
)
```

### Chain 4: __init__.py exports EVERYTHING
```python
# src/losses/__init__.py - exports 40+ classes from 9 modules
# All get imported at `from src.losses import ...` time
```

**Impact:** Even if `train.py` only imports 4 classes, the `__init__.py` eagerly loads ALL loss modules including legacy ones.

### Chain 5: Top-Level src/__init__.py Pulls Legacy Trainers
```python
# src/__init__.py lines 29-33
from .training import (
    TernaryVAETrainer,
    HyperbolicVAETrainer,
    TrainingMonitor
)
```
**Impact:** `import src` or `from src import TernaryVAE` triggers import of ALL training components including legacy trainers.

---

## 3. Duplicated Functionality

### 3.1 poincare_distance (2 Implementations)

| Location | Lines | Used By |
|----------|-------|---------|
| `src/metrics/hyperbolic.py:36` | 25 | `compute_ranking_correlation_hyperbolic()` |
| `src/losses/padic_geodesic.py:29` | 30 | `PAdicGeodesicLoss`, `GlobalRankLoss` |

**Problem:** Two slightly different implementations of the same hyperbolic distance formula. No shared utility.

### 3.2 Valuation Computation (3 Implementations)

| Location | Function | Used By |
|----------|----------|---------|
| `src/core/ternary.py:118` | `TernarySpace.valuation()` | Uses precomputed LUT (canonical) |
| `src/losses/padic_losses.py:46` | `compute_3adic_valuation_batch()` | Legacy losses |
| `src/metrics/hyperbolic.py:64` | `compute_3adic_valuation()` | Evaluation metrics |

**Problem:** Three implementations, only `src/core/ternary.py` uses the optimized LUT. Others compute on-the-fly.

### 3.3 project_to_poincare (2 Implementations)

| Location | Used By |
|----------|---------|
| `src/metrics/hyperbolic.py:19` | Evaluation |
| `src/models/hyperbolic_projection.py` | Model forward pass |

**Problem:** Similar projection logic in two places with different parameterization.

---

## 4. Active vs Legacy Scripts

### Active Scripts (Not in archive/)

| Script | Imports | Uses Trainer? |
|--------|---------|---------------|
| `scripts/train/train.py` | v5.11 models, geodesic losses | NO - inline loop |
| `scripts/eval/downstream_validation.py` | TernaryVAEV5_11_OptionC, poincare_distance | NO |
| `scripts/analysis/compare_options.py` | v5.11 models, poincare_distance | NO |
| `research/spectral_analysis/01_extract_embeddings.py` | TernaryVAEV5_11_OptionC | NO |

### Archived Scripts (in archive/)

| Script | Imports | Uses Trainer? |
|--------|---------|---------------|
| `train_ternary_v5_6.py` | TernaryVAETrainer | YES |
| `train_ternary_v5_7.py` | DualVAELoss | YES |
| `train_ternary_v5_8.py` | TernaryVAETrainer | YES (BROKEN - wrong model) |
| `train_ternary_v5_9.py` | TernaryVAETrainer | YES (BROKEN - wrong model) |
| `train_ternary_v5_9_1.py` | TernaryVAETrainer, DualVAELoss | YES |
| `train_ternary_v5_10.py` | Full infrastructure | YES |
| `train_appetitive_vae.py` | AppetitiveVAETrainer | YES |
| `train_purposeful.py` | TernaryVAETrainer, ConsequencePredictor | YES |

---

## 5. Backward Compatibility Mechanisms

### 5.1 Model Aliases (src/models/__init__.py)
```python
# Lines 9-10
TernaryVAE = TernaryVAEV5_11
TernaryVAE_OptionC = TernaryVAEV5_11_OptionC
```
**Purpose:** Allow `from src.models import TernaryVAE` without version suffix

### 5.2 Loss Exports (src/losses/__init__.py)
```python
# Exports 40+ classes including:
# - Legacy: DualVAELoss, PAdicMetricLoss, PAdicRankingLoss, etc.
# - V5.11: PAdicGeodesicLoss, RadialHierarchyLoss, GlobalRankLoss
# - Registry: LossRegistry, LossComponent, etc.
```
**Purpose:** Any code can import any loss without knowing which version to use
**Problem:** Eager loading pulls in all dependencies even if unused

### 5.3 Training Exports (src/training/__init__.py)
```python
# Exports: TernaryVAETrainer, HyperbolicVAETrainer, schedulers, monitors
```
**Purpose:** Archived scripts continue to work
**Problem:** Maintains code that v5.11 doesn't use

---

## 6. Documentation Drift

The following docs reference LEGACY classes not used in v5.11:

| Document | References |
|----------|------------|
| `README.md` | TernaryVAETrainer, DualVAELoss, schedulers |
| `API_REFERENCE.md` | DualNeuralVAEV5_10, StateNetV4, HyperbolicPrior, all legacy losses |
| `INSTALLATION_AND_USAGE.md` | DualNeuralVAEV5_10, TernaryVAETrainer |
| `src/README.md` | Entire wiring diagram references v5.10 models, TernaryVAETrainer, HyperbolicVAETrainer |

**Impact:** New users may try to use documented patterns that aren't the production path.

### 6.2 src/README.md is Completely Outdated

The internal architecture doc `src/README.md` references:
- `DualNeuralVAEV5_10` - in archive
- `ternary_vae_v5_6.py`, `ternary_vae_v5_7.py` - in archive
- `TernaryVAETrainer` as primary orchestrator - legacy
- `HyperbolicVAETrainer` - legacy
- Wiring diagram shows v5.10 architecture, not v5.11

**This document needs complete rewrite or deletion.**

---

## 7. Consolidated Active Components

### Required for v5.11.3 Production

| File | Lines | Role |
|------|-------|------|
| `scripts/train/train.py` | ~800 | Training entry point (inline loop) |
| `src/models/ternary_vae.py` | 609 | TernaryVAEV5_11, FrozenEncoder, FrozenDecoder |
| `src/models/hyperbolic_projection.py` | 288 | HyperbolicProjection, DualHyperbolicProjection |
| `src/models/differentiable_controller.py` | 301 | DifferentiableController, ThreeBodyController |
| `src/models/curriculum.py` | 217 | ContinuousCurriculumModule (unused in production) |
| `src/losses/padic_geodesic.py` | 681 | PAdicGeodesicLoss, RadialHierarchyLoss, GlobalRankLoss, MonotonicRadialLoss |
| `src/data/generation.py` | 60 | generate_all_ternary_operations |
| `src/core/ternary.py` | 379 | TERNARY singleton, valuation LUTs |
| `configs/ternary.yaml` | 105 | Production config |

**Total Active:** ~3,440 lines

### Backward Compatibility Layer (Needed for archived scripts to work)

| File | Lines | What it enables |
|------|-------|-----------------|
| `src/__init__.py` | 53 | Top-level exports with legacy trainers |
| `src/losses/__init__.py` | 162 | Legacy loss imports (40+ classes) |
| `src/models/__init__.py` | 30 | Model aliases |
| `src/training/__init__.py` | 58 | Trainer exports |
| `src/training/trainer.py` | 677 | TernaryVAETrainer |
| `src/losses/dual_vae_loss.py` | 528 | DualVAELoss aggregator |

**Total Compat Layer:** ~1,508 lines

### Legacy (Only for archived scripts)

| File | Lines | Pulled in by |
|------|-------|--------------|
| `src/losses/padic_losses.py` | 969 | DualVAELoss, components.py |
| `src/losses/hyperbolic_prior.py` | 390 | DualVAELoss |
| `src/losses/hyperbolic_recon.py` | 554 | DualVAELoss |
| `src/losses/appetitive_losses.py` | 599 | appetitive pathway |
| `src/losses/radial_stratification.py` | 161 | trainer.py |
| `src/losses/consequence_predictor.py` | 341 | purposeful pathway |
| `src/training/schedulers.py` | 178 | TernaryVAETrainer |
| `src/training/monitor.py` | 879 | TernaryVAETrainer |
| `src/training/hyperbolic_trainer.py` | 1,033 | Archived scripts |
| `src/training/archive/appetitive_trainer.py` | 674 | Archived scripts |
| `src/models/archive/ternary_vae_v5_6.py` | 538 | Archived scripts |
| `src/models/archive/ternary_vae_v5_7.py` | 625 | Archived scripts |
| `src/models/archive/ternary_vae_v5_10.py` | 1,266 | Archived scripts |
| `src/models/archive/appetitive_vae.py` | 311 | Archived scripts |

**Total Legacy:** ~8,518 lines

### Structural (Complete but not integrated)

| File | Lines | Purpose |
|------|-------|---------|
| `src/losses/base.py` | 225 | LossComponent protocol |
| `src/losses/registry.py` | 376 | LossRegistry pattern |
| `src/losses/components.py` | 593 | Loss wrappers |
| `src/training/config_schema.py` | 441 | Typed config validation |
| `src/training/environment.py` | 237 | Environment validation |

**Total Structural:** ~1,872 lines

---

## 8. Summary

**Total src/ lines: 16,701** (from `wc -l src/**/*.py`)

| Category | Lines | % of src/ |
|----------|-------|-----------|
| Active (v5.11.3) | ~3,440 | 21% |
| Compat Layer | ~1,508 | 9% |
| Legacy | ~8,518 | 51% |
| Structural (unused) | ~1,872 | 11% |
| Other (utils, metrics, observability, data, artifacts) | ~1,363 | 8% |
| **Total** | **~16,701** | 100% |

**Key Insight:** Only 21% of the codebase is used for production. 51% is legacy code maintained for backward compat with archived scripts. 11% is structural code (LossRegistry) that was never integrated.

---

## 9. Dead Code Analysis

### 9.1 src/utils/ternary_lut.py (169 lines) - DEAD CODE

This file implements:
- `VALUATION_LUT` - precomputed 3-adic valuations
- `TERNARY_LUT` - precomputed ternary representations
- `get_valuation_batch()`, `get_ternary_batch()`, `get_3adic_distance_batch()`

**Problem:** `src/core/ternary.py` already has identical LUTs built into `TernarySpace`:
- `self._valuation_lut` (line 62)
- `self._ternary_lut` (line 66)

**Usage:** ZERO imports found - no file in src/ imports from `src.utils.ternary_lut`

**Verdict:** Delete entirely. Complete duplication of `src/core/ternary.py`.

### 9.2 Modules NOT Used by train.py

The production training script (`scripts/train/train.py`) imports ONLY:
```python
from src.models import TernaryVAEV5_11, TernaryVAEV5_11_OptionC
from src.losses import PAdicGeodesicLoss, RadialHierarchyLoss, CombinedGeodesicLoss, GlobalRankLoss
from src.data.generation import generate_all_ternary_operations
from src.core import TERNARY
```

The following modules are **NOT imported** by the production training script:

| Module | Lines | Status |
|--------|-------|--------|
| `src/utils/*` | 541 | UNUSED in production |
| `src/observability/*` | 682 | UNUSED in production |
| `src/artifacts/*` | 269 | UNUSED in production |
| `src/metrics/*` | 232 | UNUSED in production |
| `src/training/*` | 2,866 | UNUSED in production |
| `src/data/dataset.py` | 79 | UNUSED (train.py uses generation directly) |
| `src/data/loaders.py` | 128 | UNUSED |
| `src/data/gpu_resident.py` | 225 | UNUSED |

**Total unused by production:** ~5,022 lines (30% of src/)

---

## 10. Scripts Analysis

### 10.1 Visualization Scripts (15 files, ~7,500 lines)

**ALL visualization scripts import LEGACY models:**

| Script | Imports |
|--------|---------|
| `visualize_ternary_manifold.py` | `DualNeuralVAEV5` (v5.6), `DualNeuralVAEV5_10` |
| `analyze_3adic_structure.py` | `DualNeuralVAEV5` (v5.6) |
| `analyze_3adic_deep.py` | `DualNeuralVAEV5` (v5.6) |
| `calabi_yau_*.py` (6 files) | `DualNeuralVAEV5` (v5.6) |
| `analyze_advanced_manifold.py` | `DualNeuralVAEV5` (v5.6) |

**None use TernaryVAEV5_11.** These scripts are effectively dead for v5.11.

### 10.2 Benchmark Scripts (3 files, ~1,200 lines)

| Script | Imports | Status |
|--------|---------|--------|
| `run_benchmark.py` | `DualNeuralVAEV5`, `DualNeuralVAEV5_10`, `CheckpointManager` | LEGACY |
| `measure_manifold_resolution.py` | `DualNeuralVAEV5`, `CheckpointManager` | LEGACY |
| `measure_coupled_resolution.py` | `DualNeuralVAEV5`, `CheckpointManager` | LEGACY |

**All benchmarks use legacy models.** Cannot benchmark v5.11.

### 10.3 Training Archive (12 files, ~5,600 lines)

All archived training scripts in `scripts/train/archive/` use legacy infrastructure.

---

## 11. Research Directory (37,086 lines)

### 11.1 Structure

| Subdirectory | Files | Lines | Uses v5.11? |
|--------------|-------|-------|-------------|
| `spectral_analysis/` | 11 | ~4,000 | YES (2 scripts) |
| `genetic_code/` | 9 | ~3,500 | YES (1 script) |
| `bioinformatics/rheumatoid_arthritis/` | 40+ | ~15,000 | NO |
| `bioinformatics/hiv/` | 5 | ~2,500 | NO |
| `bioinformatics/sars_cov_2/` | 4 | ~1,500 | NO |
| `bioinformatics/neurodegeneration/` | 6 | ~2,500 | NO |
| `p-adic-genomics/` | 3 | ~1,500 | NO |

**Only 3 scripts in research/ actually use v5.11 models.**

### 11.2 Research Scripts Using v5.11

```
research/spectral_analysis/scripts/01_extract_embeddings.py → TernaryVAEV5_11, TernaryVAEV5_11_OptionC
research/spectral_analysis/scripts/11_variational_orthogonality_test.py → FrozenDecoder
research/genetic_code/scripts/07_extract_v5_11_3_embeddings.py → TernaryVAEV5_11_OptionC
```

---

## 12. Config Archive

### 12.1 Active Config
- `configs/ternary.yaml` (104 lines) - v5.11 production config

### 12.2 Archived Configs (1,747 lines total)

| Config | Status | Notes |
|--------|--------|-------|
| `ternary_v5_6.yaml` | Legacy | Base config, kept for reference |
| `ternary_v5_7.yaml` | Legacy | Added StateNet v3 |
| `ternary_v5_8.yaml` | **ORPHANED** | Model file never existed |
| `ternary_v5_9.yaml` | **ORPHANED** | Model file never existed |
| `ternary_v5_9_2.yaml` | **ORPHANED** | Model file never existed |
| `ternary_v5_10.yaml` | Legacy | Last pre-v5.11 config |
| `appetitive_vae.yaml` | Legacy | Appetitive pathway |

**Note:** `configs/archive/README.md` incorrectly says to use `ternary_v5_10.yaml` - should point to `ternary.yaml`.

---

## 13. Additional Duplications Found

### 13.1 Valuation Computation (4 Implementations!)

Adding to section 3.2, there's actually a FOURTH implementation:

| Location | Function | Status |
|----------|----------|--------|
| `src/core/ternary.py:118` | `TernarySpace.valuation()` | CANONICAL (uses LUT) |
| `src/utils/ternary_lut.py:26` | `_compute_valuation()` | DEAD CODE (not imported) |
| `src/losses/padic_losses.py:46` | `compute_3adic_valuation_batch()` | LEGACY |
| `src/metrics/hyperbolic.py:64` | `compute_3adic_valuation()` | LEGACY |

### 13.2 Coverage Evaluation (2 Implementations)

| Location | Function | Used By |
|----------|----------|---------|
| `src/utils/metrics.py:~100` | `evaluate_coverage()` | Legacy trainers |
| `src/observability/coverage.py:39` | `CoverageEvaluator` | Not used in production |

**Neither is used by train.py** which does coverage differently inline.

---

## 14. Full Codebase Size

| Directory | Files | Lines | Notes |
|-----------|-------|-------|-------|
| `src/` | 49 | 16,701 | Core library |
| `scripts/` | 31 | 14,350 | Training, viz, benchmark |
| `research/` | 90+ | 37,086 | Research experiments |
| `configs/` | 9 | 1,851 | YAML configs |
| `docs/` | 30+ | ~5,000 | Documentation |
| **Total** | **200+** | **~75,000** | |

**Production v5.11 uses:** ~3,440 lines (4.6% of total codebase)

---

## 15. Recommendations

### Option A: Minimal Cleanup (Low Risk)
1. Move broken v5.8/v5.9 scripts to `archive/broken/`
2. Update docs to point to v5.11 patterns
3. Add deprecation warnings to legacy imports
4. Keep all code, just organize better

### Option B: Aggressive Cleanup (Higher Risk)
1. Remove LossRegistry pattern (never integrated)
2. Move legacy losses to `archive/` folder
3. Break backward compat - archived scripts won't work
4. Reduce codebase by ~50%

### Option C: Proper Consolidation (Recommended)
1. Create `src/v5_11/` with only production code
2. Keep `src/legacy/` for backward compat
3. Update `__init__.py` to lazy-load legacy modules
4. Document which path to use for what

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-19 | 0.1 | Initial incomplete analysis |
| 2025-12-19 | 0.2 | Added dependency chains, three approaches, line counts |
| 2025-12-19 | 0.3 | Added duplicated functionality analysis, accurate line counts from wc -l, Chain 5 (src/__init__.py), outdated src/README.md finding |
| 2025-12-19 | 0.4 | Added dead code analysis (ternary_lut.py), modules not used by train.py, scripts analysis (all viz/benchmark use legacy), research directory breakdown (37k lines), config archive, 4th valuation implementation, coverage duplication, full codebase size (75k lines, 4.6% used) |
| 2025-12-19 | 0.5 | Added tests analysis (all use legacy v5.6), setup.py outdated, engine directory (25k lines), detailed unused module analysis, observability infrastructure review |

---

## APPENDIX A: Previously Uncovered Files (v0.5)

### A.1 Tests Directory (3 files, ~1,000 lines)

**Location:** `tests/`

**Critical Issue:** ALL tests use LEGACY v5.6 model (`DualNeuralVAEV5`), NOT v5.11

| File | Lines | Imports | Status |
|------|-------|---------|--------|
| `test_reproducibility.py` | 235 | `DualNeuralVAEV5`, `src.utils.data` | LEGACY |
| `test_generalization.py` | 541 | `DualNeuralVAEV5`, `src.utils.data` | LEGACY |
| `test_training_validation.py` | ~200 | `DualNeuralVAEV5`, `src.utils.data` | LEGACY |

**Problems:**
1. Tests import from `src.utils.data` which doesn't exist (should be `src.data`)
2. Tests reference v5.6 model architecture, not v5.11
3. Tests look for checkpoints in `sandbox-training/checkpoints/v5_6/`
4. **No tests exist for TernaryVAEV5_11 production model**

### A.2 setup.py (91 lines) - COMPLETELY OUTDATED

**Version declared:** `5.6.0` (should be 5.11.3)

**Broken entry points:**
```python
entry_points={
    "console_scripts": [
        "ternary-train=scripts.train.train_ternary_v5_6:main",  # LEGACY
        "ternary-eval=scripts.eval.evaluate_coverage:main",     # DOESN'T EXIST
        "ternary-benchmark=scripts.benchmark.run_benchmark:main",  # LEGACY
    ],
},
```

**Issues:**
1. References `train_ternary_v5_6` which is in archive
2. References `evaluate_coverage` which doesn't exist
3. Package config may not work correctly (`packages=find_packages(where="src")`)

### A.3 Engine Directory (25,629 lines)

**Location:** `engine/`

This is a SEPARATE codebase not covered in main analysis. Contains:

| Subdirectory | Purpose | Status |
|--------------|---------|--------|
| `engine/.claude/hooks/` | Claude Code hooks | Active tooling |
| `engine/benchmarks/` | Performance benchmarks | Unknown usage |
| `engine/tests/` | Engine-specific tests | Unknown usage |
| `engine/opentimestamps/` | IP timestamping | Active tooling |
| `engine/reports/` | Build reports | Generated artifacts |

**Note:** This appears to be operational infrastructure (CI/CD, IP protection) rather than model code.

### A.4 Detailed Unused Module Analysis

#### src/utils/ (541 lines total)

| File | Lines | Purpose | Used By |
|------|-------|---------|---------|
| `reproducibility.py` | 51 | `set_seed()`, `get_generator()` | Nothing (train.py has inline seed setting) |
| `metrics.py` | 277 | `evaluate_coverage()`, `CoverageTracker`, `compute_latent_entropy()` | Legacy trainers only |
| `ternary_lut.py` | 169 | Duplicate LUTs | NOTHING (dead code) |
| `__init__.py` | 44 | Exports | - |

**Key Finding:** `set_seed()` in reproducibility.py is 51 lines, but train.py implements its own 3-line seed setting inline.

#### src/observability/ (682 lines total)

| File | Lines | Purpose | Used By |
|------|-------|---------|---------|
| `metrics_buffer.py` | 198 | `MetricsBuffer`, thread-safe accumulator | Nothing |
| `async_writer.py` | 257 | `AsyncTensorBoardWriter`, background flush | Nothing |
| `coverage.py` | 179 | `CoverageEvaluator`, vectorized coverage | Nothing |
| `__init__.py` | 48 | Exports | - |

**Key Finding:** Complete async observability infrastructure was built but NEVER integrated. Designed for:
- Zero I/O during training (MetricsBuffer)
- Background thread for TensorBoard writes (AsyncTensorBoardWriter)
- Efficient GPU-based coverage evaluation (CoverageEvaluator)

**train.py instead:** Uses blocking TensorBoard writes inline, simple coverage via set operations.

#### src/artifacts/ (269 lines total)

| File | Lines | Purpose | Used By |
|------|-------|---------|---------|
| `checkpoint_manager.py` | 256 | `CheckpointManager`, `AsyncCheckpointSaver` | Legacy trainers only |
| `__init__.py` | 13 | Exports | - |

**Key Finding:** Implements async checkpoint saving with background thread to avoid blocking training. Has features:
- Deep copy of state dicts to avoid race conditions
- Configurable checkpoint frequency
- Best/latest/numbered checkpoints

**train.py instead:** Uses blocking `torch.save()` inline.

#### src/data/ unused files (432 lines total)

| File | Lines | Purpose | Used By |
|------|-------|---------|---------|
| `dataset.py` | 79 | `TernaryOperationDataset` PyTorch Dataset | Nothing |
| `loaders.py` | 128 | `create_ternary_data_loaders()` factory | Nothing |
| `gpu_resident.py` | 225 | `GPUResidentTernaryDataset` optimized loader | Nothing |

**Key Finding:** Three different data loading strategies were implemented:
1. Standard PyTorch Dataset/DataLoader pattern (dataset.py + loaders.py)
2. GPU-resident pattern with zero transfer overhead (gpu_resident.py)

**train.py instead:** Uses `generate_all_ternary_operations()` directly into tensors, no Dataset class.

### A.5 Infrastructure Summary

**Total Unused Infrastructure Lines:**

| Module | Lines | What It Provides |
|--------|-------|------------------|
| `src/utils/` | 541 | Reproducibility, metrics, LUTs |
| `src/observability/` | 682 | Async logging, coverage evaluation |
| `src/artifacts/` | 269 | Async checkpointing |
| `src/data/` (partial) | 432 | Dataset classes, loaders |
| **Total** | **1,924** | Complete training infrastructure |

**This represents a fully-designed training infrastructure that was never wired into the production training loop.**

### A.6 Outputs Directory

**Location:** `outputs/viz/`

Contains generated visualization files with `serve.py` scripts for local viewing:
- `outputs/viz/calabi_yau/serve.py`
- `outputs/viz/calabi_yau_v58/serve.py`

These are generated artifacts, not source code.

### A.7 Updated Full Codebase Size

| Directory | Files | Lines | Notes |
|-----------|-------|-------|-------|
| `src/` | 49 | 16,701 | Core library |
| `scripts/` | 31 | 14,350 | Training, viz, benchmark |
| `research/` | 90+ | 37,086 | Research experiments |
| `engine/` | 50+ | 25,629 | CI/CD, benchmarks, hooks |
| `tests/` | 3 | ~1,000 | Unit tests (LEGACY) |
| `configs/` | 9 | 1,851 | YAML configs |
| `docs/` | 30+ | ~5,000 | Documentation |
| **Total** | **260+** | **~101,000** | |

**Production v5.11 uses:** ~3,440 lines (3.4% of total codebase)
