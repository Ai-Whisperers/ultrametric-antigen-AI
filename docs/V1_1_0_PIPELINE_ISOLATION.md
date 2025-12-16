# V1.1.0 Production Pipeline Isolation Report

**Doc-Type:** Code Isolation Analysis · Version 1.0 · Updated 2025-12-16

---

## Executive Summary

This document identifies the minimal code required to run the v1.1.0 production model (v5.11.3 architecture) and classifies all other code as legacy/unused. The production pipeline is remarkably clean: **8 essential files** power the entire system.

---

## Production Pipeline (USED)

### Entry Points

| File | Purpose | Status |
|------|---------|--------|
| `scripts/train/train.py` | Production training script | ESSENTIAL |
| `scripts/eval/downstream_validation.py` | Production validation | ESSENTIAL |
| `configs/ternary.yaml` | Default configuration | ESSENTIAL |

### Core Model Architecture

| File | Exports Used | Status |
|------|--------------|--------|
| `src/models/ternary_vae.py` | `TernaryVAEV5_11`, `TernaryVAEV5_11_OptionC`, `FrozenEncoder`, `FrozenDecoder` | ESSENTIAL |
| `src/models/hyperbolic_projection.py` | `HyperbolicProjection`, `DualHyperbolicProjection` | ESSENTIAL |
| `src/models/differentiable_controller.py` | `DifferentiableController` | OPTIONAL (use_controller=false by default) |
| `src/models/curriculum.py` | `CurriculumScheduler` | OPTIONAL (not imported by train.py) |
| `src/models/__init__.py` | Re-exports | ESSENTIAL |

### Loss Functions

| File | Exports Used | Status |
|------|--------------|--------|
| `src/losses/padic_geodesic.py` | `PAdicGeodesicLoss`, `RadialHierarchyLoss`, `GlobalRankLoss`, `poincare_distance` | ESSENTIAL |
| `src/losses/__init__.py` | Re-exports | ESSENTIAL |

### Core Domain

| File | Exports Used | Status |
|------|--------------|--------|
| `src/core/ternary.py` | `TERNARY` singleton (valuation, distance) | ESSENTIAL |
| `src/core/__init__.py` | Re-exports | ESSENTIAL |

### Data Generation

| File | Exports Used | Status |
|------|--------------|--------|
| `src/data/generation.py` | `generate_all_ternary_operations` | ESSENTIAL |
| `src/data/__init__.py` | Re-exports | ESSENTIAL |

---

## Legacy Code (NOT USED by v1.1.0)

### src/losses/ - Legacy Loss Functions

| File | Contains | Recommendation |
|------|----------|----------------|
| `appetitive_losses.py` | `AlgebraicClosureLoss` (BROKEN), `CuriosityModule`, `SymbioticBridge` | ARCHIVE |
| `consequence_predictor.py` | `ConsequencePredictor`, `evaluate_addition_accuracy` (TRIVIAL TEST BUG) | ARCHIVE |
| `hyperbolic_prior.py` | `HyperbolicPrior`, `HomeostaticHyperbolicPrior` | ARCHIVE |
| `hyperbolic_recon.py` | `HyperbolicReconLoss`, `HomeostaticReconLoss` | ARCHIVE |
| `padic_losses.py` | `PAdicRankingLoss`, `PAdicRankingLossV2` (superseded by padic_geodesic) | ARCHIVE |
| `radial_stratification.py` | `RadialStratificationLoss` (superseded by RadialHierarchyLoss) | ARCHIVE |
| `dual_vae_loss.py` | `DualVAELoss`, `ReconstructionLoss` (not used in v5.11) | ARCHIVE |
| `base.py` | `LossComponent`, `LossResult` (LossRegistry pattern - unused) | ARCHIVE |
| `registry.py` | `LossRegistry` (unused pattern) | ARCHIVE |
| `components.py` | `*LossComponent` wrappers (unused pattern) | ARCHIVE |

### src/training/ - Unused Trainers

| File | Contains | Recommendation |
|------|----------|----------------|
| `trainer.py` | `TernaryVAETrainer` (NOT used by train.py - has division bugs) | ARCHIVE |
| `hyperbolic_trainer.py` | `HyperbolicVAETrainer` (NOT used by train.py) | ARCHIVE |
| `schedulers.py` | Various schedulers (train.py uses PyTorch native) | ARCHIVE |
| `monitor.py` | `TrainingMonitor` (train.py uses TensorBoard directly) | ARCHIVE |
| `config_schema.py` | `TrainingConfig` (train.py uses argparse + yaml) | ARCHIVE |
| `environment.py` | `validate_environment` (not imported) | ARCHIVE |
| `__init__.py` | Re-exports (keep for backwards compat) | KEEP |
| `archive/appetitive_trainer.py` | Already archived | ALREADY ARCHIVED |

### src/models/ - Already Archived

| File | Status |
|------|--------|
| `archive/ternary_vae_v5_6.py` | ALREADY ARCHIVED |
| `archive/ternary_vae_v5_7.py` | ALREADY ARCHIVED |
| `archive/ternary_vae_v5_10.py` | ALREADY ARCHIVED |
| `archive/appetitive_vae.py` | ALREADY ARCHIVED |

### src/data/ - Minimal, All Used

| File | Status |
|------|--------|
| `dataset.py` | KEEP (may be useful for DataLoader approach) |
| `loaders.py` | KEEP (has val_loader bug but not used in production) |

### src/utils/ - Metrics Utilities

| File | Contains | Recommendation |
|------|----------|----------------|
| `metrics.py` | `CoverageTracker` (has wrong union bug), `evaluate_coverage` | REVIEW |
| `reproducibility.py` | `set_seed` | KEEP |

### src/metrics/ - Evaluation Metrics

| File | Contains | Recommendation |
|------|----------|----------------|
| `hyperbolic.py` | Poincare metrics (used by validation) | KEEP |
| `__init__.py` | Re-exports | KEEP |

### src/artifacts/ - Checkpoint Management

| File | Contains | Recommendation |
|------|----------|----------------|
| `checkpoint_manager.py` | Checkpoint I/O | KEEP |

---

## Dependency Graph

```
scripts/train/train.py
├── src/models/
│   ├── ternary_vae.py (TernaryVAEV5_11_OptionC)
│   │   └── hyperbolic_projection.py (DualHyperbolicProjection)
│   │   └── differentiable_controller.py (optional)
│   └── __init__.py
├── src/losses/
│   ├── padic_geodesic.py (PAdicGeodesicLoss, RadialHierarchyLoss, GlobalRankLoss)
│   └── __init__.py
├── src/data/
│   ├── generation.py (generate_all_ternary_operations)
│   └── __init__.py
└── src/core/
    ├── ternary.py (TERNARY singleton)
    └── __init__.py
```

---

## Recommended Archive Actions

### Phase 1: Move to src/losses/archive/

```
src/losses/archive/
├── appetitive_losses.py      # Contains BROKEN compose_operations
├── consequence_predictor.py  # Contains TRIVIAL TEST bug
├── hyperbolic_prior.py
├── hyperbolic_recon.py
├── padic_losses.py           # Superseded by padic_geodesic.py
├── radial_stratification.py  # Superseded by RadialHierarchyLoss
├── dual_vae_loss.py          # v5.5 era losses
├── base.py                   # Unused LossRegistry pattern
├── registry.py
└── components.py
```

### Phase 2: Move to src/training/archive/

```
src/training/archive/
├── trainer.py           # Has division-by-zero bugs
├── hyperbolic_trainer.py
├── schedulers.py
├── monitor.py
├── config_schema.py
├── environment.py
└── appetitive_trainer.py  # Already here
```

### Phase 3: Update __init__.py Files

After archiving, update `src/losses/__init__.py` to only export production classes:

```python
# Production exports (V5.11.3)
from .padic_geodesic import (
    poincare_distance,
    PAdicGeodesicLoss,
    RadialHierarchyLoss,
    GlobalRankLoss
)

__all__ = [
    'poincare_distance',
    'PAdicGeodesicLoss',
    'RadialHierarchyLoss',
    'GlobalRankLoss',
]
```

---

## Bug Isolation Summary

The 20 bugs from CODEBASE_AUDIT.md are distributed as follows:

| Location | Bug Count | Production Impact |
|----------|-----------|-------------------|
| `src/losses/appetitive_losses.py` | 1 (L3.1 broken composition) | NONE - not used |
| `src/losses/consequence_predictor.py` | 1 (L3.2 trivial test) | NONE - not used |
| `src/losses/padic_losses.py` | 3 (P1.1-P1.3 O(n²) loops) | NONE - not used |
| `src/losses/hyperbolic_prior.py` | 4 (D2.1, H4.1-H4.3) | NONE - not used |
| `src/losses/hyperbolic_recon.py` | 1 (H4.4) | NONE - not used |
| `src/training/trainer.py` | 1 (S5.2 div-by-zero) | NONE - not used |
| `src/training/appetitive_trainer.py` | 4 (S5.3-S5.5, H4.5) | NONE - already archived |
| `src/training/hyperbolic_trainer.py` | 1 (D2.2) | NONE - not used |
| `src/data/loaders.py` | 1 (S5.1 empty val_loader) | NONE - not used |
| `src/utils/metrics.py` | 3 (A8.1, V9.1, U10.1) | LOW - reporting only |

**Conclusion:** All 17/20 bugs are in UNUSED code. Only 3 minor bugs in utils/metrics.py affect production, and those are reporting bugs (not training bugs).

---

## Production Checkpoint

| Component | Path |
|-----------|------|
| Base checkpoint (v5.5) | `sandbox-training/checkpoints/v5_5/latest.pt` |
| Production weights | `sandbox-training/checkpoints/ternary_option_c_dual/best.pt` |
| Validation script | `scripts/eval/downstream_validation.py` |

---

## Verification Commands

```bash
# Verify production training works
python scripts/train/train.py --option_c --dual_projection --epochs 5

# Verify production validation works
python scripts/eval/downstream_validation.py

# Check essential file count
find src -name "*.py" -path "*/models/*" ! -path "*/archive/*" | wc -l  # Should be 5
find src -name "*.py" -path "*/losses/*" ! -path "*/archive/*" | wc -l  # Should be 12 (before cleanup)
```

---

## Changelog

| Date | Version | Description |
|------|---------|-------------|
| 2025-12-16 | 1.0 | Initial isolation analysis |

---

**Status:** Ready for Phase 1 archive operations pending approval.
