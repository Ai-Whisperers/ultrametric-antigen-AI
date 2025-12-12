# Archived Training Scripts

These training scripts are archived as part of the v5.10.1 unification.

## Status

| File | Reason | Notes |
|------|--------|-------|
| train_ternary_v5_5.py | Deprecated | Pre-rename version |
| train_ternary_v5_6.py | Legacy | Superseded by v5.10 |
| train_ternary_v5_7.py | Legacy | Superseded by v5.10 |
| train_ternary_v5_8.py | Broken | Imports v5.6 model, not v5.8 (which doesn't exist) |
| train_ternary_v5_9.py | Broken | Imports v5.6 model, not v5.9 (which doesn't exist) |
| train_ternary_v5_9_1.py | Broken | Imports v5.6 model, not v5.9 (which doesn't exist) |

## v5.8/v5.9 Issue

These scripts were supposed to use dedicated model files that were never created:
- `train_ternary_v5_8.py` imports `DualNeuralVAEV5` from v5.6
- `train_ternary_v5_9.py` imports `DualNeuralVAEV5` from v5.6

This means the "two-phase training" and "continuous feedback" config sections were never actually used.

## Current Primary Script

Use `scripts/train/train_ternary_v5_10.py` - the unified training script that:
- Uses `DualNeuralVAEV5_10` (correct model)
- Implements continuous feedback (from v5.9)
- Uses PAdicRankingLossHyperbolic with hard negatives (from v5.8)
- Adds homeostatic emergence (v5.10 new)
- Has optimized eval intervals

---

Archived: 2025-12-12 as part of v5.10.1 cleanup
