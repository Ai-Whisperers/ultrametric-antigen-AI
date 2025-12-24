# P1: Fix Critical & Silent Bugs

**Status:** Open
**Source:** TECHNICAL_DEBT_AUDIT (Category 3 & 5)
**Area:** Infrastructure / Stability

## Critical Logic Errors (P0)

- [ ] **Fix Broken Operation Composition**: `src/losses/appetitive_losses.py` (lines 428-486). The composition logic is admitted to be "confused" in comments.
- [ ] **Fix Trivial Addition Test**: `src/losses/consequence_predictor.py` (lines 219-229). Tests `z_A + z_0 - z_0 ≈ z_A` which is tautological.

## Silent Failures (P0)

- [ ] **Trainer Division by Zero**: `src/training/trainer.py` (362-363). Crashes if `num_batches=0`.
- [ ] **Appetitive Division by Zero**: `src/training/appetitive_trainer.py`. Same issue.
- [ ] **Unconditional val_loader**: `src/training/appetitive_trainer.py` (541). Crashes if validation is disabled (common in manifold training).
