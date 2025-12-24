# P2: Dead Code & Reliability Removal

**Status:** Open
**Source:** TECHNICAL_DEBT_AUDIT (Category 2)
**Area:** Infrastructure

## Cleanup Targets

- [ ] **Remove Unused Origin Buffer**: `src/losses/hyperbolic_prior.py` (67).
- [ ] **Fix Discarded Validation**: `src/training/hyperbolic_trainer.py` (290). `val_losses` are computed but thrown away.
- [ ] **Fix Wrong Union Approx**: `src/utils/metrics.py` (220). Uses `max()` instead of set union.
