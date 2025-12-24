# P0: Geometry & Stability Upgrade

**Status:** Open
**Source:** IMPROVEMENT_PLAN.md (Sec 1)
**Area:** Model Architecture

## Objective

Guarantee numerical stability for Poincare ball operations by enforcing `geoopt` usage.

## Tasks

- [ ] **Enforce `geoopt` Requirement**: Remove the `try-except` block in `poincare.py` and make `geoopt` a hard dependency.
- [ ] **Adopt `hgcn` Mobius Addition**: Review/replace manual fallback of `mobius_add` if needed.
- [ ] **Integrate Riemannian Optimizers**: Ensure training scripts use `get_riemannian_optimizer`.
