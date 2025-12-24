# Master Validation & Benchmarking Plan

**Status:** Draft · **Ref:** 2025-12-12 Docs

This plan operationalizes the "Manifold Strategy", "Jona Research Roadmap", and "Technical Debt Audit" into a unified execution path.

## 1. The "Manifold Validation" Strategy

**Ref:** `2025_HYPERBOLIC_MANIFOLD_STRATEGY.md`
**Context:** Since our dataset is finite (19,683 operations), standard "Train/Val/Test" splits are invalid. We must shift to "Exhaustive Training" with "Manifold Benchmarking".

### 1.1 Structural Changes (Priority: P0)

- [ ] **Data Splits**: Update `loaders.py` to support `val_split=0` and return `None` for val/test loaders.
- [ ] **Trainer Logic**: Update `hyperbolic_trainer.py` to skip validation loops when `val_loader` is None.
- [ ] **Config Update**: Set `train_split: 1.0` in `ternary_v5_10.yaml`.

### 1.2 "Manifold Quality" Metrics (Priority: P0)

We replace "Validation Loss" with a Composite Score for model selection:

- [ ] **Composite Score**: `0.6 * (Coverage/100) + 0.4 * max(Correlation, 0)`
- [ ] **Early Stopping**: Stop only when **BOTH** Coverage and Correlation plateau for 30 epochs.
- [ ] **Implementation**: Add `compute_composite_score` and state tracking to `src/training/monitor.py`.

## 2. Technical Debt Remediation

**Ref:** `TECHNICAL_DEBT_AUDIT_2025_12_12.md`
**Context:** Critical bugs (P0) threaten the stability of the new Manifold approach.

### 2.1 Critical Fixes (Priority: P0 - Pre-requisite)

- [ ] **Div/0 Guards**: Fix unsafe division in `trainer.py` (L362), `appetitive_trainer.py` (L408, L475).
- [ ] **Unconditional Access**: Guard `val_loader` access in `appetitive_trainer.py` (L541).
- [ ] **Broken Logic**: Disable or fix `compose_operations()` in `appetitive_losses.py` (L428) as it produces meaningless gradients.

### 2.2 Performance & Correctness (Priority: P1)

- [ ] **Metric Bugs**: Fix `CoverageTracker.update` to use _actual_ set union, not `max()`.
- [ ] **Startup Speed**: Vectorize O(n²) 3-adic distance matrix calculation in `padic_losses.py`.

## 3. Scientific Benchmarking Suite

**Ref:** `00_MASTER_ROADMAP_JONA.md`, `RELEVANT_REPOS.md`
**Context:** "Bench Validations" requires proving the learned manifold actually works for science.

### 3.1 Disentanglement (Priority: P2)

- [ ] **MIG Metric**: Implement Mutual Information Gap (`mig.py`) to prove independence of latent factors.
- [ ] **Target**: Show that "Infectivity" (if labeled) maps to orthogonal dimensions.

### 3.2 Biological Integration (Priority: P2)

- [ ] **Scanpy Loader**: Implement `src/data/bio_loader.py` for `.h5ad` support.
- [ ] **Velocity**: Prototype "Hyperbolic Velocity" metric based on `scvelo`.

### 3.3 Visual Validation (Priority: P3)

- [ ] **Hyperparameter Pareto**: Use `hiplot` to visualize `LR` vs `Rho` vs `Coverage`.
- [ ] **Embedding Viz**: Since TensorBoard fails for 16D, generate static PCA/UMAP plots of the Poincaré disk (using `hyperbolic-tsne` if available).

## 4. Execution Roadmap

| Phase       | Goal           | Key Tasks                                                             |
| :---------- | :------------- | :-------------------------------------------------------------------- |
| **Phase 1** | **Stability**  | Fix P0 Tech Debt (Div/0, Loader Guards).                              |
| **Phase 2** | **Structure**  | Implement Manifold Loader (`split=1.0`) & Monitor changes.            |
| **Phase 3** | **Validation** | Verify "Manifold Metrics" (Composite Score) correctly drive training. |
| **Phase 4** | **Science**    | Add MIG metric and Scanpy loader.                                     |

## 5. Immediate Next Steps

1.  **Authorize Phase 1**: Fix the Division-by-Zero and Loader bugs (P0).
2.  **Authorize Phase 2**: Implement the `val_split=0` logic.

Shall we proceed with **Phase 1 (Stability Fixes)**?
