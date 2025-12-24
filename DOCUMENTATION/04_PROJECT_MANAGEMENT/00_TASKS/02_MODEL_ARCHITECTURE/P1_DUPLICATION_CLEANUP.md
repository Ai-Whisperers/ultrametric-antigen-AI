# P1: Code Duplication Cleanup

**Status:** Open
**Source:** DUPLICATION_REPORT
**Area:** Architecture / Maintenance

## Targets

- [ ] **Benchmark Boilerplate**: Extract setup code from `scripts/benchmark/measure_*.py` into `src/benchmark/utils.py`.
- [ ] **Visualization Math**: Consolidate `quintic_fibration`, `mirror_symmetry` from `scripts/visualization/calabi_yau_*.py` into `src/geometry/projections.py`.
- [ ] **Trainer Inheritance**: (See P1_REFACTOR.md). Unified base class for `TernaryVAETrainer` and `AppetitiveVAETrainer`.
