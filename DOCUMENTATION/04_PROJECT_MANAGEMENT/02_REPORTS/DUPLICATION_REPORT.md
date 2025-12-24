# Detailed Code Duplication Analysis

This report provides a deeper analysis of the code duplication identified in the codebase, categorizing it by type and risk level.

## 1. Archival Snapshots (Low Risk)

**Pattern:** High duplication between files in `archive/` directories and current versions.
**Examples:**

- `src/models/archive/ternary_vae_v5_6.py` vs `v5_7.py` vs `v5_10.py`
- `scripts/train/archive/train_ternary_v5_*.py`

**Explanation:**
This is "Evolutionary Duplication". Whole files were copied to preserve the state of the model at specific versions (v5.6, v5.7, etc.) while development continued.

- **Nature of Duplication:** 90%+ identical code. Entire class definitions (`TernaryVAE`, `Encoder`, `Decoder`) are repeated.
- **Risk:** Low. These files are historical records. They are not expected to change.
- **Recommendation:** No action needed unless disk space is a concern.

## 2. Forked Logic / Divergent Implementations (High Risk)

**Pattern:** Core logic copied and modified to support a feature variant, creating two parallel implementations that must be maintained.
**Examples:**

- `src/training/trainer.py` (Main) vs `src/training/archive/appetitive_trainer.py` (Variant)

**Explanation:**
The `AppetitiveVAETrainer` structurally mirrors the main `TernaryVAETrainer` but adds specific methods for 3-adic valuation (`_compute_3adic_valuation`) and curiosity-driven losses.

- **Nature of Duplication:**
  - Identical `__init__` boilerplate (config handling, device setup).
  - Identical `train_epoch` loop structure (batch iteration, logging, timing).
  - Identical `validate` loop.
- **Risk:** High. Improvements to the core training loop (e.g., better checkpointing, logging fixes, performance optimizations) in `trainer.py` will NOT automatically propagate to `appetitive_trainer.py`. The "Appetitive" logic effectively rots as the main trainer evolves.
- **Recommendation:** Refactor to use inheritance.
  - Create a base `Trainer` class in `src/training/base.py`.
  - `TernaryVAETrainer` inherits `Trainer`.
  - `AppetitiveVAETrainer` inherits `TernaryVAETrainer` (or base) and overrides only the loss calculation and logging methods.

## 3. Test Harness Redundancy (Medium Risk)

**Pattern:** Scripts that perform different tasks but share massive amounts of setup and boilerplate code.
**Examples:**

- `scripts/benchmark/measure_coupled_resolution.py`
- `scripts/benchmark/measure_manifold_resolution.py`

**Explanation:**
These scripts share the "Harness" code:

- **Nature of Duplication:**
  - Imports and path setup (`sys.path.append(...)`).
  - Model loading & initialization (`CoupledSystemBenchmark.__init__` vs `ManifoldResolutionBenchmark.__init__`).
  - Configuration parsing (`convert_to_python_types`, `main` entry point CLI args).
- **Risk:** Medium. If the model initialization API changes (e.g., config schema changes), all benchmark scripts must be updated individually.
- **Recommendation:** Extract a `BenchmarkBase` class or `setup_benchmark_model` utility function in `src/benchmark/utils.py`.

## 4. Visualization Variants (Medium Risk)

**Pattern:** Mathematical functions copied across multiple script variants (e.g., "fast" vs "extended" vs "paper").
**Examples:**

- `scripts/visualization/calabi_yau_v58_extended.py`
- `scripts/visualization/calabi_yau_v58_fast.py`

**Explanation:**
The core mathematical logic for the projections is duplicated:

- **Nature of Duplication:** Functions like `quintic_fibration(z)`, `mirror_symmetry(z)`, `k3_surface(z)` are copy-pasted identically.
- **Risk:** Medium. If a bug is found in the math of the `quintic_fibration` projection, fixing it in `extended.py` leaves `fast.py` incorrect.
- **Recommendation:** Move these mathematical projection functions into a shared library, e.g., `src/geometry/projections.py`.

## Summary of Actionable Items

| Files                           | Action                                             | Effort |
| ------------------------------- | -------------------------------------------------- | ------ |
| `training/*.py`                 | Refactor into BaseTrainer class hierarchy          | High   |
| `visualization/calabi_yau_*.py` | Extract math functions to `src/geometry`           | Low    |
| `benchmark/measure_*.py`        | Extract setup boilerplate to `src/benchmark/utils` | Low    |
