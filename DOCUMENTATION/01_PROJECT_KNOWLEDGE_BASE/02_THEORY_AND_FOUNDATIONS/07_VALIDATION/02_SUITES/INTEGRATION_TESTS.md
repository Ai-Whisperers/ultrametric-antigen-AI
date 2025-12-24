# Integration Test Suite

**Location:** `tests/suites/integration/`
**Goal:** Verify component compatibility and environment stability.

## Included Tests

### 1. Environment Smoke Test (`test_ai_environment_smoke.py`)

A fast "sanity check" to ensure the dev environment is correctly configured.

| Function Test                               | Description                                                                                                |
| :------------------------------------------ | :--------------------------------------------------------------------------------------------------------- |
| `test_environment_import_and_instantiation` | Verifies that critical dependencies (`torch`, `geoopt`, `src`) load and the VAE class can be instantiated. |

### 2. Generalization & Robustness (`test_generalization_v5_11.py`)

Verifies that the model handles variations in input structure typically found in biological data (permutations, compositions).

| Function Test                         | Description                                                                                                                                          |
| :------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_permutation_robustness_smoke`   | Checks that permuting input symbols (e.g., codon swaps) does not break embedding flow or violate hyperbolic constraints.                             |
| `test_compositional_arithmetic_smoke` | Verifies the "Arithmetic Composition" pipeline: ensuring that composing two operations allows the model to predict a result vector without crashing. |
