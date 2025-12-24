# Unit Test Suite Definition

## Scope

Verification of atomic logic in `src/`.

## Test Plan

| Module         | Component             | Test Case                                                 | ID             |
| :------------- | :-------------------- | :-------------------------------------------------------- | :------------- |
| `src.models`   | `HyperbolicLayer`     | Verify Poincare distance invariance.                      | `UNIT-001`     |
| `src.losses`   | `ELBO`                | Verify breakdown (KL + Recon).                            | `UNIT-002`     |
| `src.data`     | `FastaLoader`         | Verify handling of degenerate bases (N).                  | `UNIT-003`     |
| `src.geometry` | `poincare_distance`   | Verify distance(x, x) approx 0.                           | `UNIT-GEO-001` |
| `src.geometry` | `poincare_distance`   | Verify triangle inequality.                               | `UNIT-GEO-002` |
| `src.geometry` | `mobius_add`          | Verify commutativity for zero vector (x+0 = x).           | `UNIT-GEO-003` |
| `src.geometry` | `project_to_poincare` | Verify norm < 1.0 after projection.                       | `UNIT-GEO-004` |
| `src.geometry` | `parallel_transport`  | Verify norm preservation of tangent vector.               | `UNIT-GEO-005` |
| `src.models`   | `FrozenEncoder`       | Verify outputs are deterministic (no dropout).            | `UNIT-MOD-001` |
| `src.models`   | `TernaryVAEV5_11`     | Verify forward pass shape (Batch, 9, 3).                  | `UNIT-MOD-002` |
| `src.models`   | `TernaryVAEV5_11`     | Verify gradient flow to Projection only (Encoder frozen). | `UNIT-MOD-003` |
| `src.losses`   | `ELBO`                | Verify non-negative KL divergence.                        | `UNIT-LOS-001` |
| `src.losses`   | `DualVAELoss`         | Verify reduction (scalar output).                         | `UNIT-LOS-002` |
| `src.losses`   | `RepulsionLoss`       | Verify loss > 0 for identical points.                     | `UNIT-LOS-003` |
