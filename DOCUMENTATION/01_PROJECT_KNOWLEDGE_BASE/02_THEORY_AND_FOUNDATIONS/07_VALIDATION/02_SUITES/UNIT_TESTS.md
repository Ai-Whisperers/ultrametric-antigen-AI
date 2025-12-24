# Unit Test Suite

**Location:** `tests/suites/unit/`
**Goal:** Verify mathematical correctness and component initialization.

## Included Tests

### 1. Geometry Checks (`test_geometry.py`)

Verifies the foundational hyperbolic, ternary, and functional operations.

| Function Test                       | Description                                                       |
| :---------------------------------- | :---------------------------------------------------------------- |
| `test_poincare_distance_identity`   | Ensures distance from a point to itself is 0.                     |
| `test_poincare_distance_shape`      | Verifies correct tensor shapes for distance calculations.         |
| `test_projection_limits`            | Checks that `project_to_poincare` keeps norms strictly < 1.0.     |
| `test_mobius_addition_zero`         | Verifies identity property of Mobius addition ($x \oplus 0 = x$). |
| `test_poincare_triangle_inequality` | Verifies metric space property: $d(a,c) \le d(a,b) + d(b,c)$.     |

### 2. Model Component Checks (`test_models.py`)

Verifies the neural architecture assembly and gradient flow.

| Function Test                    | Description                                                                                             |
| :------------------------------- | :------------------------------------------------------------------------------------------------------ |
| `test_vae_initialization`        | Ensures the `VAEBuilder` correctly assembles the model.                                                 |
| `test_vae_forward_shape`         | Checks input/output tensor shapes for the VAE forward pass.                                             |
| `test_gradients_only_projection` | Critical Limit Check: Ensures gradients flow to the `Projection` layer but STOP at the `FrozenEncoder`. |
| `test_dual_projection_flow`      | Verifies that the dual-projection (A/B) architecture receives gradients correctly.                      |
