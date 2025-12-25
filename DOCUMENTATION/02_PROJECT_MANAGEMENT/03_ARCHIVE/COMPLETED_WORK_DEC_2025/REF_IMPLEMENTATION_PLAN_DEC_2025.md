# Refactoring Plan: Testability & Abstraction

> **Goal:** Decouple components to facilitate unit testing, mocking, and future extensions.

## Proposed Changes

### 1. Core Interfaces

**Goal:** Define clear contracts for model components using Python Protocols.

#### [NEW] [src/core/interfaces.py](file:///c:/Users/Alejandro/Documents/Ivan/Work/ternary-vaes-bioinformatics/src/core/interfaces.py)

- `EncoderProtocol`: Interface for VAE encoders.
- `DecoderProtocol`: Interface for VAE decoders.
- `ProjectionProtocol`: Interface for Hyperbolic projections.
- `ModelBuilderProtocol`: Interface for model factories.

### 2. Dependency Injection & Factories

**Goal:** Move complex construction logic out of `train.py` and `__init__` methods.

#### [NEW] [src/factories/model_factory.py](file:///c:/Users/Alejandro/Documents/Ivan/Work/ternary-vaes-bioinformatics/src/factories/model_factory.py)

- **Class:** `TernaryModelFactory`
- **Method:** `create_model(config: Dict) -> TernaryVAEV5_11`
- **Method:** `create_components(config) -> (Encoder, Decoder, Projection)`: Allows mostly independent testing of component creation.

#### [MODIFY] [src/models/ternary_vae.py](file:///c:/Users/Alejandro/Documents/Ivan/Work/ternary-vaes-bioinformatics/src/models/ternary_vae.py)

- Update `__init__` to accept optional injected instances of `encoder_A`, `encoder_B`, etc.
- Maintain backward compatibility (if instances not provided, create defaults).

### 3. Automated Test Harness

**Goal:** Create a reusable harness that standardizes model verification.

#### [NEW] [tests/harnesses/model_harness.py](file:///c:/Users/Alejandro/Documents/Ivan/Work/ternary-vaes-bioinformatics/tests/harnesses/model_harness.py)

- **Class:** `ModelTestHarness`
- **Functionality:**
  - `check_shapes(input_shape, expected_output_keys)`
  - `check_gradient_flow()`: Verifies that gradients reach trainable params.
  - `check_invariance(input_tensor, transformation)`: optional checks.

## Verification Plan

- Create `tests/unit/test_refactor_verification.py` using the new `ModelTestHarness` to verify existing models.
