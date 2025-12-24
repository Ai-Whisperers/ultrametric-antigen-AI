# Master Test Strategy

**Status**: Active
**Version**: 1.0 (2025-12-24)

This document defines the "Pyramid of Testing" for the Ternary VAEs Bioinformatics project, ensuring we validate both _code correctness_ and _scientific validity_.

## 1. The Pyramid of Testing

We allocate testing effort according to the 70/20/10 rule.

### Layer 1: Unit Tests (70%)

- **Scope**: Individual functions, classes (e.g., `HyperbolicLayer`, `ELBOLoss`).
- **Goal**: Correctness. Does the math compute?
- **Location**: `tests/unit/`
- **Speed**: < 100ms per test.
- **Mocks**: Aggressive mocking of data/IO.

### Layer 2: Integration Tests (20%)

- **Scope**: Pipelines, Model+Loss+Data loops.
- **Goal**: Compatibility. Do tensor shapes match? Does gradient descent step?
- **Location**: `tests/integration/`
- **Speed**: < 30s per test.
- **Mocks**: Use synthetic "toy" datasets.

### Layer 3: Scientific/E2E (10%)

- **Scope**: Full reproduction of papers, convergence checks.
- **Goal**: Validity. Does the latent space imply biology?
- **Location**: `tests/e2e/`
- **Speed**: Minutes to Hours.
- **Mocks**: None. Use real (subsetted) biological data.

## 2. Testing Governance

### The "Goldilocks Protocol"

Every new scientific component MUST pass three gates:

1.  **Unit**: It runs without error.
2.  **Integration**: It fits in the VAE pipeline.
3.  **Scientific**: It produces better-than-random stats on benchmarks.

### Continuous Integration (CI)

- **PRs**: Must pass Unit + Integration (Smoke).
- **Nightly**: Runs Scientific/E2E suite.

## 3. Directory Structure

- `DOCUMENTATION/05_VALIDATION/01_PLANS/`: The "Why" and "How" (This file).
- `DOCUMENTATION/05_VALIDATION/02_SUITES/`: The "What" (Test designs).
- `tests/`: The "Code" (Implementation).
