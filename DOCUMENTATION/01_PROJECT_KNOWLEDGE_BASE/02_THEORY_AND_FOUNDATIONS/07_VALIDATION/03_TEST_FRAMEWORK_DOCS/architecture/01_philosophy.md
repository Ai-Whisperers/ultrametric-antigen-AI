# Testing Philosophy

**Goal**: Tests should verify _behavior_, not _implementation_.

## Core Principles

1.  **Behavior Over Implementation**

    - Test _what_ the system does, not _how_ it does it.
    - Avoid testing private methods.
    - If you refactor the code but the behavior stays the same, the test should NOT break.

2.  **Factory-Driven Data**

    - Never use hardcoded magic values in tests.
    - Use Factories (`tests/factories/`) to generate semantic data.
    - Example: `UserFactory.create(state='banned')` instead of `{ "status": "banned" }`.

3.  **Abstraction Layer**

    - Tests should read like English requirements.
    - Complex setup/teardown logic belongs in `tests/core/`.
    - UI selectors belong in `tests/e2e_support/pages/`.

4.  **The Pyramid**
    - **Unit (70%)**: Fast, isolated, covers all branches.
    - **Integration (20%)**: Checks component wiring.
    - **E2E (10%)**: Simulates critical user journeys.
