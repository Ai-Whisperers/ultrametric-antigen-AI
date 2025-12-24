# CI/CD Workflow

## Pipeline Stages

1.  **Lint & Static Analysis**

    - `flake8`, `black`, `mypy`.
    - Fail fast.

2.  **Unit Tests (Smoke)**

    - Run `pytest tests/suites/unit -m "not slow"`.
    - Must satisfy 100% pass rate.

3.  **Integration Tests**

    - Run `pytest tests/suites/integration`.
    - Requires DB service.

4.  **Scientific Validation**
    - Run `pytest tests/suites/e2e`.
    - Generates artifacts (scores, plots).

## Triggers

- **Pull Request**: Runs Stages 1 & 2.
- **Merge to Main**: Runs Stages 1, 2, 3.
- **Nightly**: Runs Stage 4 (Heavy compute).
