# CI/CD Automation Plan

**Status**: Draft

## 1. Trigger Strategy

- **Push to `main`**: Run `tests/unit/` and `tests/integration/`.
- **Schedule (Daily)**: Run `tests/e2e/`.

## 2. Environment

- **Container**: Use project `Dockerfile`.
- **Hardware**: CPU for Unit, GPU (if available) for E2E.

## 3. Coverage Targets

- **Core Modules**: 90% (Strict).
- **Experiments**: 50% (Loose).
