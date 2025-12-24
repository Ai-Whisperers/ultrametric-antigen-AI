# Abstractions

## Drivers (`tests/core/drivers/`)

Wrappers around external tools.

- `api_driver`: Wraps `requests` or `httpx`. Handles auth tokens automatically.
- `db_driver`: Wraps `sqlalchemy` or raw SQL. Handles rapid setup/teardown.
- `browser_driver`: Wraps `selenium` or `playwright`.

## Matchers (`tests/core/matchers/`)

Custom assertions for domain-specific checks.

- `expect(z_hyp).toBeOnPoincareDisk()`
- `expect(model).toHaveGradientFlow()`
