# Test Implementation Guide

**Status**: Active
**Purpose**: Technical standard for writing tests in `tests/`.

## 1. Environment & Tooling

- **Runner**: `pytest`
- **Plugins**: `pytest-cov` (coverage), `pytest-mock` (mocking).
- **Config**: `pyproject.toml` or `pytest.ini` (ensure `pythonpath = .`).

## 2. Fixture Strategy (`tests/conftest.py`)

We explicitly define shared resources to avoid "setup fatigue".

| Fixture       | Scope    | Description                                       |
| :------------ | :------- | :------------------------------------------------ |
| `device`      | Session  | Returns `cuda` if available, else `cpu`.          |
| `ternary_ops` | Session  | A standard batch of valid ternary operations.     |
| `model_v5_11` | Function | An instantiated `TernaryVAEV5_11` (small config). |
| `poincare`    | Session  | A `PoincareBall` manifold instance.               |

## 3. Unit Test Standards (`tests/unit/`)

- **Isolation**: No network, no disk I/O (mock it).
- **Precision**: Use `torch.testing.assert_close`.
- **Performance**: Each test must run in < 100ms.

### Example: Geometry Test

```python
def test_poincare_distance_invariance(poincare, device):
    x = torch.zeros(1, 2, device=device)
    y = torch.tensor([[0.5, 0.0]], device=device)
    dist = poincare.dist(x, y)
    assert torch.isclose(dist, torch.tensor([1.0986]), atol=1e-4)
```

## 4. Integration Standards (`tests/integration/`)

- **Scope**: Pipeline validation.
- **Data**: Use synthetic data generation from `src.data.generation`.
- **Check**: Gradients flow, Loss decreases, Checkpoints save.

## 5. Scientific Standards (`tests/e2e/`)

- **Scope**: Paper replication.
- **Input**: Real subsets of `data/processed/`.
- **Thresholds**: Defined in `SCIENTIFIC_SUITE.md`.

## 6. Implementation Order

1.  **Foundation**: Create `tests/conftest.py`.
2.  **Logic**: Implement `tests/unit/test_geometry.py`.
3.  **Architecture**: Implement `tests/unit/test_models.py`.
4.  **Science**: Implement `tests/e2e/test_scientific.py`.
