# Testing

Guide to testing in Ternary VAE.

---

## Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── config/              # Config module tests
│   ├── geometry/            # Geometry operations
│   ├── losses/              # Loss functions
│   ├── models/              # Model tests
│   └── training/            # Training utilities
├── integration/             # Cross-module tests
├── benchmarks/              # Performance tests
└── conftest.py              # Shared fixtures
```

---

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### Specific Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Single module
pytest tests/unit/geometry/ -v

# Single file
pytest tests/unit/geometry/test_poincare.py -v

# Single test
pytest tests/unit/geometry/test_poincare.py::test_distance_symmetry -v
```

### With Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### Markers

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Only GPU tests
pytest tests/ -m gpu

# Only integration tests
pytest tests/ -m integration
```

---

## Writing Tests

### Basic Test Structure

```python
# tests/unit/geometry/test_poincare.py
import pytest
import torch
from src.geometry import poincare_distance, exp_map_zero

class TestPoincareDistance:
    """Tests for poincare_distance function."""

    def test_distance_to_self_is_zero(self):
        """Distance from point to itself should be zero."""
        x = torch.tensor([0.3, 0.4])
        d = poincare_distance(x, x)
        assert torch.allclose(d, torch.tensor(0.0), atol=1e-6)

    def test_distance_is_symmetric(self):
        """d(x,y) should equal d(y,x)."""
        x = torch.tensor([0.1, 0.2])
        y = torch.tensor([0.3, 0.4])
        d_xy = poincare_distance(x, y)
        d_yx = poincare_distance(y, x)
        assert torch.allclose(d_xy, d_yx)

    def test_triangle_inequality(self):
        """d(x,z) <= d(x,y) + d(y,z)."""
        x = torch.tensor([0.1, 0.0])
        y = torch.tensor([0.0, 0.1])
        z = torch.tensor([-0.1, 0.0])

        d_xz = poincare_distance(x, z)
        d_xy = poincare_distance(x, y)
        d_yz = poincare_distance(y, z)

        assert d_xz <= d_xy + d_yz + 1e-5
```

### Using Fixtures

```python
# tests/conftest.py
import pytest
import torch
from src.models import TernaryVAE
from src.config import TrainingConfig

@pytest.fixture
def config():
    """Default training config for tests."""
    return TrainingConfig(
        epochs=1,
        batch_size=16,
        geometry={"latent_dim": 8, "curvature": 1.0},
    )

@pytest.fixture
def model(config):
    """Small model for fast tests."""
    return TernaryVAE(
        input_dim=19683,
        latent_dim=config.geometry.latent_dim,
        hidden_dims=[64, 32],  # Small for speed
    )

@pytest.fixture
def sample_batch():
    """Sample input batch."""
    x = torch.randint(0, 19683, (16,))
    x_onehot = torch.zeros(16, 19683)
    x_onehot.scatter_(1, x.unsqueeze(1), 1)
    return x_onehot, x

# Usage in tests
def test_forward_pass(model, sample_batch):
    x_onehot, x = sample_batch
    outputs = model(x_onehot)
    assert outputs["reconstruction"].shape == (16, 19683)
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0])
def test_different_curvatures(curvature):
    """Test with multiple curvature values."""
    x = torch.tensor([0.1, 0.2])
    y = torch.tensor([0.3, 0.4])
    d = poincare_distance(x, y, curvature=curvature)
    assert d > 0
    assert not torch.isnan(d)

@pytest.mark.parametrize("latent_dim,expected_params", [
    (8, 100000),   # Approximate
    (16, 200000),
    (32, 400000),
])
def test_model_size(latent_dim, expected_params):
    """Test model has expected parameter count."""
    model = TernaryVAE(input_dim=19683, latent_dim=latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == pytest.approx(expected_params, rel=0.2)
```

### Testing Exceptions

```python
def test_invalid_curvature_raises():
    """Negative curvature should raise ValueError."""
    with pytest.raises(ValueError, match="Curvature must be positive"):
        poincare_distance(x, y, curvature=-1.0)

def test_invalid_config_raises():
    """Invalid config should raise ConfigValidationError."""
    from src.config import GeometryConfig, ConfigValidationError

    with pytest.raises(ConfigValidationError):
        GeometryConfig(curvature=-1.0)
```

### Markers

```python
import pytest

@pytest.mark.slow
def test_full_training():
    """Test that takes several minutes."""
    # Train for 100 epochs
    pass

@pytest.mark.gpu
def test_cuda_forward():
    """Test requiring CUDA GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = TernaryVAE(input_dim=19683, latent_dim=16).cuda()
    x = torch.randn(16, 19683).cuda()
    outputs = model(x)
    assert outputs["z_hyperbolic"].device.type == "cuda"

@pytest.mark.integration
def test_train_and_evaluate():
    """End-to-end integration test."""
    pass
```

---

## Test Categories

### Unit Tests

Test individual functions/classes in isolation:

```python
class TestExpMapZero:
    """Unit tests for exp_map_zero."""

    def test_origin_maps_to_origin(self):
        """exp_map(0) = 0."""
        v = torch.zeros(16)
        result = exp_map_zero(v, curvature=1.0)
        assert torch.allclose(result, torch.zeros(16), atol=1e-6)

    def test_stays_in_ball(self):
        """Result norm should be < 1."""
        v = torch.randn(100, 16) * 10  # Large vectors
        result = exp_map_zero(v, curvature=1.0)
        norms = result.norm(dim=1)
        assert (norms < 1.0).all()
```

### Integration Tests

Test multiple modules together:

```python
@pytest.mark.integration
def test_training_loop():
    """Test training loop runs without error."""
    config = TrainingConfig(epochs=2, batch_size=16)
    model = TernaryVAE(input_dim=19683, latent_dim=8)
    optimizer = torch.optim.Adam(model.parameters())
    registry = create_registry_from_training_config(config)

    # Create dummy data
    data = torch.randint(0, 19683, (100,))
    data_oh = torch.zeros(100, 19683).scatter_(1, data.unsqueeze(1), 1)
    loader = DataLoader(TensorDataset(data_oh, data), batch_size=16)

    # Train
    initial_loss = None
    for epoch in range(config.epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            result = registry.compose(outputs, batch_y)

            if initial_loss is None:
                initial_loss = result.total.item()

            result.total.backward()
            optimizer.step()

    final_loss = result.total.item()
    assert final_loss < initial_loss  # Loss should decrease
```

### Property-Based Tests

```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=0.01, max_value=0.99))
def test_projection_stays_in_ball(radius):
    """Property: projection always stays in ball."""
    x = torch.randn(16) * radius
    from src.geometry import project_to_poincare

    result = project_to_poincare(x, max_radius=0.95, curvature=1.0)
    assert result.norm() < 1.0
```

---

## Mocking

```python
from unittest.mock import Mock, patch

def test_checkpoint_saves(tmp_path):
    """Test checkpoint callback saves files."""
    from src.training.callbacks import CheckpointCallback

    model = Mock()
    model.state_dict.return_value = {"weight": torch.tensor([1.0])}

    callback = CheckpointCallback(save_dir=str(tmp_path), save_interval=1)
    callback.on_epoch_end(0, {"val_loss": 0.5}, model)

    assert (tmp_path / "checkpoint_epoch_0.pt").exists()

@patch('src.observability.metrics_buffer.SummaryWriter')
def test_metrics_logged(mock_writer):
    """Test metrics are logged to TensorBoard."""
    from src.observability import MetricsBuffer

    buffer = MetricsBuffer(tensorboard_dir="runs/")
    buffer.add("loss", 0.5, step=0)
    buffer.flush()

    mock_writer.return_value.add_scalar.assert_called()
```

---

## CI/CD Integration

Tests run automatically on:
- Every push to any branch
- Every pull request to `main`

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/ -v --tb=short
```

---

## Debugging Tests

### Verbose Output

```bash
pytest tests/ -v -s  # Show print statements
```

### Drop to Debugger

```bash
pytest tests/ --pdb  # Drop to pdb on failure
```

### Run Last Failed

```bash
pytest tests/ --lf  # Run only last failed tests
```

---

*See also: [[Contributing-Guide]], [[Configuration]]*
