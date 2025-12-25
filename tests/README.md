# Testing Guide

Comprehensive testing infrastructure for the Ternary VAEs Bioinformatics project.

## Test Structure

```
tests/
├── conftest.py           # Root fixtures (shared across all tests)
├── README.md             # This file
│
├── core/                 # Test infrastructure
│   ├── assertions.py     # Custom assertion classes
│   ├── base.py           # Base test classes (LossTestCase, ModelTestCase)
│   ├── helpers.py        # Test helpers and mocks
│   ├── builders/         # Test data builders
│   └── matchers/         # Custom matchers (geometry, etc.)
│
├── factories/            # Object factories for test data
│   ├── base.py           # BaseFactory abstract class
│   ├── data.py           # TernaryOperationFactory
│   ├── models.py         # ModelConfigFactory
│   └── embeddings.py     # Embedding factories
│
├── harnesses/            # Test harnesses for complex testing
│   └── model_harness.py  # ModelTestHarness
│
├── unit/                 # Unit tests (fast, isolated)
│   ├── conftest.py       # Unit test fixtures
│   ├── core/             # Tests for src/core/
│   ├── geometry/         # Tests for src/geometry/
│   ├── losses/           # Tests for src/losses/
│   │   └── conftest.py   # Loss-specific fixtures
│   ├── models/           # Tests for src/models/
│   └── training/         # Tests for src/training/
│
└── suites/               # Organized test suites
    ├── unit/             # Grouped unit tests
    ├── integration/      # Integration tests
    └── e2e/              # End-to-end tests
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run only unit tests
pytest tests/unit/

# Run specific test file
pytest tests/unit/losses/test_dual_vae_loss.py

# Run tests matching pattern
pytest -k "test_initialization"

# Run with verbose output
pytest -v

# Run fast tests only (exclude slow)
pytest -m "not slow"
```

### Coverage Reports

```bash
# Terminal report
pytest tests/ --cov=src --cov-report=term

# HTML report
pytest tests/ --cov=src --cov-report=html
# Then open htmlcov/index.html

# XML report (for CI)
pytest tests/ --cov=src --cov-report=xml
```

## Fixtures

### Session-Scoped (Expensive, Reused)

```python
@pytest.fixture(scope="session")
def device():
    """Returns 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session")
def poincare():
    """Returns a PoincaréBall manifold instance."""
    from src.geometry.poincare import get_manifold
    return get_manifold(c=1.0)
```

### Function-Scoped (Fresh Each Test)

```python
@pytest.fixture
def ternary_ops(device):
    """Returns a batch of valid ternary operations."""
    return torch.randint(-1, 2, (32, 9), device=device).float()

@pytest.fixture
def vae_outputs(device):
    """Returns mock VAE outputs for loss testing."""
    return {
        'logits_A': torch.randn(32, 9, 3, device=device),
        'mu_A': torch.randn(32, 16, device=device),
        # ... more
    }
```

## Custom Assertions

Use domain-specific assertions from `tests/core/assertions.py`:

```python
from tests.core.assertions import TensorAssertions, GeometryAssertions

# Tensor properties
TensorAssertions.assert_shape(tensor, (32, 16))
TensorAssertions.assert_finite(tensor, "Loss should be finite")
TensorAssertions.assert_in_range(tensor, 0.0, 1.0)

# Geometry
GeometryAssertions.assert_on_poincare_disk(points, max_norm=0.95)
GeometryAssertions.assert_distance_symmetry(dist_ab, dist_ba)
```

## Base Test Classes

Use base classes from `tests/core/base.py` for standard patterns:

```python
from tests.core.base import LossTestCase

class TestMyLoss(LossTestCase):
    loss_class = MyLoss
    loss_kwargs = {'margin': 0.1}

    def _get_test_inputs(self, batch_size=32, requires_grad=False):
        z = torch.randn(batch_size, 16, requires_grad=requires_grad)
        indices = torch.randint(0, 19683, (batch_size,))
        return (z, indices)
```

## Factories

Use factories from `tests/factories/`:

```python
from tests.factories.data import TernaryOperationFactory
from tests.factories.models import ModelConfigFactory

ops = TernaryOperationFactory.build(batch_size=32, device="cpu")
config = ModelConfigFactory.minimal()
```

## Markers

```python
@pytest.mark.slow       # Long-running tests
@pytest.mark.gpu        # GPU-required tests
@pytest.mark.integration  # Integration tests
@pytest.mark.e2e        # End-to-end tests
```

Run with markers:
```bash
pytest -m "not slow"           # Skip slow tests
pytest -m "gpu"                # Only GPU tests
pytest -m "integration or e2e" # Integration and E2E
```

## Coverage Targets

| Module | Current | Target |
|--------|---------|--------|
| `core/ternary.py` | 98% | 95% |
| `geometry/poincare.py` | 100% | 95% |
| `losses/dual_vae_loss.py` | 78% | 80% |
| `losses/components.py` | 63% | 70% |
| `losses/registry.py` | 92% | 90% |

## Configuration

- **pytest.ini**: Main configuration (python path, test discovery, markers)
- **.coveragerc**: Coverage configuration (omits, branch tracking)
