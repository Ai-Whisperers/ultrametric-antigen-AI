# Testing Guide

> **Test suite organization and best practices.**

---

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── unit/                 # Unit tests (mirror src/)
│   ├── models/
│   │   ├── test_base_vae.py           # 33 tests
│   │   ├── test_epistasis_module.py   # 32 tests
│   │   └── test_structure_aware_vae.py # 35 tests
│   ├── losses/
│   │   └── test_epistasis_loss.py     # 29 tests
│   ├── diseases/
│   │   └── test_uncertainty_integration.py # 21 tests
│   ├── training/
│   │   └── test_transfer_pipeline.py  # 30 tests
│   └── encoders/
│       └── test_alphafold_encoder.py  # 18 tests
├── integration/          # Cross-module tests
│   └── test_full_pipeline.py          # 33 tests
└── fixtures/             # Test data
```

---

## Running Tests

### All Tests

```bash
pytest tests/
```

### Specific Tests

```bash
# Single file
pytest tests/unit/models/test_base_vae.py -v

# Single test
pytest tests/unit/models/test_base_vae.py::TestBaseVAE::test_encode -v

# By marker
pytest tests/ -m "not slow"
```

### With Coverage

```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html
```

---

## Test Categories

### Unit Tests

Test individual functions/methods in isolation:

```python
def test_reparameterize_shape():
    model = SimpleVAE(input_dim=64, latent_dim=16)
    mu = torch.zeros(4, 16)
    logvar = torch.zeros(4, 16)
    z = model.reparameterize(mu, logvar)
    assert z.shape == (4, 16)
```

### Integration Tests

Test module interactions:

```python
def test_full_pipeline():
    # Create model
    model = StructureAwareVAE(...)

    # Create data
    sequences = [...]
    structures = [...]

    # Run full pipeline
    results = model(sequences, structures)

    # Verify end-to-end
    assert "reconstruction" in results
    assert "z" in results
```

---

## Fixtures

### Shared Fixtures (`conftest.py`)

```python
@pytest.fixture
def sample_sequences():
    return ["MKTIIALSYILCLVFA", "MKTIIALSYILCLVFG"]

@pytest.fixture
def sample_model():
    return SimpleVAE(input_dim=64, latent_dim=16)

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Using Fixtures

```python
def test_model_forward(sample_model, sample_sequences):
    x = encode_sequences(sample_sequences)
    output = sample_model(x)
    assert output is not None
```

---

## Test Statistics

| Category | Tests | Pass Rate |
|:---------|:------|:----------|
| Unit tests | 198 | 97.5% |
| Integration tests | 33 | 97.0% |
| **Total** | **231** | **97.4%** |

---

## Best Practices

### Do

- Test one thing per test
- Use descriptive test names
- Use fixtures for shared setup
- Test edge cases
- Keep tests fast

### Don't

- Don't test implementation details
- Don't use random seeds without setting them
- Don't skip tests without reason
- Don't ignore flaky tests

---

_Last updated: 2025-12-28_
