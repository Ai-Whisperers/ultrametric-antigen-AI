# Contributing Guide

> **How to contribute to Ternary VAE.**

---

## Getting Started

1. **Fork** the repository
2. **Clone** your fork
3. **Create** a feature branch
4. **Make** changes with tests
5. **Submit** a pull request

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev tools

# Install pre-commit hooks
pre-commit install
```

---

## Code Standards

### Style

- **Formatter**: Black (line-length 120)
- **Import sorting**: isort
- **Linting**: ruff
- **Type checking**: mypy

```bash
# Run all checks
black src/ tests/ --line-length 120
isort src/ tests/
ruff check src/ tests/
mypy src/
```

### Docstrings

Use Google-style docstrings:

```python
def compute_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distance between x and y.

    Args:
        x: First tensor of shape (batch, dim).
        y: Second tensor of shape (batch, dim).

    Returns:
        Distance tensor of shape (batch,).

    Raises:
        ValueError: If shapes don't match.
    """
```

### Type Hints

Required for all functions:

```python
from typing import Dict, List, Optional, Tuple

def analyze(
    sequences: List[str],
    drugs: Optional[List[str]] = None,
) -> Dict[str, float]:
    ...
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/unit/models/test_base_vae.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

```python
import pytest
from src.models.base_vae import BaseVAE

class TestBaseVAE:
    def test_encode_returns_tuple(self):
        model = SimpleVAE(input_dim=64, latent_dim=16)
        x = torch.randn(4, 64)
        mu, logvar = model.encode(x)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)
```

---

## Pull Request Process

### Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Linting passes (`ruff check src/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation updated
- [ ] Commit messages are descriptive

### Commit Messages

Use conventional commits:

```
feat: Add uncertainty quantification to HIVAnalyzer
fix: Correct pLDDT weighting in structure encoder
docs: Update architecture documentation
test: Add tests for transfer pipeline
refactor: Simplify BaseVAE interface
```

---

## License

By contributing, you agree that your contributions will be licensed under the project's PolyForm Non-Commercial 1.0.0 license.

---

_Last updated: 2025-12-28_
