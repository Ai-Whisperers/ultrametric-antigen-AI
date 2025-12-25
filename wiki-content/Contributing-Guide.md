# Contributing Guide

How to contribute to Ternary VAE Bioinformatics.

---

## Welcome Contributors!

We welcome contributions of all kinds:
- Bug reports and fixes
- Feature requests and implementations
- Documentation improvements
- Test coverage improvements
- Performance optimizations

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics
git remote add upstream https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
```

### 2. Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Verify setup
pytest tests/ -v --tb=short
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Contributor License Agreement (CLA)

Before your first PR can be merged, you must sign the CLA:

1. Read [CLA.md](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics/blob/main/LEGAL_AND_IP/CLA.md)
2. Add your name to the CLA signatories
3. Include in your PR

This ensures all contributions can be used under the project license.

---

## Code Style

### Python Style

We use automated formatting:

```bash
# Format code
black src/ tests/ --line-length 120
isort src/ tests/

# Or use the skill
/format
```

**Guidelines**:
- Line length: 120 characters
- Use type hints for all function signatures
- Google-style docstrings
- Prefer explicit over implicit

### Docstring Format

```python
def poincare_distance(x: torch.Tensor, y: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Compute distance between points in the Poincare ball.

    Uses the formula: d(x,y) = (2/√κ) * arctanh(√κ * ||(-x)⊕y||)

    Args:
        x: First point, shape (*, D).
        y: Second point, shape (*, D).
        curvature: Poincare ball curvature (must be positive).

    Returns:
        Pairwise distances, shape (*,).

    Raises:
        ValueError: If curvature is not positive.

    Example:
        >>> x = torch.tensor([0.1, 0.2])
        >>> y = torch.tensor([0.3, 0.4])
        >>> d = poincare_distance(x, y)
    """
```

### Import Order

```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party
import torch
import numpy as np
from torch import nn

# 3. Local
from src.config import TrainingConfig
from src.geometry import poincare_distance
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/unit/models/test_ternary_vae.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Fast tests only (skip slow)
pytest tests/ -m "not slow"
```

### Writing Tests

```python
# tests/unit/geometry/test_distance.py
import pytest
import torch
from src.geometry import poincare_distance

class TestPoincareDistance:
    """Tests for poincare_distance function."""

    def test_distance_to_self_is_zero(self):
        """Distance from point to itself should be zero."""
        x = torch.tensor([0.3, 0.4])
        d = poincare_distance(x, x)
        assert torch.allclose(d, torch.tensor(0.0), atol=1e-6)

    def test_distance_is_symmetric(self):
        """d(x,y) == d(y,x)."""
        x = torch.tensor([0.1, 0.2])
        y = torch.tensor([0.3, 0.4])
        assert torch.allclose(poincare_distance(x, y), poincare_distance(y, x))

    def test_distance_at_origin(self):
        """Distance from origin should equal artanh(||x||)."""
        x = torch.tensor([0.5, 0.0])
        origin = torch.tensor([0.0, 0.0])
        d = poincare_distance(origin, x, curvature=1.0)
        expected = 2 * torch.atanh(x.norm())
        assert torch.allclose(d, expected, atol=1e-5)

    @pytest.mark.parametrize("curvature", [0.5, 1.0, 2.0])
    def test_different_curvatures(self, curvature):
        """Should work with different curvatures."""
        x = torch.tensor([0.1, 0.2])
        y = torch.tensor([0.3, 0.4])
        d = poincare_distance(x, y, curvature=curvature)
        assert d > 0
        assert not torch.isnan(d)
```

### Test Markers

```python
import pytest

@pytest.mark.slow
def test_full_training():
    """Test that takes > 1 minute."""
    pass

@pytest.mark.gpu
def test_cuda_operations():
    """Test requiring GPU."""
    pass

@pytest.mark.integration
def test_end_to_end():
    """Integration test."""
    pass
```

---

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guide (`black`, `isort`)
- [ ] All tests pass (`pytest tests/`)
- [ ] New code has tests
- [ ] Documentation updated if needed
- [ ] Commit messages are clear

### 2. PR Template

```markdown
## Summary
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CLA signed
```

### 3. Review Process

1. Automated checks run (lint, test, type check)
2. Code owners review
3. Address feedback
4. Approval and merge

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code change (no new feature/fix)
- `test`: Adding tests
- `chore`: Maintenance

**Examples**:
```
feat(geometry): add parallel transport function

Implements parallel transport between tangent spaces
in the Poincare ball model.

Closes #123
```

```
fix(losses): prevent NaN in KL divergence

Add epsilon clamping to prevent log(0).
```

---

## Issue Reporting

### Bug Reports

Include:
1. **Description**: What happened?
2. **Expected**: What should happen?
3. **Reproduction**: Minimal code to reproduce
4. **Environment**: Python version, OS, GPU
5. **Error message**: Full traceback

```markdown
## Bug Report

### Description
Training crashes with NaN loss after ~50 epochs.

### Expected Behavior
Training should complete without NaN.

### Reproduction
```python
from src.models import TernaryVAE
model = TernaryVAE(latent_dim=16)
# ... minimal code
```

### Environment
- Python: 3.11.5
- PyTorch: 2.1.0
- CUDA: 11.8
- OS: Ubuntu 22.04

### Error
```
RuntimeError: Loss is NaN
  File "src/training/loop.py", line 45
```
```

### Feature Requests

Include:
1. **Problem**: What are you trying to do?
2. **Proposal**: How would you solve it?
3. **Alternatives**: Other approaches considered
4. **Context**: Why is this important?

---

## Documentation

### Wiki Updates

1. Edit files in `wiki-content/`
2. Submit PR to main repo
3. After merge, wiki is updated

### Docstring Updates

Update docstrings in source files. API reference auto-generates (planned).

### Adding Examples

Add to `examples/` directory with clear README.

---

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v5.11.10`
4. Push tag: `git push origin v5.11.10`
5. Create GitHub Release

---

## Questions?

- **GitHub Discussions**: General questions
- **GitHub Issues**: Bugs and features
- **Email**: `support@aiwhisperers.com`

---

*Thank you for contributing!*
