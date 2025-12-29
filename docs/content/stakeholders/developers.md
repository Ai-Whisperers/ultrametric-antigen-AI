# For Developers & Engineers

> **Architecture, setup, and contribution guide.**

---

## Quick Start

```bash
# Clone
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics

# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run tests
pytest tests/

# Train a model
python scripts/train/train.py --config configs/ternary.yaml
```

---

## Architecture Overview

See [Architecture](../architecture/README.md) for full details.

### Key Modules

```
src/
├── models/           # VAE architectures
│   ├── base_vae.py   # Base class (start here!)
│   └── ternary_vae.py # Production model
├── losses/           # Loss functions
├── training/         # Training infrastructure
├── diseases/         # Disease analyzers
└── encoders/         # Specialized encoders
```

### Entry Points

| Script | Purpose |
|:-------|:--------|
| `scripts/train/train.py` | Main training |
| `scripts/analyze_all_datasets.py` | Dataset analysis |
| `scripts/clinical_applications.py` | Clinical outputs |

---

## Development Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU)

### Code Style

```bash
# Format
black src/ tests/ --line-length 120
isort src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific tests
pytest tests/unit/models/test_base_vae.py -v
```

---

## Contributing

### Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run linting and tests
5. Submit pull request

### Code Standards

- Type hints for all functions
- Google-style docstrings
- Tests for new functionality
- No secrets in code

### Pull Request Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Linting passes (`ruff check src/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation updated if needed
- [ ] Commit messages are descriptive

---

## Key Design Decisions

### Single Source of Truth

- Biology constants: `src/biology/codons.py`
- Configuration: `src/config/`
- Documentation: `docs/content/`

### Inheritance

```
BaseVAE
├── SimpleVAE
├── TernaryVAE
├── StructureAwareVAE
└── ... (19+ variants)
```

### Dependency Injection

```python
trainer = TernaryVAETrainer(
    model,
    config,
    device,
    monitor=custom_monitor,  # Injected
)
```

---

## API Documentation

Auto-generated from source:
- [docs/source/api/](../../source/api/)

Build locally:
```bash
cd docs
make html
# Open _build/html/index.html
```

---

## Debugging

### Common Issues

| Issue | Solution |
|:------|:---------|
| CUDA out of memory | Reduce batch size in config |
| Import errors | Check virtual environment activation |
| Test failures | Run `pytest -v` for details |

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Contact

- GitHub Issues: Bug reports, feature requests
- Pull Requests: Contributions welcome!

---

_Last updated: 2025-12-28_
