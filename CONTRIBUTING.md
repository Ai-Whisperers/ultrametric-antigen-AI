# Contributing to Ternary VAEs Bioinformatics

Thank you for your interest in contributing! This project welcomes contributions from researchers, students, and the open-source community.

## License Agreement

By contributing to this project, you agree that your contributions will be licensed under the **PolyForm Noncommercial License 1.0.0**. This means:

- Your contributions remain free for academic and research use
- Commercial entities cannot use your contributions without a separate license
- You retain copyright to your contributions

## Getting Started

### Prerequisites

- Python 3.8+ (3.10+ recommended)
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black isort flake8 mypy ruff

# Run tests to verify setup
pytest
```

## Code Style

We follow these conventions:

- **Formatter**: `black` (line length 120)
- **Import sorting**: `isort`
- **Linting**: `ruff` and `flake8`
- **Type checking**: `mypy`

### Before Submitting

```bash
# Format code
black src/ tests/ --line-length 120
isort src/ tests/

# Run linting
ruff check src/
flake8 src/

# Run type checking
mypy src/

# Run tests
pytest
```

## Types of Contributions

### Bug Reports

Open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (Python version, PyTorch version, GPU)

### Feature Requests

Open an issue describing:
- The problem you're trying to solve
- Your proposed solution
- Alternative approaches considered

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Documentation

- Fix typos or unclear explanations
- Add examples
- Improve docstrings
- Write tutorials

### Research Contributions

- Share experimental results
- Propose new loss functions or architectures
- Contribute bioinformatics datasets
- Write analysis notebooks

## Pull Request Process

1. **Title**: Use conventional commit format
   - `feat: Add new loss function`
   - `fix: Correct gradient computation`
   - `docs: Update README`
   - `test: Add coverage tests`

2. **Description**: Explain what and why, not just how

3. **Tests**: Include tests for new features

4. **Documentation**: Update relevant docs

5. **Review**: Address reviewer feedback promptly

## Project Structure

```
src/
├── core/           # Foundational ternary space
├── data/           # Data generation and loading
├── models/         # Neural architectures
├── losses/         # Loss functions
├── training/       # Training orchestration
├── metrics/        # Evaluation metrics
├── observability/  # Logging and monitoring
└── utils/          # Utilities
```

## Architecture Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Frozen Encoder**: v5.5 encoder is read-only for 100% coverage guarantee
3. **Hyperbolic Geometry**: All latent operations use Poincare ball
4. **Reproducibility**: Deterministic seeds and checkpointing

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Other unprofessional conduct

### Enforcement

Violations may be reported to support@aiwhisperers.com. All complaints will be reviewed and investigated.

## Questions?

- Open a GitHub issue for technical questions
- Email support@aiwhisperers.com for licensing questions
- Check existing issues and documentation first

## Recognition

Contributors will be acknowledged in:
- AUTHORS.md
- Release notes
- Academic publications (where appropriate)

Thank you for contributing to open science!
