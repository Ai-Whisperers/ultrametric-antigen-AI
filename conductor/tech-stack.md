# Technology Stack: Ternary VAE v5.11

## Core Language & Runtime
- **Python (>= 3.8)**: Primary development language.
- **PyTorch (>= 2.0.0)**: Core deep learning framework for model development and training.

## Specialized Libraries
- **geoopt (>= 0.5.0)**: Essential for Riemannian optimization and operations within the Poincaré ball (hyperbolic space).
- **NumPy (>= 1.24.0) & SciPy (>= 1.10.0)**: Numerical computation and scientific analysis.
- **scikit-learn (>= 1.2.0)**: Used for evaluation metrics and manifold analysis.

## Data & Configuration
- **PyYAML (>= 6.0)**: Standard for project configuration and hyperparameter management.
- **pandas (>= 2.0.0)**: Data manipulation and analysis for results and benchmarking.

## Monitoring & Observability
- **TensorBoard (>= 2.13.0)**: Primary visualization platform for training metrics and system dynamics.
- **tqdm (>= 4.65.0)**: Progress tracking for training loops and long-running scripts.
- **colorama & tabulate**: Enhanced CLI output for readability and reporting.

## Quality Assurance & DevOps
- **pytest (>= 7.3.0)**: Core testing framework.
- **pytest-cov (>= 4.1.0)**: Tool for measuring and reporting test coverage.
- **mypy**: Static type checking for code robustness.
- **ruff**: Fast linting and formatting (as indicated by `ruff.toml`).
