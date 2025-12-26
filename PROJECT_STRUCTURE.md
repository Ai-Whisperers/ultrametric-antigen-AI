# Project Structure Guide

This document provides an overview of the directory structure for the Ternary VAE Bioinformatics project.

**Last Updated**: 2025-12-26

---

## Quick Reference

| Directory | Purpose |
|-----------|---------|
| `src/` | Core Python library (models, losses, training) |
| `scripts/` | Executable scripts for training, analysis, visualization |
| `tests/` | Comprehensive test suite (mirrors src/ structure) |
| `configs/` | YAML/JSON runtime configuration files |
| `data/` | Raw research datasets |
| `research/` | Research experiments and bioinformatics analysis |
| `results/` | Generated outputs: training runs, analysis results |
| `runs/` | Training checkpoints and run artifacts |
| `outputs/` | Visualization outputs |
| `reports/` | Generated reports and audit results |
| `DOCUMENTATION/` | Comprehensive multi-tier documentation |

---

## Source Code (`src/`)

The core Python package implementing the Ternary VAE architecture.

```
src/
├── config/           # Configuration classes and schemas
├── core/             # Ternary operation interfaces
├── models/           # VAE architectures (V5.11, SwarmVAE, etc.)
├── losses/           # Loss functions (p-adic, hyperbolic, domain-specific)
├── training/         # Training infrastructure and callbacks
├── optimizers/       # Custom optimizers (Riemannian, multi-objective)
├── encoders/         # Specialized encoders (codon, circadian, surface)
├── geometry/         # Hyperbolic geometry (Poincare ball)
├── biology/          # Biological constants (amino acids, codons)
├── analysis/         # Analysis modules (CRISPR, immunology)
├── diseases/         # Disease-specific models (MS, RA, Long COVID)
├── evolution/        # Viral evolution tracking
├── validation/       # Scientific validation modules
├── metrics/          # Evaluation metrics
├── utils/            # Utilities and helpers
├── visualization/    # Plotting and diagram generation
├── observability/    # Logging, metrics buffering
├── factories/        # Factory patterns for model/loss creation
├── artifacts/        # Checkpoint management
├── data/             # Dataset classes and loaders
├── objectives/       # Drug optimization objectives
├── stability/        # mRNA stability analysis
└── quantum/          # Quantum biology descriptors
```

### Key Modules

- **`models/ternary_vae.py`**: Canonical V5.11 architecture
- **`losses/dual_vae_loss.py`**: Complete loss system
- **`training/trainer.py`**: Main training loop
- **`biology/codons.py`**: Single source of truth for codon encoding

---

## Scripts (`scripts/`)

Executable scripts organized by function.

```
scripts/
├── train/            # Training scripts
│   └── train.py      # Main training entry point
├── analysis/         # Code analysis and auditing
├── visualization/    # Visualization generation
├── hiv/              # HIV-specific analysis scripts
├── clinical/         # Clinical application scripts
├── eval/             # Evaluation and validation
├── benchmark/        # Performance benchmarking
├── docs/             # Documentation generation
├── ingest/           # Data ingestion
├── setup/            # Repository setup
├── literature/       # Literature implementation
├── maintenance/      # Codebase maintenance
└── epsilon_vae/      # Epsilon VAE experiments
```

### Common Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train/train.py` | Train V5.11 models |
| `scripts/train_codon_vae_hiv.py` | HIV-specific codon VAE training |
| `scripts/analyze_all_datasets.py` | Comprehensive dataset analysis |
| `scripts/clinical_applications.py` | Clinical decision support |
| `scripts/research_discoveries.py` | Research findings pipeline |

---

## Tests (`tests/`)

Comprehensive test suite mirroring the `src/` structure.

```
tests/
├── conftest.py       # Shared pytest fixtures
├── unit/             # Unit tests (mirrors src/)
│   ├── test_models/
│   ├── test_losses/
│   ├── test_training/
│   └── ...
├── integration/      # Cross-module integration tests
├── e2e/              # End-to-end scientific validation
├── fixtures/         # Test data and fixtures
├── factories/        # Test factories
├── core/             # Test utilities and helpers
└── harnesses/        # Test harness infrastructure
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Configuration (`configs/`)

Runtime configuration files for training and experiments.

```
configs/
├── ternary.yaml          # Standard training config
├── ternary_fast_test.yaml # Quick test config
├── env.example           # Environment template
└── archive/              # Archived configurations
```

### Configuration vs Code

| Location | Purpose |
|----------|---------|
| `configs/` | YAML/JSON runtime parameters (learning rate, epochs, etc.) |
| `src/config/` | Python configuration classes and schema definitions |

---

## Data Directories

### Raw Data (`data/`)

Research datasets and external data sources.

```
data/
├── external/         # External data sources
│   ├── github/       # GitHub datasets
│   ├── huggingface/  # HuggingFace datasets
│   ├── kaggle/       # Kaggle datasets
│   └── zenodo/       # Zenodo datasets
└── research/
    └── datasets/     # Stanford HIV drug resistance data
```

### Data Access Layer (`data_access/`)

Client library for accessing research data sources.

```
data_access/
├── clients/          # API clients (NCBI, UniProt, HIVDB, etc.)
├── config/           # Client configuration
├── integration/      # Data integration utilities
├── notebooks/        # Example notebooks
└── tests/            # Client tests
```

---

## Output Directories

### Results (`results/`)

Primary output directory for generated results.

```
results/
├── training/              # Training outputs
├── training_runs/         # Historical run data
├── run_history/           # Run metadata
├── checkpoints/           # Model checkpoints
├── clinical_applications/ # Clinical decision support outputs
├── research_discoveries/  # Research findings
├── alphafold_inputs/      # AlphaFold job inputs
├── alphafold_predictions/ # AlphaFold predictions
├── benchmarks/            # Benchmark results
├── logs/                  # Training logs
├── metrics/               # Metric outputs
└── validation/            # Validation results
```

### Training Runs (`runs/`)

Active training run checkpoints and artifacts.

### Visualizations (`outputs/`)

Generated visualization outputs.

```
outputs/
└── viz/              # Visualization files (PNG, HTML, etc.)
```

### Development (`sandbox-training/`)

Development and testing artifacts (not for production).

---

## Research (`research/`)

Research experiments and bioinformatics analysis.

```
research/
├── alphafold3/           # AlphaFold3 integration
└── bioinformatics/
    ├── codon_encoder_research/
    │   ├── hiv/          # HIV analysis (200K+ sequences)
    │   ├── neurodegeneration/
    │   ├── rheumatoid_arthritis/
    │   └── sars_cov2/
    ├── genetic_code/     # Genetic code analysis
    └── spectral_analysis_over_models/
```

---

## Documentation (`DOCUMENTATION/`)

Comprehensive multi-tier documentation system.

```
DOCUMENTATION/
├── 00_*.md                    # Navigation and quick start
├── 01_PROJECT_KNOWLEDGE_BASE/ # Theory and foundations
├── 02_PROJECT_MANAGEMENT/     # Tasks, roadmaps, metrics
├── 03_PRESENTATION_TIERS/     # Tiered documentation
├── 04_DEVELOPMENT/            # Development guides
├── 05_STANDARDS/              # Code and doc standards
└── 06_DIAGRAMS/               # Visual diagrams
```

---

## Support Directories

### Reports (`reports/`)

Generated reports and audit results.

### Conductor (`conductor/`)

Project orchestration and guidelines.

```
conductor/
├── product.md            # Product overview
├── tech-stack.md         # Technology stack
├── workflow.md           # Development workflow
├── code_styleguides/     # Style guides
└── tracks/               # Development tracks
```

### Hidden Directories

| Directory | Purpose |
|-----------|---------|
| `.github/` | GitHub Actions workflows |
| `.vscode/` | VS Code settings |
| `.claude/` | Claude Code configuration |
| `.cursor/` | Cursor IDE configuration |

---

## File Naming Conventions

### Python Files

- Modules: `snake_case.py` (e.g., `ternary_vae.py`)
- Tests: `test_*.py` (e.g., `test_ternary_vae.py`)

### Configuration Files

- YAML: `snake_case.yaml`
- JSON: `snake_case.json`

### Documentation

- Markdown: `UPPER_CASE.md` for top-level, `Title_Case.md` for sections

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
- [README.md](README.md) - Getting started guide
- [DOCUMENTATION/00_MASTER_INDEX.md](DOCUMENTATION/00_MASTER_INDEX.md) - Full documentation index
