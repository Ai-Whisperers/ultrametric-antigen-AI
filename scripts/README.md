# Scripts Directory

> **Entry points for training, analysis, and utilities**

**Last Updated:** December 26, 2025

---

## Directory Structure

```
scripts/
├── train.py                    # Main training entry point
├── train_codon_vae_hiv.py      # HIV-specific codon VAE training
├── analyze_all_datasets.py     # Comprehensive dataset analysis
├── clinical_applications.py    # Clinical decision support
├── research_discoveries.py     # Research findings pipeline
│
├── analysis/           # Code analysis and auditing
├── benchmark/          # Performance benchmarking
├── clinical/           # Clinical decision support tools
├── docs/               # Documentation generation
├── epsilon_vae/        # Epsilon VAE training (experimental)
├── eval/               # Model evaluation
├── hiv/                # HIV-specific analysis pipelines
├── ingest/             # Data ingestion utilities
├── legal/              # License and copyright tools
├── literature/         # Literature implementation scripts
├── maintenance/        # Codebase maintenance utilities
├── setup/              # Environment setup
└── visualization/      # Visualization generation
```

---

## Script Categories

### HIV Analysis (`hiv/`)
Primary HIV research pipelines:
- `clinical_applications.py` - Generate clinical decision support
- `research_discoveries.py` - Run 5 research discovery pipelines
- `analyze_all_datasets.py` - Comprehensive dataset analysis
- `train_codon_vae_hiv.py` - Train HIV-specific codon VAE
- `download_hiv_datasets.py` - Download public HIV datasets
- `validate_hiv_setup.py` - Validate HIV analysis environment

### Clinical Tools (`clinical/`)
- `clinical_dashboard.py` - Interactive clinical dashboard
- `clinical_integration.py` - Integration with clinical systems

### Literature Analysis (`literature/`)
- `advanced_literature_implementations.py` - Advanced implementations
- `literature_implementations.py` - Standard implementations
- `cutting_edge_implementations.py` - Latest research implementations
- `advanced_research.py` - Advanced research experiments

### Maintenance (`maintenance/`)
- `maintain_codebase.py` - General codebase maintenance
- `doc_auditor.py` - Documentation auditing
- `doc_builder.py` - Documentation generation
- `validate_all_implementations.py` - Validate all implementations
- `project_diagrams_generator.py` - Generate architecture diagrams

### Training (root level)
- `train.py` - Main training entry point
- `train_codon_vae_hiv.py` - HIV-specific codon VAE training

### Visualization (`visualization/`)
- Various visualization scripts for manifolds, Calabi-Yau, etc.

---

## Quick Start

```bash
# Train the main model
python scripts/train.py

# Train HIV-specific codon VAE
python scripts/train_codon_vae_hiv.py

# Run clinical applications
python scripts/clinical_applications.py

# Run research discoveries
python scripts/research_discoveries.py

# Comprehensive dataset analysis
python scripts/analyze_all_datasets.py
```

---

## Notes

- Main training scripts are at the root level for easy access
- Domain-specific scripts are organized into subdirectories
- See individual subdirectory READMEs for detailed documentation
- See [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) for full project layout
