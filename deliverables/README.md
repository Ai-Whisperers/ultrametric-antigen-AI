# Bioinformatics Deliverables

**P-adic Geometry for Computational Biology**

Version 2.0 | December 2025 | AI Whisperers

---

## Overview

This package provides bioinformatics tools built on p-adic (hyperbolic) geometric methods for:

- **Antimicrobial Peptide Design** - Multi-objective optimization with safety assessment
- **Protein Stability Prediction** - Rosetta-blind mutation detection
- **Arbovirus Surveillance** - Pan-arbovirus RT-PCR primer design
- **HIV Clinical Tools** - TDR screening and LA injectable selection

### Key Features

| Feature | Capability | Key Metric |
|---------|------------|------------|
| **P-adic Geometry** | Hierarchical sequence embedding | -0.83 Spearman |
| **Codon Encoder** | Force constant prediction | r = 0.86 |
| **HIV Analysis** | TDR + LA injectable eligibility | Clinical decision support |
| **AMP Design** | Multi-objective optimization | Pareto-optimal peptides |
| **Primer Design** | Pan-arbovirus RT-PCR | Cross-reactivity analysis |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all demos
python scripts/biotools.py demo-all

# Run specific demos
python scripts/biotools.py demo-hiv          # HIV resistance analysis
python scripts/biotools.py demo-amp          # AMP design
python scripts/biotools.py demo-primers      # Primer design
python scripts/biotools.py demo-stability    # Protein stability

# Generate showcase figures
python scripts/biotools.py showcase
```

### Python Quick Start

```python
# VAE Service - unified interface
from shared.vae_service import get_vae_service

vae = get_vae_service()
z = vae.encode_sequence("KLWKKWKKWLK")
stability = vae.get_stability_score(z)
print(f"Stability: {stability:.3f}")

# Peptide analysis
from shared.peptide_utils import compute_peptide_properties

props = compute_peptide_properties("GIGKFLHSAKKFGKAFVGEIMNS")
print(f"Charge: {props['net_charge']}, Hydrophobicity: {props['hydrophobicity']:.2f}")

# HIV resistance screening
from partners.hiv_research_package.src import TDRScreener

screener = TDRScreener()
result = screener.screen_patient(sequence, "P001")
print(f"TDR: {'Positive' if result.tdr_positive else 'Negative'}")
```

---

## Directory Structure

```
deliverables/
│
├── README.md                    # This file
├── requirements.txt             # Dependencies
│
├── demos/                       # Showcase demonstrations
│   ├── full_platform_demo.ipynb # Complete platform demo
│   └── figures/                 # Demo output figures
│
├── docs/                        # Documentation
│   ├── API_REFERENCE.md        # Complete API docs
│   ├── DELIVERABLES_IMPROVEMENT_PLAN.md
│   ├── HEMOLYSIS_PREDICTOR.md
│   ├── PRIMER_DESIGN.md
│   └── USAGE_EXAMPLES.md
│
├── shared/                      # Common Python modules
│   ├── vae_service.py          # Unified VAE interface
│   ├── config.py               # Configuration management
│   ├── peptide_utils.py        # Sequence analysis
│   ├── hemolysis_predictor.py  # Toxicity prediction
│   ├── constants.py            # Shared constants
│   └── tests/                  # Shared tests
│
├── scripts/                     # Command-line tools
│   ├── biotools.py             # CLI interface (demo-all, showcase)
│   └── generate_showcase_figures.py
│
├── checkpoints/                 # VAE checkpoints
│   └── README.md               # Checkpoint documentation
│
├── tests/                       # Test suite
│   ├── conftest.py             # Shared fixtures
│   ├── test_hiv_package.py
│   ├── test_alejandra_rojas.py
│   ├── test_e2e_demos.py       # End-to-end demo tests
│   └── ...
│
├── results/                     # Output results
│   └── figures/                # Generated figures
│
└── partners/                    # Research partner modules
    ├── hiv_research_package/   # HIV clinical tools
    │   ├── src/                # Complete library
    │   └── notebooks/
    ├── alejandra_rojas/        # Arbovirus surveillance
    │   ├── src/                # NCBI client, primer design
    │   └── notebooks/
    ├── carlos_brizuela/        # AMP optimization
    │   └── scripts/            # NSGA-II optimizer
    └── jose_colbes/            # Protein stability
        └── scripts/            # Geometric predictor
```

---

## Partner Research Packages

### Carlos Brizuela - AMP Navigator
**Focus:** Multi-objective antimicrobial peptide design

| Script | Description |
|--------|-------------|
| `B1_pathogen_specific_design.py` | WHO priority pathogen targeting |
| `B8_microbiome_safe_amps.py` | Microbiome-compatible peptides |
| `B10_synthesis_optimization.py` | Codon optimization for expression |

**Notebook:** `partners/carlos_brizuela/notebooks/brizuela_amp_navigator.ipynb`

---

### Jose Colbes - Rotamer Scoring
**Focus:** Protein side-chain stability and Rosetta-blind detection

| Script | Description |
|--------|-------------|
| `C1_rosetta_blind_detection.py` | Identify Rosetta-missed instabilities |
| `C4_mutation_effect_predictor.py` | Predict mutation DDG effects |

**Notebook:** `partners/jose_colbes/notebooks/colbes_scoring_function.ipynb`

---

### Alejandra Rojas - Serotype Forecaster
**Focus:** Arbovirus evolution tracking and RT-PCR primer design

| Script | Description |
|--------|-------------|
| `A2_pan_arbovirus_primers.py` | Design primers for DENV, ZIKV, CHIKV |
| `arbovirus_hyperbolic_trajectory.py` | Track serotype evolution |
| `primer_stability_scanner.py` | Identify stable genomic regions |

**Notebook:** `partners/alejandra_rojas/notebooks/rojas_serotype_forecast.ipynb`

---

### HIV Research Package
**Focus:** Clinical decision support for HIV treatment

| Script | Description |
|--------|-------------|
| `H6_tdr_screening.py` | Transmitted drug resistance screening |
| `H7_la_injectable_selection.py` | Long-acting injectable selection |

---

## Shared Module API

### Core Functions

```python
from shared import (
    # Sequence validation
    validate_sequence,

    # Property calculation
    compute_peptide_properties,
    compute_ml_features,
    compute_amino_acid_composition,

    # Prediction
    HemolysisPredictor,

    # Synthesis
    PrimerDesigner,

    # Logging
    get_logger,
    setup_logging,
)
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `HemolysisPredictor` | ML-based HC50 prediction and therapeutic index |
| `PrimerDesigner` | Codon-optimized PCR primer design |
| `UncertaintyQuantifier` | Bootstrap/ensemble confidence intervals |

---

## Running Tests

```bash
cd deliverables

# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_e2e_demos.py -v      # End-to-end demos
python -m pytest tests/test_hiv_package.py -v    # HIV package
python -m pytest tests/test_alejandra_rojas.py -v # Arbovirus package
python -m pytest tests/test_shared.py -v          # Shared modules

# Run with coverage
python -m pytest tests/ --cov=shared --cov=partners -v

# Run only fast tests (skip slow ones)
python -m pytest tests/ -v -m "not slow"
```

### Test Structure

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_e2e_demos.py` | 22 | End-to-end demo workflows |
| `test_hiv_package.py` | 32 | HIV TDR/LA/alignment tests |
| `test_alejandra_rojas.py` | 16 | Primer design tests |
| `test_jose_colbes.py` | 9 | Stability prediction tests |
| `test_shared.py` | 16 | Config and utility tests |

---

## Troubleshooting

### VAE Service Issues

**"Running in mock mode"**
- VAE service works without PyTorch but uses simplified algorithms
- To use real VAE: install PyTorch and run `git lfs pull` for checkpoints

**"Git LFS pointer" error**
```bash
git lfs install
git lfs pull
```

**Slow performance**
```python
from shared.config import get_config
config = get_config()
config.use_gpu = True  # Enable GPU if available
```

### Import Errors

**"Module not found"**
```bash
# Ensure you're in the deliverables directory
cd deliverables

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Test Failures

**Pre-existing failures (3 tests)**
- HIV: Minor assertion message format
- Jose Colbes: `classify_residue()` signature change

These don't affect core functionality.

---

## Command-Line Interface

```bash
# List available tools
python scripts/biotools.py --list

# Analyze a peptide
python scripts/biotools.py analyze GIGKFLHSAKKFGKAFVGEIMNS

# Run pathogen-specific design
python scripts/biotools.py pathogen --target S_aureus --output results/
```

---

## Dependencies

### Required
- Python 3.9+
- NumPy

### Recommended
- scikit-learn (ML predictions)
- pandas (data handling)
- matplotlib (visualization)

### Optional
- PyTorch (VAE integration)
- BioPython (sequence parsing)
- Primer3 (advanced primer design)

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

**Current Version:** 2.0.0 (December 2025)
- Unified VAE service module
- Complete HIV research package
- Enhanced arbovirus tools
- Comprehensive CLI and demos
- 101+ passing tests

---

## License

PolyForm Noncommercial License 1.0.0

See LICENSE file in the repository root for full license text.

---

## Citation

```bibtex
@software{ternary_vae_bioinformatics,
  author = {AI Whisperers},
  title = {Ternary VAE Bioinformatics: P-adic Geometry for Computational Biology},
  year = {2025},
  url = {https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics}
}
```

---

## Contact

- **Organization:** AI Whisperers
- **Repository:** https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics
