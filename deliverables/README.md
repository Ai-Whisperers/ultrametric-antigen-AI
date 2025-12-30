# Bioinformatics Deliverables

**P-adic Geometry for Computational Biology**

Version 1.0 | December 2025 | AI Whisperers

---

## Overview

This package provides bioinformatics tools built on p-adic (hyperbolic) geometric methods for:

- **Antimicrobial Peptide Design** - Multi-objective optimization with safety assessment
- **Protein Stability Prediction** - Rotamer scoring and mutation effects
- **Arbovirus Surveillance** - Serotype tracking and primer design
- **HIV Clinical Tools** - Drug resistance and treatment selection

---

## Quick Start

```python
# Install dependencies
pip install numpy scikit-learn

# Import shared utilities
from shared import (
    compute_peptide_properties,
    HemolysisPredictor,
    PrimerDesigner,
)

# Analyze a peptide
peptide = "GIGKFLHSAKKFGKAFVGEIMNS"
props = compute_peptide_properties(peptide)
print(f"Charge: {props['net_charge']}, Hydrophobicity: {props['hydrophobicity']:.2f}")

# Predict toxicity
predictor = HemolysisPredictor()
result = predictor.predict(peptide)
print(f"HC50: {result['hc50_predicted']:.1f} uM, Risk: {result['risk_category']}")

# Design primers
designer = PrimerDesigner()
primers = designer.design_for_peptide(peptide[:15], codon_optimization='ecoli')
print(f"Forward: {primers.forward}")
```

---

## Directory Structure

```
deliverables/
│
├── README.md                    # This file
├── docs/                        # Core documentation
│   ├── INDEX.md                # Documentation map
│   ├── HEMOLYSIS_PREDICTOR.md  # Toxicity prediction guide
│   ├── PRIMER_DESIGN.md        # PCR primer design guide
│   ├── UNCERTAINTY_QUANTIFICATION.md
│   ├── CURATED_DATABASES.md    # Data sources
│   └── USAGE_EXAMPLES.md       # Complete workflows
│
├── shared/                      # Common Python modules
│   ├── __init__.py             # Public API exports
│   ├── peptide_utils.py        # Sequence analysis
│   ├── hemolysis_predictor.py  # Toxicity prediction
│   ├── primer_design.py        # PCR primer design
│   ├── uncertainty.py          # Confidence intervals
│   └── tests/                  # Test suite
│
├── tutorials/                   # Interactive notebooks
│   ├── 00_bioinformatics_toolkit_overview.ipynb
│   ├── 01_getting_started.ipynb
│   └── 02_activity_prediction.ipynb
│
├── scripts/                     # Command-line tools
│   └── biotools.py             # CLI interface
│
├── results/                     # Output results
│   ├── pathogen_specific/
│   ├── microbiome_safe/
│   ├── pan_arbovirus_primers/
│   └── ...
│
└── partners/                    # Research partner modules
    ├── carlos_brizuela/        # AMP optimization
    ├── jose_colbes/            # Protein stability
    ├── alejandra_rojas/        # Arbovirus surveillance
    └── hiv_research_package/   # HIV clinical tools
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
python -m pytest shared/tests/ -v

# Or run specific tests
python -m pytest shared/tests/test_peptide_utils.py -v
```

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
