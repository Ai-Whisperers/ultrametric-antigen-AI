# Bioinformatics Tools Documentation Index

**Version**: 1.0.0
**Updated**: 2025-12-30
**Authors**: AI Whisperers

---

## Overview

This documentation covers the shared bioinformatics utilities for antimicrobial peptide (AMP) research, protein stability prediction, and molecular biology workflows.

---

## Documentation Map

```
docs/
├── INDEX.md                      # This file
├── CURATED_DATABASES.md          # Experimental data sources
├── UNCERTAINTY_QUANTIFICATION.md # Prediction confidence intervals
├── HEMOLYSIS_PREDICTOR.md        # Toxicity prediction
├── PRIMER_DESIGN.md              # PCR primer design
└── USAGE_EXAMPLES.md             # Complete workflows
```

---

## Quick Navigation

### Getting Started
- [Toolkit Overview](../tutorials/00_bioinformatics_toolkit_overview.ipynb) - Comprehensive introduction to all tools
- [Getting Started Tutorial](../tutorials/01_getting_started.ipynb) - Interactive basics
- [Activity Prediction](../tutorials/02_activity_prediction.ipynb) - ML model training
- [Shared Module README](../shared/README.md) - API overview
- [Usage Examples](./USAGE_EXAMPLES.md) - Complete code examples

### Core Modules
| Module | Documentation | Description |
|--------|---------------|-------------|
| Peptide Utilities | [README](../shared/README.md#1-peptide-utilities-peptide_utilspy) | Sequence analysis |
| Hemolysis Predictor | [HEMOLYSIS_PREDICTOR.md](./HEMOLYSIS_PREDICTOR.md) | Toxicity prediction |
| Uncertainty Quantification | [UNCERTAINTY_QUANTIFICATION.md](./UNCERTAINTY_QUANTIFICATION.md) | Confidence intervals |
| Primer Design | [PRIMER_DESIGN.md](./PRIMER_DESIGN.md) | PCR primer design |

### Data Sources
| Database | Documentation | Entries |
|----------|---------------|---------|
| AMP Activity | [CURATED_DATABASES.md#amp](./CURATED_DATABASES.md#1-antimicrobial-peptide-activity-database) | 224 |
| Protein Stability (DDG) | [CURATED_DATABASES.md#ddg](./CURATED_DATABASES.md#2-protein-stability-ddg-database) | 219 |
| Hemolysis Training | [CURATED_DATABASES.md#hemolysis](./CURATED_DATABASES.md#3-hemolysis-training-data) | 40 |

---

## Key Features

### 1. Peptide Analysis
Compute biophysical properties of antimicrobial peptides:
- Net charge, hydrophobicity, amphipathicity
- Amino acid composition
- ML-ready feature vectors (25 dimensions)

```python
from shared import compute_peptide_properties
props = compute_peptide_properties("GIGKFLHSAKKFGKAFVGEIMNS")
```

### 2. Toxicity Prediction
Predict hemolytic activity (HC50) and therapeutic index:
- Machine learning model trained on 40 validated peptides
- Risk categorization (High/Moderate/Low)
- Therapeutic index calculation

```python
from shared import HemolysisPredictor
predictor = HemolysisPredictor()
result = predictor.predict("GIGKFLHSAKKFGKAFVGEIMNS")
```

### 3. Uncertainty Quantification
Add confidence intervals to predictions:
- Bootstrap intervals (any model)
- Ensemble intervals (tree-based models)
- Quantile regression (heteroscedastic data)

```python
from shared import bootstrap_prediction_interval
mean, lower, upper = bootstrap_prediction_interval(
    model, X_train, y_train, X_test, confidence=0.90
)
```

### 4. Primer Design
Design PCR primers for peptide cloning:
- Codon optimization (E. coli, human)
- Primer3 integration (optional)
- Tm and GC content calculation

```python
from shared import PrimerDesigner
designer = PrimerDesigner()
primers = designer.design_for_peptide("GIGKFLHSAKKFGKAFVGEIMNS")
```

---

## Module Dependencies

```
shared/
├── __init__.py          # Main exports
├── config.py            # Configuration management
├── constants.py         # Reference data (AA properties, codons)
├── peptide_utils.py     # Sequence analysis [numpy]
├── vae_service.py       # VAE integration [torch, optional]
├── uncertainty.py       # Confidence intervals [sklearn]
├── hemolysis_predictor.py # Toxicity prediction [sklearn]
├── primer_design.py     # PCR primers [primer3, optional]
├── logging_utils.py     # Logging framework [standard lib]
└── tests/               # Test suite [pytest]
```

### Required Dependencies
- Python 3.9+
- NumPy

### Optional Dependencies
| Feature | Dependency | Notes |
|---------|------------|-------|
| ML predictions | scikit-learn | Hemolysis, uncertainty |
| VAE integration | PyTorch | Latent space encoding |
| Advanced primers | Primer3 | External tool |

---

## Data Loaders

### AMP Activity Data
```python
from partners.antimicrobial_peptides.scripts.dramp_activity_loader import DRAMPLoader

loader = DRAMPLoader()
db = loader.generate_curated_database()
X, y = db.get_training_data(target="Escherichia coli")
```

### Protein Stability Data
```python
from partners.protein_stability_ddg.scripts.protherm_ddg_loader import ProThermLoader

loader = ProThermLoader()
db = loader.generate_curated_database()
X, y, feature_names = db.get_training_data()
```

---

## Typical Workflows

### 1. Screen Peptide Candidates
```
Candidate Sequences
       ↓
Validate Sequences ──→ Invalid? → Reject
       ↓
Compute Properties
       ↓
Predict Hemolysis
       ↓
Predict Activity (if model available)
       ↓
Calculate Therapeutic Index
       ↓
Rank by Development Score
       ↓
Design Primers for Top Candidates
```

### 2. Train Activity Predictor
```
Load Curated Database
       ↓
Split Train/Test
       ↓
Scale Features
       ↓
Train Model (Gradient Boosting)
       ↓
Cross-Validate
       ↓
Compute Predictions with Uncertainty
       ↓
Evaluate Metrics (RMSE, coverage)
```

### 3. Design Cloning Strategy
```
Peptide Sequence
       ↓
Validate Sequence
       ↓
Convert to DNA (codon optimized)
       ↓
Design Forward/Reverse Primers
       ↓
Check Tm, GC, self-complementarity
       ↓
Add Restriction Sites
       ↓
Order Primers
```

---

## Testing

Run the test suite:
```bash
cd deliverables
python -m pytest shared/tests/ -v
```

Current coverage: 67 tests

---

## References

### AMP Databases
1. APD3: Antimicrobial Peptide Database - https://aps.unmc.edu/
2. DRAMP: Data Repository of AMPs - http://dramp.cpu-bioinfor.org/
3. DBAASP: Database of Antimicrobial Activity - https://dbaasp.org/

### Stability Databases
4. ProTherm: Thermodynamic Database - https://web.iitm.ac.in/bioinfo2/prothermdb/

### Tools
5. Primer3: Primer Design - https://primer3.org/

---

## Partner Research Tools

| Partner | Package | Key Scripts | Notebook |
|---------|---------|-------------|----------|
| Alejandra Rojas | `partners/arbovirus_surveillance/` | A2_pan_arbovirus_primers.py | [Serotype Forecaster](../partners/arbovirus_surveillance/notebooks/rojas_serotype_forecast.ipynb) |
| Carlos Brizuela | `partners/antimicrobial_peptides/` | B1, B8, B10 AMP design tools | [AMP Navigator](../partners/antimicrobial_peptides/notebooks/brizuela_amp_navigator.ipynb) |
| Jose Colbes | `partners/protein_stability_ddg/` | C1, C4 protein stability tools | [Rotamer Scoring](../partners/protein_stability_ddg/notebooks/colbes_scoring_function.ipynb) |
| HIV Research | `partners/hiv_research_package/` | H6, H7 clinical decision tools | - |

### Partner Notebook Quick Links

| Notebook | Focus Area | Key Features |
|----------|------------|--------------|
| **AMP Navigator** | Antimicrobial peptide design | NSGA-II optimization, Pareto fronts, synthesis feasibility |
| **Rotamer Scoring** | Protein stability prediction | Rosetta-blind detection, mutation effects, DDG prediction |
| **Serotype Forecaster** | Arbovirus surveillance | Hyperbolic trajectories, risk assessment, RT-PCR primers |

See individual `README.md` files in each package for usage details.

---

## Project Structure (Cleaned)

```
deliverables/
├── docs/                    # Core documentation (this folder)
├── shared/                  # Common utilities and modules
├── tutorials/               # Jupyter notebooks
├── scripts/                 # Utility scripts
├── results/                 # Shared output directories
└── partners/                # Research partner modules
    ├── arbovirus_surveillance/     # Arbovirus primer research
    │   ├── README.md
    │   ├── scripts/
    │   ├── notebooks/
    │   ├── data/
    │   └── docs/
    ├── antimicrobial_peptides/     # AMP optimization research
    │   ├── README.md
    │   ├── scripts/
    │   ├── notebooks/
    │   ├── data/
    │   └── docs/
    ├── protein_stability_ddg/         # Protein stability research
    │   ├── README.md
    │   ├── scripts/
    │   ├── notebooks/
    │   ├── data/
    │   └── docs/
    └── hiv_research_package/ # HIV clinical tools
        ├── README.md
        ├── scripts/
        └── docs/
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.1 | 2025-12-30 | Cleaned up redundant docs, consolidated structure |
| 1.0.0 | 2025-12-30 | Initial documentation release |

---

## License

PolyForm Noncommercial License 1.0.0
