# Shared Bioinformatics Utilities

**Version**: 1.0.0
**Updated**: 2025-12-30
**Authors**: AI Whisperers

---

## Overview

The `shared` module provides centralized utilities for bioinformatics analysis across all deliverables in this project. It consolidates common functionality to eliminate code duplication and ensure consistent behavior.

## Installation

The shared module is automatically available when working within the deliverables directory:

```python
import sys
sys.path.insert(0, "/path/to/deliverables")

from shared import (
    compute_peptide_properties,
    HemolysisPredictor,
    PrimerDesigner,
    get_logger,
)
```

## Module Components

### 1. Peptide Utilities (`peptide_utils.py`)

Functions for analyzing peptide sequences:

| Function | Description |
|----------|-------------|
| `compute_peptide_properties(seq)` | Calculate charge, hydrophobicity, length, ratios |
| `compute_ml_features(seq)` | Generate 25-dimensional ML feature vector |
| `compute_amino_acid_composition(seq)` | Get AA frequency vector (20 values) |
| `compute_physicochemical_descriptors(seq)` | Extended properties (aromaticity, aliphatic index) |
| `validate_sequence(seq)` | Check if sequence contains valid amino acids |
| `decode_latent_to_sequence(z)` | Convert latent vector to peptide sequence |

### 2. Hemolysis Predictor (`hemolysis_predictor.py`)

Predict hemolytic toxicity of peptides:

```python
from shared import HemolysisPredictor

predictor = HemolysisPredictor()
result = predictor.predict("GIGKFLHSAKKFGKAFVGEIMNS")

print(f"HC50: {result['hc50_predicted']:.1f} uM")
print(f"Risk: {result['risk_category']}")
print(f"Probability: {result['hemolytic_probability']:.2f}")
```

### 3. Uncertainty Quantification (`uncertainty.py`)

Add confidence intervals to ML predictions:

```python
from shared import bootstrap_prediction_interval, UncertaintyPredictor

# Bootstrap method
mean, lower, upper = bootstrap_prediction_interval(
    model, X_train, y_train, X_test,
    n_bootstrap=100, confidence=0.90
)

# Or use the wrapper class
predictor = UncertaintyPredictor(model, scaler, method="bootstrap")
result = predictor.predict_with_uncertainty(X_test)
```

### 4. Primer Design (`primer_design.py`)

Design PCR primers for cloning peptides:

```python
from shared import PrimerDesigner

designer = PrimerDesigner()
primers = designer.design_for_peptide(
    "GIGKFLHSAKKFGKAFVGEIMNS",
    codon_optimization="ecoli"
)

print(f"Forward: {primers.forward}")
print(f"Reverse: {primers.reverse}")
```

### 5. Logging Framework (`logging_utils.py`)

Standardized logging across all tools:

```python
from shared import get_logger, setup_logging

setup_logging(level="INFO", log_file="analysis.log")
logger = get_logger("my_analysis")

logger.info("Starting analysis")
logger.prediction("MIC", 4.5, confidence=0.92)
logger.model_metrics("predictor", {"rmse": 0.35, "r": 0.85})
```

### 6. Configuration (`config.py`)

Centralized configuration management:

```python
from shared import get_config

config = get_config()
print(f"Project root: {config.project_root}")
print(f"VAE available: {config.has_vae}")
```

### 7. Constants (`constants.py`)

Reference data for calculations:

- `AMINO_ACIDS`: 20 standard amino acids
- `HYDROPHOBICITY`: Kyte-Doolittle scale
- `CHARGES`: Amino acid charges at pH 7.4
- `VOLUMES`: Molecular volumes
- `MOLECULAR_WEIGHTS`: Dalton weights
- `CODON_TABLE`: Standard genetic code
- `WHO_CRITICAL_PATHOGENS`: Priority pathogen list

## Curated Databases

### AMP Activity Database (224 entries)

Located in `carlos_brizuela/scripts/dramp_activity_loader.py`:

- **Total entries**: 224 peptide-pathogen-MIC records
- **Unique peptides**: 155 sequences
- **Coverage**:
  - *E. coli*: 105 entries
  - *S. aureus*: 72 entries
  - *P. aeruginosa*: 27 entries
  - *A. baumannii*: 20 entries

### Protein Stability Database (219 entries)

Located in `jose_colbes/scripts/protherm_ddg_loader.py`:

- **Total mutations**: 219 experimentally validated
- **Proteins covered**: 17 well-studied proteins
- **DDG range**: -1.2 to 5.5 kcal/mol

## Testing

Run the test suite:

```bash
cd deliverables
python -m pytest shared/tests/ -v
```

**Test coverage**: 67 tests covering all core functionality.

## API Quick Reference

### Peptide Analysis

```python
# Basic properties
props = compute_peptide_properties("KKLLDD")
# Returns: {net_charge, hydrophobicity, length, hydrophobic_ratio, cationic_ratio}

# ML features (25-dim vector)
features = compute_ml_features("KKLLDD")

# Validation
is_valid, error = validate_sequence("KKLLDD")
```

### Hemolysis Prediction

```python
predictor = HemolysisPredictor()

# Single prediction
result = predictor.predict(sequence)
# Returns: {hc50_predicted, is_hemolytic, hemolytic_probability, risk_category}

# Therapeutic index
ti = predictor.compute_therapeutic_index(sequence, mic_value=10.0)
# Returns: {therapeutic_index, interpretation, ...}
```

### Uncertainty Quantification

```python
# Bootstrap intervals
mean, lower, upper = bootstrap_prediction_interval(
    model, X_train, y_train, X_test,
    n_bootstrap=100, confidence=0.90
)

# Ensemble intervals (for GradientBoosting)
mean, lower, upper = ensemble_prediction_interval(model, X_test)

# Quantile regression
median, lower, upper = quantile_prediction_interval(
    X_train, y_train, X_test, alpha=0.10
)
```

### Primer Design

```python
designer = PrimerDesigner(
    min_length=18, max_length=25,
    min_gc=40, max_gc=60,
    min_tm=55, max_tm=65
)

# For DNA sequence
primers = designer.design_primers(dna_sequence)

# For peptide (with codon optimization)
primers = designer.design_for_peptide(peptide, codon_optimization="ecoli")

# Convert peptide to DNA
dna = designer.peptide_to_dna(peptide)
```

### Logging

```python
setup_logging(level="INFO", log_file="output.log")
logger = get_logger("module_name")

logger.debug("Debug message")
logger.info("Info message")
logger.analysis("Analysis result: r=0.85")
logger.prediction("MIC", value, confidence=0.9)
logger.model_metrics("model_name", {"rmse": 0.3})
logger.warning("Warning message")
logger.error("Error message")
```

## File Structure

```
shared/
├── __init__.py              # Main exports
├── config.py                # Configuration management
├── constants.py             # Reference data
├── peptide_utils.py         # Peptide analysis functions
├── vae_service.py           # VAE integration
├── uncertainty.py           # Uncertainty quantification
├── hemolysis_predictor.py   # Hemolysis prediction
├── primer_design.py         # PCR primer design
├── logging_utils.py         # Logging framework
├── README.md                # This file
└── tests/
    ├── __init__.py
    ├── test_peptide_utils.py
    └── test_constants.py
```

## Dependencies

**Required**:
- Python 3.9+
- NumPy

**Optional** (for full functionality):
- scikit-learn (ML predictions)
- PyTorch (VAE integration)
- Primer3 (advanced primer design)

## License

PolyForm Noncommercial License 1.0.0
