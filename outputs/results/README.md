# Results Directory

> **All outputs from training, analysis, and research pipelines**

**Last Updated:** December 29, 2025

---

## Overview

This directory contains all result outputs from the Ternary VAE bioinformatics project, organized by category:

- **Research**: Scientific discoveries from p-adic analysis
- **Clinical**: Clinical decision support outputs
- **Partners**: Collaboration validation results
- **Validation**: AlphaFold3 structural predictions
- **Training**: Model training logs and metrics
- **Literature**: Literature implementation results

---

## Directory Structure

```
results/
├── README.md                    # This file
│
├── research/                    # Scientific discoveries
│   ├── README.md               # Research documentation
│   ├── discoveries/hiv/        # HIV-specific findings
│   ├── research_discoveries/   # Core discoveries
│   ├── advanced_research/      # Extended analysis
│   └── comprehensive_analysis/ # Full reports
│
├── clinical/                    # Clinical applications
│   ├── README.md               # Clinical documentation
│   ├── clinical_applications/  # Decision support
│   ├── clinical_dashboard/     # Dashboard data
│   └── clinical_integration/   # Integration outputs
│
├── partners/                    # Collaboration results
│   ├── README.md               # Partner documentation
│   ├── brizuela/               # AMP optimization
│   ├── colbes/                 # Rotamer stability
│   └── rojas/                  # Dengue forecasting
│
├── validation/                  # Structural validation
│   ├── README.md               # Validation documentation
│   ├── alphafold_predictions/  # AF3 outputs (LFS)
│   └── benchmarks/             # Method comparisons
│
├── training/                    # Training outputs
│   ├── logs/                   # Training logs
│   ├── metrics/                # Numeric metrics
│   └── run_history/            # TensorBoard data
│
├── literature/                  # Literature implementations
│   ├── literature_implementations/
│   ├── advanced_literature_implementations/
│   └── cutting_edge_implementations/
│
└── ablation/                    # Ablation study results
    └── ablation_results_*.json
```

---

## Quick Access

### Key Reports

| Category | Document | Path |
|----------|----------|------|
| **HIV Research** | Drug resistance findings | `research/discoveries/hiv/DISCOVERY_HIV_PADIC_RESISTANCE.md` |
| **Clinical** | Vaccine candidates | `clinical/clinical_applications/CLINICAL_REPORT.md` |
| **Partners** | Validation summary | `partners/VALIDATION_SUMMARY.md` |
| **Validation** | AlphaFold3 results | `validation/alphafold_predictions/VALIDATION_RESULTS.md` |

### Key Metrics

| Metric | Value | Source |
|--------|-------|--------|
| Hierarchy ceiling | -0.8321 | Mathematical limit |
| AF3 correlation | r = -0.89 | Sentinel glycan validation |
| NRTI resistance distance | d = 6.05 | Drug resistance |
| Rotamer accuracy | 87% | Partner validation |
| Dengue forecast accuracy | 82% | Partner validation |

---

## Category Details

### Research (`research/`)

Scientific discoveries from p-adic bioinformatics:

| Finding | Value | File |
|---------|-------|------|
| Drug class signatures | NRTI d=6.05, PI d=3.60 | `discoveries/hiv/` |
| Elite controller HLAs | B27 d=7.38 | `discoveries/hiv/` |
| Sentinel glycans | 7 sites identified | `research_discoveries/` |
| Evolutionary correlation | r=0.8339 | `comprehensive_analysis/` |

**README:** `research/README.md`

### Clinical (`clinical/`)

Clinical decision support outputs:

| Output | Description | File |
|--------|-------------|------|
| Vaccine candidates | Top epitope targets | `clinical_applications/` |
| MDR screening | 2,489 high-risk sequences | `clinical_applications/` |
| Tat targets | 247 druggable proteins | `clinical_applications/` |

**README:** `clinical/README.md`

### Partners (`partners/`)

Collaboration validation results:

| Partner | Domain | Accuracy | Files |
|---------|--------|----------|-------|
| Brizuela | AMPs | Pending | `brizuela/` |
| Colbes | Rotamers | 87% | `colbes/` |
| Rojas | Dengue | 82% | `rojas/` |

**README:** `partners/README.md`

### Validation (`validation/`)

AlphaFold3 structural predictions:

| Prediction | ipTM | Goldilocks | Files |
|------------|------|------------|-------|
| Wild-type | 0.89 | - | `alphafold_predictions/` |
| ΔN429 | 0.34 | 28% | `alphafold_predictions/` |
| ΔN58 | 0.67 | 18% | `alphafold_predictions/` |

**README:** `validation/README.md`

### Training (`training/`)

Model training artifacts:

| Content | Format | Location |
|---------|--------|----------|
| Logs | TensorBoard | `logs/` |
| Metrics | JSON, NPY | `metrics/` |
| History | TensorBoard | `run_history/` |

### Ablation (`ablation/`)

Ablation study results (31 experiments):

```python
import json
with open("results/ablation/ablation_results_20251227_085130.json") as f:
    results = json.load(f)
```

---

## Usage Examples

### Load Clinical Data

```python
import json
from pathlib import Path

RESULTS = Path("outputs/results")

# Load clinical decision support
with open(RESULTS / "clinical/clinical_applications/clinical_decision_support.json") as f:
    clinical = json.load(f)

# Get vaccine candidates
for candidate in clinical["vaccine_candidates"]:
    print(f"{candidate['peptide']}: {candidate['priority']}")
```

### Load Research Findings

```python
# Load partner validation results
with open(RESULTS / "partners/colbes/rotamer_stability.json") as f:
    rotamers = json.load(f)

accuracy = sum(1 for r in rotamers if r["correct"]) / len(rotamers)
print(f"Rotamer accuracy: {accuracy:.1%}")
```

### Access TensorBoard Logs

```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator(
    str(RESULTS / "training/run_history/archive/ternary_vae_20251212_225619")
)
ea.Reload()

# Get metric
scalars = ea.Scalars("Hyperbolic_Correlation_Hyp_VAE_B")
```

---

## File Types

| Extension | Description | Tracking |
|-----------|-------------|----------|
| `.md` | Reports and documentation | Git |
| `.json` | Structured data | Git (small) / LFS (large) |
| `.npy` | Numeric arrays | Git LFS |
| TensorBoard | Training logs | Git |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| `../README.md` | Main outputs documentation |
| `../DOCUMENTATION.md` | Comprehensive guide |
| `../models/README.md` | Checkpoint documentation |
| `../../.claude/CLAUDE.md` | Project context |

---

## Maintenance

### Adding New Results

1. Choose appropriate category directory
2. Use consistent JSON/Markdown format
3. Update category README
4. Update this file if adding new category

### Large Files

Add to `.gitattributes` for LFS tracking:
```gitattributes
results/validation/alphafold_predictions/**/*.json filter=lfs
results/**/*.npy filter=lfs
```
