# HIV Bioinformatics Analysis

## p-adic Hyperbolic Codon Embedding for HIV Mutation Analysis

This module applies p-adic hyperbolic geometry to analyze HIV-1 mutations across 200,000+ records, providing insights into drug resistance, immune escape, antibody neutralization, and coreceptor tropism.

**Total Records Analyzed:** 202,085 across 10 integrated datasets

---

## 📖 Documentation

> **Start here:** [EXECUTIVE_SUMMARY.md](documentation/EXECUTIVE_SUMMARY.md) for a complete overview of findings.

| Document | Description |
|----------|-------------|
| [Executive Summary](documentation/EXECUTIVE_SUMMARY.md) | Overview of all findings for researchers |
| [Quick Start Guide](documentation/quickstart/QUICK_START.md) | Get running in 10 minutes |
| [Methodology](documentation/methodology/METHODOLOGY.md) | Complete technical methodology |
| [Novelty Assessment](documentation/NOVELTY_ASSESSMENT.md) | Novel vs. confirmed discoveries |
| [Literature Review](documentation/LITERATURE_REVIEW.md) | 150 papers for further reading |
| [API Reference](documentation/api_reference/API_REFERENCE.md) | Python module documentation |
| [FAQ](documentation/faq/FAQ.md) | Common questions & troubleshooting |

### Detailed Findings

| Analysis | Records | Key Document |
|----------|---------|--------------|
| Drug Resistance | 7,154 | [Drug Resistance Findings](documentation/findings/DRUG_RESISTANCE_FINDINGS.md) |
| CTL Escape | 2,115 | [CTL Escape Findings](documentation/findings/CTL_ESCAPE_FINDINGS.md) |
| Antibody Neutralization | 189,879 | [Antibody Findings](documentation/findings/ANTIBODY_NEUTRALIZATION_FINDINGS.md) |
| Coreceptor Tropism | 2,932 | [Tropism Findings](documentation/findings/TROPISM_FINDINGS.md) |
| Cross-Dataset Integration | All | [Integration Findings](documentation/findings/INTEGRATION_FINDINGS.md) |

---

## 🚀 Quick Start

```bash
# Navigate to scripts directory
cd scripts

# Run complete analysis (all 5 analyses, ~25 minutes)
python run_complete_analysis.py

# Or run individual analyses:
python analyze_stanford_resistance.py       # Drug resistance (~5 min)
python analyze_ctl_escape_expanded.py       # CTL epitopes (~3 min)
python analyze_catnap_neutralization.py     # Antibody neutralization (~10 min)
python analyze_tropism_switching.py         # Coreceptor tropism (~2 min)
python cross_dataset_integration.py         # Integration (~5 min)
```

---

## Directory Structure

```
hiv/
├── README.md                           # This file
├── scripts/
│   ├── run_complete_analysis.py        # Master pipeline script
│   ├── analyze_stanford_resistance.py  # Drug resistance analysis
│   ├── analyze_ctl_escape_expanded.py  # CTL epitope analysis
│   ├── analyze_catnap_neutralization.py # Antibody neutralization
│   ├── analyze_tropism_switching.py    # Coreceptor tropism
│   ├── cross_dataset_integration.py    # Multi-dataset integration
│   ├── unified_data_loader.py          # Data loading utilities
│   ├── position_mapper.py              # HXB2 coordinate mapping
│   └── codon_extraction.py             # Codon encoding functions
├── documentation/                       # 📖 COMPREHENSIVE DOCUMENTATION
│   ├── EXECUTIVE_SUMMARY.md            # Start here!
│   ├── NOVELTY_ASSESSMENT.md           # Novel vs. confirmed findings
│   ├── LITERATURE_REVIEW.md            # 150 papers to review
│   ├── methodology/                    # Technical methods
│   ├── findings/                       # Detailed results (5 files)
│   ├── statistical_analysis/           # Statistics guide
│   ├── figures_guide/                  # Visualization descriptions
│   ├── supplementary/                  # Data dictionary, glossary
│   ├── reproducibility/                # Reproduction instructions
│   ├── limitations/                    # Caveats and limitations
│   ├── future_work/                    # Research roadmap
│   ├── quickstart/                     # 10-minute setup
│   ├── api_reference/                  # Python API docs
│   ├── benchmarking/                   # Method comparisons
│   └── faq/                            # Common questions
├── results/                            # Analysis outputs
│   ├── stanford_resistance/            # Drug resistance results
│   ├── ctl_escape_expanded/            # CTL epitope results
│   ├── catnap_neutralization/          # Antibody results
│   ├── tropism/                        # Tropism results
│   └── integrated/                     # Cross-dataset results
├── legacy_scripts/                     # Original analysis scripts
│   ├── 01_hiv_escape_analysis.py
│   ├── 02_hiv_drug_resistance.py
│   └── glycan_shield/
└── data/
    └── codon_encoder_3adic.pt          # Trained codon encoder
```

---

## Core Concepts

### p-adic Hyperbolic Embedding

The genetic code's 64 codons are embedded into a **Poincaré ball** (hyperbolic space) where:

- **Center (radius ~0.1):** High 3-adic valuation → rare, functionally important
- **Boundary (radius ~0.9):** Low 3-adic valuation → common, flexible
- **Distance:** Measures evolutionary accessibility between codons

### Key Metrics

| Metric | Our Model | Interpretation |
|--------|-----------|----------------|
| Hierarchy Correlation | -0.832 | Strong radial structure |
| Cluster Accuracy | 79.7% | Good amino acid grouping |
| Synonymous Accuracy | 98.9% | Excellent codon grouping |

---

## Key Findings Summary

### Novel Discoveries

| Finding | Status | Details |
|---------|--------|---------|
| Position 22 as top tropism determinant | **NOVEL** | Separation score 0.591, exceeds classic 11/25 rule |
| Distance-resistance correlation | **NOVEL** | r = 0.41 for NRTIs, first geometric quantification |
| Breadth-centrality correlation for bnAbs | **NOVEL** | Broader antibodies target central epitopes |
| 328 safe vaccine targets | **NOVEL** | Multi-constraint optimization, resistance-free |

### Confirmed Findings

| Finding | Status | Validates |
|---------|--------|-----------|
| B57/B27 HLA protection | Confirmation | Fellay et al. 2007, Pereyra et al. 2010 |
| 11/25 tropism rule | Confirmation | Fouchier et al. 1992 |
| bnAb potency profiles | Confirmation | CATNAP literature |

---

## Analyses Available

### 1. Drug Resistance Analysis

**Script:** `analyze_stanford_resistance.py`
**Records:** 7,154 patient sequences, 90,269 mutations
**Output:** `results/stanford_resistance/`

| Drug Class | Mutations | Correlation (r) |
|------------|-----------|-----------------|
| NRTIs | 21,456 | 0.41 |
| NNRTIs | 28,103 | 0.38 |
| PIs | 23,847 | 0.34 |
| INIs | 16,863 | 0.29 |

### 2. CTL Escape Analysis

**Script:** `analyze_ctl_escape_expanded.py`
**Records:** 2,115 epitopes, 240 HLA types
**Output:** `results/ctl_escape_expanded/`

| HLA Supertype | Epitopes | Escape Velocity |
|---------------|----------|-----------------|
| A*02 | 193 | 0.342 |
| B*57 | 87 | 0.218 (protective) |
| B*27 | 52 | 0.256 (protective) |

### 3. Antibody Neutralization Analysis

**Script:** `analyze_catnap_neutralization.py`
**Records:** 189,879 virus-antibody pairs
**Output:** `results/catnap_neutralization/`

| bnAb | Breadth (%) | IC50 (μg/mL) |
|------|-------------|--------------|
| 3BNC117 | 78.8 | 0.242 |
| 10E8 | 76.7 | 0.221 |
| VRC01 | 68.9 | 0.580 |

### 4. Tropism Analysis

**Script:** `analyze_tropism_switching.py`
**Records:** 2,932 V3 sequences
**Output:** `results/tropism/`

- **Accuracy:** 85% (AUC = 0.86)
- **Top determinant:** Position 22 (34% importance)
- **Novel finding:** Position 22 outperforms classic positions 11 and 25

### 5. Cross-Dataset Integration

**Script:** `cross_dataset_integration.py`
**Output:** `results/integrated/`

- **16,054** resistance-epitope overlaps identified
- **328** vaccine targets with no resistance overlap
- **Trade-off scoring** for dual-pressure positions

---

## Usage Examples

### Run Complete Analysis Pipeline

```bash
cd scripts
python run_complete_analysis.py
# Takes ~25 minutes, produces all results and reports
```

### Analyze Mutation Geometry

```python
from codon_extraction import encode_mutation_pair

# Analyze M184V (major NRTI resistance mutation)
features = encode_mutation_pair('M', 'V')
print(f"M→V hyperbolic distance: {features['hyperbolic_distance']:.3f}")
print(f"Radial change: {features['radial_change']:.3f}")
```

### Load and Explore Results

```python
import pandas as pd
from pathlib import Path

# Load vaccine targets
targets = pd.read_csv(Path("results/integrated/vaccine_targets.csv"))
print(f"Total vaccine targets: {len(targets)}")

# Filter resistance-free targets
safe_targets = targets[targets['resistance_overlap'] == 'No']
print(f"Resistance-free targets: {len(safe_targets)}")

# Top 10 by score
print(safe_targets.nlargest(10, 'score')[['epitope', 'protein', 'hla_count', 'score']])
```

### Find Epitope Overlaps

```python
from position_mapper import find_overlapping_epitopes
from unified_data_loader import load_lanl_ctl

epitopes = load_lanl_ctl()

# Find epitopes overlapping RT position 103 (K103N resistance)
overlaps = find_overlapping_epitopes(103, 'RT', epitopes)
print(f"K103N affects {len(overlaps)} CTL epitopes")
for _, ep in overlaps.iterrows():
    print(f"  {ep['Epitope']} (HLA: {ep['HLA']})")
```

---

## Biological Interpretation

### Radial Position in Hyperbolic Space

| Radial Position | Interpretation |
|-----------------|----------------|
| Low (0.3-0.5) | Highly constrained, essential function |
| Medium (0.5-0.7) | Moderately constrained |
| High (0.7-0.9) | Variable, tolerated changes |

### Trade-off Scores

| Score Range | Interpretation |
|-------------|----------------|
| < 2.0 | Low trade-off |
| 2.0 - 4.0 | Moderate trade-off |
| > 4.0 | High trade-off (clinically important) |

### Escape Velocity

- **Low velocity:** Escape is difficult (constrained epitope) - good vaccine target
- **High velocity:** Escape is easy (variable epitope) - poor vaccine target

---

## Data Sources

| Dataset | Source | Records |
|---------|--------|---------|
| Stanford HIVDB | https://hivdb.stanford.edu | 7,154 |
| LANL CTL | https://www.hiv.lanl.gov | 2,115 |
| CATNAP | https://www.hiv.lanl.gov/content/sequence/CATNAP | 189,879 |
| V3 Coreceptor | https://huggingface.co/datasets/tnhaider/HIV_V3_coreceptor | 2,932 |

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=12.0.0
```

---

## Further Reading

- **For methodology:** See [METHODOLOGY.md](documentation/methodology/METHODOLOGY.md)
- **For novel findings:** See [NOVELTY_ASSESSMENT.md](documentation/NOVELTY_ASSESSMENT.md)
- **For related papers:** See [LITERATURE_REVIEW.md](documentation/LITERATURE_REVIEW.md) (150 papers)
- **For limitations:** See [LIMITATIONS_AND_CAVEATS.md](documentation/limitations/LIMITATIONS_AND_CAVEATS.md)
- **For future work:** See [FUTURE_DIRECTIONS.md](documentation/future_work/FUTURE_DIRECTIONS.md)

---

## Citation

```bibtex
@software{hiv_padic_analysis,
  author = {{Ternary VAE Bioinformatics Research Group}},
  title = {HIV Evolution Analysis Using P-adic Hyperbolic Geometry},
  year = {2025},
  version = {1.0},
  note = {202,085 records analyzed across 10 datasets}
}
```

---

## License

PolyForm Noncommercial License 1.0.0

Copyright 2024-2025 AI Whisperers

---

_Last updated: 2025-12-25_
