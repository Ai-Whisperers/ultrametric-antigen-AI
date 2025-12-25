# HIV Bioinformatics Analysis

## p-adic Hyperbolic Codon Embedding for HIV Mutation Analysis

This module applies p-adic hyperbolic geometry to analyze HIV-1 mutations, providing insights into immune escape and drug resistance mechanisms.

---

## Quick Start

```bash
# 1. One-time setup (generates codon encoder)
python scripts/setup/setup_hiv_analysis.py

# 2. Validate setup
python scripts/validate_hiv_setup.py

# 3. Run analyses
python scripts/run_hiv_analysis.py              # All core analyses
python scripts/run_hiv_analysis.py --escape     # CTL escape only
python scripts/run_hiv_analysis.py --drug-resistance  # Drug resistance only
```

---

## Directory Structure

```
hiv/
├── README.md                    # This file
├── ANALYSIS_REPORT.md           # Detailed findings report
├── scripts/
│   ├── 01_hiv_escape_analysis.py      # CTL escape mutation analysis
│   ├── 02_hiv_drug_resistance.py      # Drug resistance analysis
│   ├── 03_hiv_handshake_analysis.py   # Handshake interface analysis
│   ├── 04_hiv_hiding_landscape.py     # Hiding landscape analysis
│   ├── 05_visualize_hiding_landscape.py
│   ├── 06_validate_integrase_vulnerability.py
│   ├── 07_validate_all_conjectures.py
│   ├── 08_hybrid_integrase_validation.py
│   ├── 09_cluster_approaches_by_codon.py
│   ├── 10_visualize_approach_clusters.py
│   └── hyperbolic_utils.py            # Shared utilities
├── glycan_shield/
│   ├── 01_glycan_sentinel_analysis.py
│   ├── 02_alphafold3_input_generator.py
│   └── 03_create_batch_json.py
├── data/
│   ├── codon_encoder_3adic.pt         # Trained codon encoder
│   └── metrics/                       # Analysis metrics
└── results/
    ├── hiv_escape_results.json        # CTL escape results
    └── hiv_resistance_results.json    # Drug resistance results
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

## Analyses Available

### 1. CTL Escape Analysis (`01_hiv_escape_analysis.py`)

Analyzes how HIV-1 escapes cytotoxic T lymphocyte (CTL) recognition through mutations in epitopes.

**Key Findings:**
- 77.8% of escape mutations cross cluster boundaries
- High-efficacy/low-cost escapes cluster at distances 5.8-6.9
- Boundary crossing correlates with escape success

### 2. Drug Resistance Analysis (`02_hiv_drug_resistance.py`)

Maps drug resistance mutations across four antiretroviral drug classes.

**Key Findings by Drug Class:**

| Class | Mean Distance | Interpretation |
|-------|---------------|----------------|
| NRTI | 6.08 | Requires substantial codon changes |
| NNRTI | 5.04 | K103N has lowest distance (3.80) |
| PI | 4.63 | High variance; M46I very low (0.65) |
| INSTI | 4.92 | DTG has high genetic barrier |

### 3. Glycan Shield Analysis (`glycan_shield/`)

Analyzes glycosylation sites that shield HIV envelope from antibody recognition.

### 4. Integrase Vulnerability (`06_validate_integrase_vulnerability.py`)

Identifies structural vulnerabilities in HIV integrase for therapeutic targeting.

---

## Usage Examples

### Run Full Analysis Pipeline

```bash
python scripts/run_hiv_analysis.py --all
```

### Analyze Specific Mutation

```python
from hyperbolic_utils import load_codon_encoder, codon_to_onehot, poincare_distance
import torch

# Load encoder
encoder, mapping, _ = load_codon_encoder(device="cpu", version="3adic")

# Encode codons
def get_embedding(codon):
    x = torch.tensor([codon_to_onehot(codon)]).float()
    with torch.no_grad():
        return encoder.encode(x)[0]

# Compare wild-type vs mutant
wt_codon = "ATG"  # Methionine
mut_codon = "ATC"  # Isoleucine

wt_emb = get_embedding(wt_codon)
mut_emb = get_embedding(mut_codon)

distance = poincare_distance(wt_emb.unsqueeze(0), mut_emb.unsqueeze(0))
print(f"M→I distance: {distance.item():.4f}")
```

### Batch Analysis

```python
import json
from pathlib import Path

# Load results
results_dir = Path("results")
with open(results_dir / "hiv_escape_results.json") as f:
    escape_data = json.load(f)

# Analyze patterns
for epitope, data in escape_data["epitopes"].items():
    print(f"\n{epitope}:")
    for mut in data["escape_mutations"]:
        status = "CROSSED" if mut["boundary_crossed"] else "within"
        print(f"  {mut['mutation']}: d={mut['hyperbolic_distance']:.3f} [{status}]")
```

---

## Biological Interpretation

### Distance Ranges

| Distance | Interpretation | Examples |
|----------|----------------|----------|
| < 1 | Minimal change | M46I (0.65) - accessory mutation |
| 1-3 | Minor shift | Q148H (2.37), K103N (3.80) |
| 3-5 | Moderate change | Y181C (4.45), E92Q (4.32) |
| 5-7 | Significant shift | Most resistance mutations |
| > 7 | Major reorganization | T215Y (7.17), D314N (7.17) |

### Boundary Crossing

- **Crossed:** Mutation moves to different amino acid cluster
- **Within:** Mutation stays in same cluster (synonymous-like)

Boundary crossing correlates with:
- Higher immune escape efficacy
- Greater drug resistance
- Larger functional impact

---

## Dependencies

```
torch>=2.0.0
numpy>=2.0.0
scipy>=1.10.0
```

---

## References

1. **p-adic Numbers in Genetics:** The 3-adic valuation provides a natural metric for codon relationships.

2. **Hyperbolic Geometry:** Poincaré ball models hierarchical structures better than Euclidean space.

3. **HIV Drug Resistance:** Stanford HIV Drug Resistance Database (https://hivdb.stanford.edu/)

4. **CTL Epitopes:** Los Alamos HIV Immunology Database (https://www.hiv.lanl.gov/)

---

## License

PolyForm Noncommercial License 1.0.0

Copyright 2024-2025 AI Whisperers
