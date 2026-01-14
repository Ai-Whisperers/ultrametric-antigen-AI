# Data Directory

> **All data files for the Ternary VAE Bioinformatics project**

**Last Updated:** December 26, 2025

---

## Directory Structure

```
data/
├── external/           # External data sources
│   ├── github/         # Git submodules
│   └── huggingface/    # Hugging Face datasets
└── research/           # Research datasets (downloaded)
    ├── catnap_assay.txt         # CATNAP neutralization data
    ├── ctl_summary.csv          # CTL epitope summaries
    ├── stanford_hivdb_*.txt     # Stanford HIVDB drug resistance
    └── ...
```

---

## Data Sources

### External (`external/`)
- Git submodules for external repositories
- Hugging Face datasets (cached)

### Research (`research/`)
HIV research datasets downloaded from public sources:

| File | Source | Records | Description |
|------|--------|---------|-------------|
| `catnap_assay.txt` | LANL | 189,879 | Antibody neutralization assays |
| `ctl_summary.csv` | LANL | 2,115 | CTL epitope summaries |
| `stanford_hivdb_pi.txt` | Stanford | - | Protease inhibitor resistance |
| `stanford_hivdb_nrti.txt` | Stanford | - | NRTI resistance |
| `stanford_hivdb_nnrti.txt` | Stanford | - | NNRTI resistance |
| `stanford_hivdb_ini.txt` | Stanford | - | Integrase inhibitor resistance |

---

## Usage

```python
import pandas as pd

# Load CTL data
ctl_data = pd.read_csv("data/research/ctl_summary.csv")

# Load Stanford HIVDB
with open("data/research/stanford_hivdb_pi.txt") as f:
    pi_data = f.read()
```

---

## Data Access Layer

For programmatic access, use the `data_access/` module:

```python
from data_access import HIVDBClient, CATNAPClient

# Access Stanford HIVDB
hivdb = HIVDBClient()
resistance = hivdb.get_drug_resistance("PI")

# Access CATNAP
catnap = CATNAPClient()
neutralization = catnap.get_assays()
```

See `data_access/README.md` for full documentation.
