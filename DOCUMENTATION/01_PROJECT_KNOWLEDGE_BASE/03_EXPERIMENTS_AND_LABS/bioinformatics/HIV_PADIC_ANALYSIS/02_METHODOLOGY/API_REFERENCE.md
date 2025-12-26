# API Reference

## Python Module Documentation

**Version:** 1.0
**Last Updated:** December 25, 2025

---

## Module Overview

```
scripts/
├── unified_data_loader.py   # Data loading utilities
├── position_mapper.py       # HXB2 coordinate mapping
├── codon_extraction.py      # Codon encoding functions
├── analyze_*.py             # Analysis scripts
└── run_complete_analysis.py # Pipeline orchestration
```

---

## 1. unified_data_loader.py

### Functions

#### `load_stanford_hivdb(drug_class=None)`

Load Stanford HIV Drug Resistance Database records.

**Parameters:**
- `drug_class` (str, optional): Filter by class ('PI', 'NRTI', 'NNRTI', 'INI'). If None, loads all.

**Returns:**
- `pd.DataFrame`: Resistance data with columns [SeqID, CompMutList, drug columns, position columns]

**Example:**
```python
from unified_data_loader import load_stanford_hivdb

# Load all
df = load_stanford_hivdb()
print(f"Total records: {len(df)}")  # 7,154

# Load only PI data
pi_df = load_stanford_hivdb('PI')
print(f"PI records: {len(pi_df)}")  # 2,171
```

---

#### `load_lanl_ctl()`

Load LANL CTL epitope database.

**Returns:**
- `pd.DataFrame`: Epitope data with columns [Epitope, Protein, HXB2_start, HXB2_end, HLA, Subtype]

**Example:**
```python
from unified_data_loader import load_lanl_ctl

df = load_lanl_ctl()
print(f"Epitopes: {len(df)}")  # 2,115
print(f"Unique HLAs: {df['HLA'].nunique()}")  # 240
```

---

#### `load_catnap()`

Load CATNAP neutralization assay data.

**Returns:**
- `pd.DataFrame`: Neutralization data with columns [Antibody, Virus, IC50, IC80, ID50]

**Example:**
```python
from unified_data_loader import load_catnap

df = load_catnap()
print(f"Records: {len(df)}")  # 189,879
print(f"Antibodies: {df['Antibody'].nunique()}")  # 1,123
```

---

#### `load_v3_coreceptor()`

Load V3 loop coreceptor tropism data.

**Returns:**
- `pd.DataFrame`: V3 sequences with columns [Sequence, CCR5, CXCR4]

**Example:**
```python
from unified_data_loader import load_v3_coreceptor

df = load_v3_coreceptor()
print(f"Sequences: {len(df)}")  # 2,932
print(f"X4: {df['CXCR4'].sum()}")  # 702
```

---

#### `get_dataset_summary()`

Get summary statistics for all datasets.

**Returns:**
- `dict`: Summary with record counts and key statistics

**Example:**
```python
from unified_data_loader import get_dataset_summary

summary = get_dataset_summary()
for dataset, info in summary.items():
    print(f"{dataset}: {info['records']} records")
```

---

## 2. position_mapper.py

### Constants

#### `HXB2_REGIONS`

Dictionary of gene/protein coordinates in HXB2 reference.

```python
HXB2_REGIONS = {
    "gag": (790, 2292),
    "pol": (2085, 5096),
    "PR": (2253, 2549),
    "RT": (2550, 3869),
    "IN": (4230, 5096),
    "env": (6225, 8795),
    "gp120": (6225, 7758),
    "V3": (7110, 7217),
    # ... more regions
}
```

---

### Functions

#### `protein_pos_to_hxb2(position, protein)`

Convert protein-relative position to HXB2 coordinate.

**Parameters:**
- `position` (int): Position within protein (1-indexed)
- `protein` (str): Protein name ('PR', 'RT', 'IN', 'gp120', etc.)

**Returns:**
- `int`: HXB2 nucleotide position

**Example:**
```python
from position_mapper import protein_pos_to_hxb2

# PR position 82 (resistance position)
hxb2_pos = protein_pos_to_hxb2(82, 'PR')
print(f"HXB2: {hxb2_pos}")  # 2496
```

---

#### `hxb2_to_protein_pos(hxb2_position, protein)`

Convert HXB2 coordinate to protein-relative position.

**Parameters:**
- `hxb2_position` (int): HXB2 nucleotide position
- `protein` (str): Target protein name

**Returns:**
- `int`: Position within protein (1-indexed), or None if outside protein

**Example:**
```python
from position_mapper import hxb2_to_protein_pos

pos = hxb2_to_protein_pos(2496, 'PR')
print(f"PR position: {pos}")  # 82
```

---

#### `parse_mutation(mutation_str)`

Parse mutation string into components.

**Parameters:**
- `mutation_str` (str): Mutation notation (e.g., "M184V", "K103N")

**Returns:**
- `dict`: {wt_aa, position, mut_aa}

**Example:**
```python
from position_mapper import parse_mutation

mut = parse_mutation("M184V")
print(mut)  # {'wt_aa': 'M', 'position': 184, 'mut_aa': 'V'}
```

---

#### `find_overlapping_epitopes(position, protein, epitope_df)`

Find CTL epitopes overlapping a given position.

**Parameters:**
- `position` (int): Protein position
- `protein` (str): Protein name
- `epitope_df` (pd.DataFrame): Epitope data from load_lanl_ctl()

**Returns:**
- `pd.DataFrame`: Subset of epitopes overlapping the position

**Example:**
```python
from position_mapper import find_overlapping_epitopes
from unified_data_loader import load_lanl_ctl

epitopes = load_lanl_ctl()
overlaps = find_overlapping_epitopes(103, 'RT', epitopes)
print(f"Overlapping epitopes: {len(overlaps)}")
```

---

## 3. codon_extraction.py

### Constants

#### `HIV_CODON_TABLE`

Representative codon for each amino acid based on HIV codon usage.

```python
HIV_CODON_TABLE = {
    'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA',
    'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'K': 'AAA', 'L': 'CTT', 'M': 'ATG', 'N': 'AAT',
    'P': 'CCT', 'Q': 'CAA', 'R': 'AGA', 'S': 'TCT',
    'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT',
    '*': 'TAA'
}
```

---

### Functions

#### `encode_codon_3adic(codon)`

Encode a codon using 3-adic valuation into Poincaré ball.

**Parameters:**
- `codon` (str): Three-letter nucleotide sequence

**Returns:**
- `np.ndarray`: 16-dimensional embedding vector

**Example:**
```python
from codon_extraction import encode_codon_3adic

embedding = encode_codon_3adic('ATG')
print(f"Shape: {embedding.shape}")  # (16,)
print(f"Norm: {np.linalg.norm(embedding):.3f}")  # < 1.0
```

---

#### `encode_amino_acid(aa)`

Encode an amino acid using its representative codon.

**Parameters:**
- `aa` (str): Single-letter amino acid code

**Returns:**
- `np.ndarray`: 16-dimensional embedding vector

**Example:**
```python
from codon_extraction import encode_amino_acid

embedding = encode_amino_acid('M')  # Methionine
print(f"Shape: {embedding.shape}")  # (16,)
```

---

#### `encode_mutation_pair(wt_aa, mut_aa)`

Calculate geometric features for a mutation.

**Parameters:**
- `wt_aa` (str): Wild-type amino acid
- `mut_aa` (str): Mutant amino acid

**Returns:**
- `dict`: Geometric features including:
  - `euclidean_distance`: Euclidean distance in embedding space
  - `hyperbolic_distance`: Poincaré ball distance
  - `radial_change`: Change in distance from origin
  - `angular_change`: Angular displacement

**Example:**
```python
from codon_extraction import encode_mutation_pair

features = encode_mutation_pair('M', 'V')
print(f"Hyperbolic distance: {features['hyperbolic_distance']:.3f}")
```

---

#### `encode_amino_acid_sequence(sequence)`

Encode a full amino acid sequence.

**Parameters:**
- `sequence` (str): Amino acid sequence

**Returns:**
- `dict`: Sequence-level features including:
  - `mean_embedding`: Average embedding across positions
  - `per_position`: List of position embeddings
  - `mean_radius`: Mean radial position
  - `spread`: Standard deviation of radii

**Example:**
```python
from codon_extraction import encode_amino_acid_sequence

features = encode_amino_acid_sequence('SLYNTVATL')
print(f"Mean radius: {features['mean_radius']:.3f}")
print(f"Spread: {features['spread']:.3f}")
```

---

#### `find_glycan_sites(sequence)`

Find potential N-linked glycosylation sites (NXT/NXS motifs).

**Parameters:**
- `sequence` (str): Amino acid sequence

**Returns:**
- `list`: Positions (0-indexed) of potential glycan sites

**Example:**
```python
from codon_extraction import find_glycan_sites

sites = find_glycan_sites('CTRPNNNTRKGIHIGPGRAFYATGDIIGDIRQAHC')
print(f"Glycan sites: {sites}")  # [4, 5, 6] for NNN
```

---

## 4. Analysis Scripts

### analyze_stanford_resistance.py

**Main Function:**
```python
def main():
    """Run complete Stanford HIVDB analysis."""
```

**Output Files:**
- `mutation_distances.csv`
- `position_constraints.csv`
- `cross_resistance_matrix.csv`
- `ANALYSIS_REPORT.md`
- 4 PNG visualizations

---

### analyze_ctl_escape_expanded.py

**Main Function:**
```python
def main():
    """Run complete CTL escape analysis."""
```

**Output Files:**
- `epitope_data.csv`
- `hla_analysis.csv`
- `CTL_ANALYSIS_REPORT.md`
- 5 PNG visualizations

---

### analyze_catnap_neutralization.py

**Main Function:**
```python
def main():
    """Run complete CATNAP neutralization analysis."""
```

**Output Files:**
- `breadth_data.csv`
- `bnab_sensitivity.csv`
- `virus_susceptibility.csv`
- `NEUTRALIZATION_REPORT.md`
- 5 PNG visualizations

---

### analyze_tropism_switching.py

**Main Function:**
```python
def main():
    """Run complete tropism analysis."""
```

**Output Files:**
- `v3_data.csv`
- `position_importance.csv`
- `TROPISM_REPORT.md`
- 3 PNG visualizations

---

### cross_dataset_integration.py

**Main Function:**
```python
def main():
    """Run cross-dataset integration analysis."""
```

**Output Files:**
- `resistance_epitope_overlaps.csv`
- `tradeoff_scores.csv`
- `vaccine_targets.csv`
- `INTEGRATION_REPORT.md`
- 3 PNG visualizations

---

## 5. Usage Examples

### Complete Workflow

```python
# Load all data
from unified_data_loader import (
    load_stanford_hivdb,
    load_lanl_ctl,
    load_catnap,
    load_v3_coreceptor
)

# Load datasets
resistance = load_stanford_hivdb()
epitopes = load_lanl_ctl()
neutralization = load_catnap()
tropism = load_v3_coreceptor()

print(f"Loaded {len(resistance) + len(epitopes) + len(neutralization) + len(tropism)} total records")
```

### Custom Mutation Analysis

```python
from codon_extraction import encode_mutation_pair

# Analyze a new mutation
wt, mut = 'M', 'I'  # M184I
features = encode_mutation_pair(wt, mut)

print(f"M→I mutation:")
print(f"  Hyperbolic distance: {features['hyperbolic_distance']:.3f}")
print(f"  Radial change: {features['radial_change']:.3f}")
```

### Epitope Overlap Search

```python
from position_mapper import find_overlapping_epitopes
from unified_data_loader import load_lanl_ctl

epitopes = load_lanl_ctl()

# Find epitopes overlapping RT position 103 (K103N)
overlaps = find_overlapping_epitopes(103, 'RT', epitopes)
print(f"K103N overlaps with {len(overlaps)} epitopes")
for _, ep in overlaps.iterrows():
    print(f"  {ep['Epitope']} (HLA: {ep['HLA']})")
```

---

## 6. Error Handling

All functions raise standard Python exceptions:

| Exception | Cause |
|-----------|-------|
| FileNotFoundError | Data file missing |
| ValueError | Invalid input (e.g., unknown amino acid) |
| KeyError | Missing column in data |

**Example:**
```python
try:
    embedding = encode_amino_acid('X')  # Invalid AA
except ValueError as e:
    print(f"Error: {e}")
```

---

## 7. Performance Notes

| Function | Typical Time | Memory |
|----------|-------------|--------|
| load_stanford_hivdb() | 2-3 sec | 50 MB |
| load_catnap() | 5-10 sec | 500 MB |
| encode_mutation_pair() | <1 ms | <1 KB |
| encode_amino_acid_sequence(100 AA) | 5 ms | 10 KB |

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
