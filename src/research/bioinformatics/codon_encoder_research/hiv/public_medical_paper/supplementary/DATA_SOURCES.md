# Data Sources

**Purpose**: Document the databases used in this research and their characteristics.

---

## Overview

| Database | Records | Primary Use | Access |
|:---------|--------:|:------------|:-------|
| Stanford HIV Drug Resistance | 90,269 mutations | Drug resistance analysis | Public |
| LANL CTL Epitope Database | 2,115 epitopes | Cellular immunity | Public |
| CATNAP | 189,879 records | Antibody neutralization | Public |
| V3 Tropism Dataset | 2,932 sequences | Coreceptor tropism | Public |
| **Total Integrated** | **202,085** | Multi-pressure analysis | - |

---

## 1. Stanford HIV Drug Resistance Database

### Overview

The Stanford HIV Drug Resistance Database is the global standard for HIV drug resistance interpretation. It provides curated data on resistance mutations and their clinical significance.

### Data Used

| Metric | Value |
|:-------|------:|
| Total sequences analyzed | 7,154 |
| Total mutations | 90,269 |
| Unique mutation positions | 3,647 |
| Drug classes covered | 4 (NRTI, NNRTI, PI, INSTI) |

### Key Fields

| Field | Description |
|:------|:------------|
| Mutation | Amino acid change (e.g., K103N) |
| Position | Codon position in gene |
| Drug Class | NRTI, NNRTI, PI, INSTI |
| Fold Change | Resistance level (ratio vs. wild-type) |
| Type | Primary or accessory |

### Access

- **URL**: https://hivdb.stanford.edu/
- **Citation**: Rhee et al., Nucleic Acids Research, 2003
- **License**: Public access for research

---

## 2. LANL CTL Epitope Database

### Overview

The Los Alamos National Laboratory maintains the most comprehensive database of HIV-1 CTL epitopes, including HLA restrictions and escape mutations.

### Data Used

| Metric | Value |
|:-------|------:|
| Total epitopes | 2,115 |
| Unique HLA restrictions | 240 |
| Proteins covered | 9 (Gag, Pol, Env, Nef, Rev, Tat, Vif, Vpr, Vpu) |
| Escape mutations documented | 847 |

### Key Fields

| Field | Description |
|:------|:------------|
| Epitope Sequence | 8-11 amino acid peptide |
| Protein | Source protein |
| Position | Start-end in protein |
| HLA | Restricting HLA allele(s) |
| Escape Mutations | Known escape variants |
| Clinical Association | Controller status, viral load |

### Access

- **URL**: https://www.hiv.lanl.gov/content/immunology/
- **Citation**: Yusim et al., HIV Molecular Immunology, 2023
- **License**: Public access for research

---

## 3. CATNAP (Compile, Analyze, and Tally NAb Panels)

### Overview

CATNAP aggregates neutralization data for broadly neutralizing antibodies against HIV-1, enabling systematic analysis of antibody breadth and potency.

### Data Used

| Metric | Value |
|:-------|------:|
| Total neutralization records | 189,879 |
| Antibodies characterized | 100+ |
| Virus strains tested | 2,500+ |
| Clades represented | A, B, C, D, CRF01_AE, others |

### Key Fields

| Field | Description |
|:------|:------------|
| Antibody | bnAb identifier |
| Virus | Strain identifier |
| IC50 | Neutralization potency (μg/mL) |
| IC80 | Neutralization at 80% |
| Clade | Viral subtype |
| Epitope Class | CD4bs, V2, V3, MPER, interface |

### Access

- **URL**: https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/
- **Citation**: Yoon et al., Nucleic Acids Research, 2015
- **License**: Public access for research

---

## 4. V3 Tropism Dataset

### Overview

V3 loop sequences with phenotypic tropism determination, used to develop and validate tropism prediction algorithms.

### Data Used

| Metric | Value |
|:-------|------:|
| Total V3 sequences | 2,932 |
| CCR5-only (R5) | 2,229 (76%) |
| CXCR4-using (X4/R5X4) | 702 (24%) |
| Sequence length | 35 amino acids |

### Key Fields

| Field | Description |
|:------|:------------|
| Sequence | V3 loop amino acid sequence |
| Tropism | R5, X4, or R5X4 |
| Clade | Viral subtype |
| Source | Clinical or culture-derived |
| Validation | Phenotypic assay used |

### Access

- **URL**: https://www.hiv.lanl.gov/content/sequence/HIV/
- **Citation**: Multiple sources compiled
- **License**: Public access for research

---

## Data Integration Methods

### Mapping Across Databases

| Integration | Method | Purpose |
|:------------|:-------|:--------|
| Resistance ↔ Epitope | Position-based overlap | Trade-off identification |
| Epitope ↔ Antibody | Protein region mapping | Multi-pressure landscape |
| V3 ↔ Antibody | Tropism + neutralization | Tropism-sensitivity correlation |

### Quality Control

| Check | Criteria | Result |
|:------|:---------|:-------|
| Sequence validation | Standard codon usage | 99.2% pass |
| Position mapping | HXB2 reference alignment | Standardized |
| Duplicate removal | Identical entries | 3.2% removed |
| Outlier detection | Statistical thresholds | 0.8% flagged |

---

## Computational Resources

### Analysis Infrastructure

| Component | Specification |
|:----------|:--------------|
| Primary analysis | Python 3.11, PyTorch 2.0 |
| Geometric framework | Custom implementation |
| Statistical analysis | SciPy, statsmodels |
| Visualization | Matplotlib, Seaborn |

### Reproducibility

All analysis scripts and processed data are maintained in the research repository. Raw data links to original database sources.

---

## Data Availability Statement

This research uses publicly available databases. Researchers wishing to reproduce or extend this work can access:

1. **Stanford HIVDB**: Direct download via API
2. **LANL Databases**: Web interface and bulk download
3. **CATNAP**: Web interface and bulk download
4. **Processed Data**: Available upon request for research collaboration

---

## Citation Guidelines

When using findings from this research, please cite:

1. **Original databases** as specified by their maintainers
2. **This research** (citation information available upon publication)
3. **Geometric framework** (if methodology is extended)

---

*For data access inquiries or collaboration, contact the research team.*
