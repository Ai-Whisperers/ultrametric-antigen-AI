# Data Dictionary

## Complete Reference for All Generated Data Files

**Version:** 1.0
**Last Updated:** December 25, 2025

---

## Table of Contents

1. [Drug Resistance Data Files](#1-drug-resistance-data-files)
2. [CTL Epitope Data Files](#2-ctl-epitope-data-files)
3. [Antibody Neutralization Data Files](#3-antibody-neutralization-data-files)
4. [Tropism Data Files](#4-tropism-data-files)
5. [Integration Data Files](#5-integration-data-files)
6. [Common Fields](#6-common-fields)

---

## 1. Drug Resistance Data Files

### 1.1 mutation_distances.csv

**Location:** `results/stanford_resistance/mutation_distances.csv`

**Description:** Per-mutation geometric features and resistance data

**Rows:** 90,269 (one per mutation instance)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| seq_id | string | Sequence identifier | "SEQ001" |
| mutation | string | Mutation notation | "M184V" |
| position | int | Amino acid position | 184 |
| wt_aa | char | Wild-type amino acid | "M" |
| mut_aa | char | Mutant amino acid | "V" |
| wt_codon | string | Wild-type codon | "ATG" |
| mut_codon | string | Mutant codon | "GTT" |
| drug_class | string | Drug class (PI/NRTI/NNRTI/INI) | "NRTI" |
| euclidean_distance | float | Euclidean distance in embedding | 0.234 |
| hyperbolic_distance | float | Hyperbolic distance (Poincare) | 0.456 |
| radial_change | float | Change in radial position | 0.089 |
| angular_change | float | Angular displacement (radians) | 0.234 |
| max_fold_change | float | Maximum fold-change across drugs | 125.3 |
| mean_fold_change | float | Geometric mean fold-change | 45.6 |
| mutation_type | string | Primary/Accessory/Unknown | "Primary" |
| boundary_crossing | int | Number of cluster boundaries crossed | 2 |
| neighbor_density | float | Local embedding density | 4.56 |

### 1.2 position_constraints.csv

**Location:** `results/stanford_resistance/position_constraints.csv`

**Description:** Per-position constraint metrics

**Rows:** 847 (unique positions with mutations)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| position | int | Amino acid position | 184 |
| protein | string | Protein region (PR/RT/IN) | "RT" |
| n_mutations | int | Number of different mutations | 5 |
| mean_radius | float | Mean radial position | 0.72 |
| radius_variance | float | Variance of radii | 0.034 |
| mean_distance | float | Mean hyperbolic distance | 0.567 |
| max_fold_change | float | Maximum resistance fold-change | 150.0 |
| constraint_score | float | Computed constraint metric | 2.34 |
| conservation | float | Sequence conservation score | 0.89 |

### 1.3 cross_resistance_matrix.csv

**Location:** `results/stanford_resistance/cross_resistance_matrix.csv`

**Description:** Pairwise drug cross-resistance matrix

**Rows:** 25 drugs × 25 drugs

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| drug1 | string | First drug name | "AZT" |
| drug2 | string | Second drug name | "3TC" |
| jaccard | float | Jaccard similarity of mutations | 0.234 |
| shared_mutations | int | Number of shared mutations | 12 |
| geometric_overlap | float | Overlap in embedding space | 0.456 |

---

## 2. CTL Epitope Data Files

### 2.1 epitope_data.csv

**Location:** `results/ctl_escape_expanded/epitope_data.csv`

**Description:** Complete epitope information with geometric features

**Rows:** 2,115 (one per epitope)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| epitope_id | int | Unique epitope identifier | 1 |
| epitope_seq | string | Epitope amino acid sequence | "SLYNTVATL" |
| length | int | Epitope length | 9 |
| protein | string | HIV protein | "Gag" |
| hxb2_start | int | Start position (HXB2) | 77 |
| hxb2_end | int | End position (HXB2) | 85 |
| hla | string | HLA restriction | "A*02:01" |
| hla_supertype | string | HLA supertype | "A2" |
| n_hla | int | Number of HLA alleles | 12 |
| mean_radius | float | Mean radial position | 0.67 |
| spread | float | Radial spread (std dev) | 0.12 |
| escape_velocity | float | Computed escape velocity | 0.34 |
| mean_embedding | list[float] | Mean 16D embedding | [0.12, 0.34, ...] |
| conservation | float | Sequence conservation | 0.87 |
| cluster_id | int | Assigned cluster | 3 |
| vaccine_score | float | Vaccine priority score | 1.45 |

### 2.2 hla_analysis.csv

**Location:** `results/ctl_escape_expanded/hla_analysis.csv`

**Description:** HLA-stratified escape analysis

**Rows:** 240 (unique HLA alleles)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| hla | string | HLA allele | "A*02:01" |
| supertype | string | HLA supertype | "A2" |
| n_epitopes | int | Number of restricted epitopes | 193 |
| mean_radius | float | Mean epitope radius | 0.67 |
| mean_escape_velocity | float | Mean escape velocity | 0.34 |
| std_escape_velocity | float | Std of escape velocity | 0.12 |
| dominant_protein | string | Most common protein | "Gag" |
| population_freq | float | Population frequency | 0.28 |

### 2.3 cluster_assignments.csv

**Location:** `results/ctl_escape_expanded/cluster_assignments.csv`

**Description:** Epitope cluster memberships

**Rows:** 2,115 (one per epitope)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| epitope_id | int | Epitope identifier | 1 |
| cluster_id | int | Primary cluster assignment | 3 |
| cluster_probability | float | Cluster membership probability | 0.89 |
| silhouette_score | float | Cluster quality for this epitope | 0.67 |

---

## 3. Antibody Neutralization Data Files

### 3.1 breadth_data.csv

**Location:** `results/catnap_neutralization/breadth_data.csv`

**Description:** Antibody breadth and potency metrics

**Rows:** 1,123 (unique antibodies)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| antibody | string | Antibody name | "VRC01" |
| epitope_class | string | Epitope class | "CD4bs" |
| n_tested | int | Viruses tested | 9724 |
| n_sensitive | int | Viruses neutralized | 6701 |
| breadth | float | Fraction neutralized (IC50<50) | 0.689 |
| geo_mean_ic50 | float | Geometric mean IC50 | 0.580 |
| median_ic50 | float | Median IC50 | 0.234 |
| potency_rank | int | Potency ranking | 7 |
| epitope_centrality | float | Target region centrality | 0.81 |
| is_bnab | bool | Known broadly neutralizing | True |

### 3.2 bnab_sensitivity.csv

**Location:** `results/catnap_neutralization/bnab_sensitivity.csv`

**Description:** Detailed bnAb sensitivity profiles

**Rows:** 16 (known bnAbs analyzed)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| antibody | string | bnAb name | "VRC01" |
| epitope_class | string | Epitope class | "CD4bs" |
| n_tested | int | Viruses tested | 9724 |
| pct_sensitive | float | Percent sensitive | 68.9 |
| geo_mean_ic50 | float | Geometric mean IC50 | 0.580 |
| median_ic50 | float | Median IC50 | 0.234 |
| min_ic50 | float | Minimum IC50 | 0.0002 |
| max_ic50 | float | Maximum IC50 | 200.0 |
| epitope_radius | float | Target epitope radius | 0.81 |
| epitope_spread | float | Target epitope spread | 0.09 |

### 3.3 virus_susceptibility.csv

**Location:** `results/catnap_neutralization/virus_susceptibility.csv`

**Description:** Virus susceptibility patterns

**Rows:** 2,960 (unique viruses)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| virus | string | Virus identifier | "Q23.17" |
| subtype | string | HIV-1 subtype | "A" |
| n_antibodies | int | Antibodies tested | 47 |
| n_sensitive | int | Antibodies neutralizing | 42 |
| pct_sensitive | float | Percent susceptible | 89.4 |
| tier | string | Neutralization tier | "1A" |
| mean_ic50 | float | Mean IC50 across antibodies | 0.089 |
| cd4bs_susceptible | bool | CD4bs bnAb sensitive | True |
| v3_susceptible | bool | V3-glycan bnAb sensitive | True |
| mper_susceptible | bool | MPER bnAb sensitive | True |

### 3.4 potency_by_class.csv

**Location:** `results/catnap_neutralization/potency_by_class.csv`

**Description:** Epitope class summary statistics

**Rows:** 5 (epitope classes)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| epitope_class | string | Epitope class name | "CD4bs" |
| n_antibodies | int | Antibodies in class | 5 |
| n_records | int | Neutralization records | 17899 |
| geometric_mean_ic50 | float | Geo mean IC50 | 1.121 |
| median_ic50 | float | Median IC50 | 0.566 |
| mean_log_ic50 | float | Mean log10(IC50) | 0.050 |
| std_log_ic50 | float | Std of log10(IC50) | 1.137 |
| min_ic50 | float | Minimum IC50 | 0.0002 |
| max_ic50 | float | Maximum IC50 | 200.0 |

---

## 4. Tropism Data Files

### 4.1 v3_data.csv

**Location:** `results/tropism/v3_data.csv`

**Description:** V3 sequences with tropism labels and embeddings

**Rows:** 2,932 (V3 sequences)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| sequence_id | int | Sequence identifier | 1 |
| v3_sequence | string | V3 amino acid sequence | "CTRPNNNTRK..." |
| length | int | V3 length | 35 |
| is_r5 | bool | CCR5-tropic | True |
| is_x4 | bool | CXCR4-tropic | False |
| is_dual | bool | Dual-tropic | False |
| net_charge | int | Net charge | 4 |
| pos11_aa | char | Amino acid at position 11 | "S" |
| pos22_aa | char | Amino acid at position 22 | "D" |
| pos25_aa | char | Amino acid at position 25 | "D" |
| mean_radius | float | Mean embedding radius | 0.934 |
| predicted_tropism | string | ML prediction | "R5" |
| prediction_score | float | Prediction confidence | 0.87 |

### 4.2 position_importance.csv

**Location:** `results/tropism/position_importance.csv`

**Description:** Per-position tropism discrimination

**Rows:** 35 (V3 positions) - Note: varies by alignment

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| position | int | V3 position number | 22 |
| separation | float | R5/X4 centroid separation | 0.591 |
| r5_mean_radius | float | Mean radius for R5 | 0.949 |
| x4_mean_radius | float | Mean radius for X4 | 0.942 |
| radius_pval | float | Mann-Whitney p-value | 1.6e-10 |
| is_key_position | bool | Known determinant | False |
| feature_importance | float | ML feature importance | 0.341 |

### 4.3 classifier_metrics.csv

**Location:** `results/tropism/classifier_metrics.csv`

**Description:** ML classifier performance

**Rows:** 5 (classifiers tested)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| classifier | string | Model name | "Random Forest" |
| accuracy | float | Test accuracy | 0.850 |
| auc | float | Area under ROC | 0.843 |
| precision | float | Precision | 0.73 |
| recall | float | Recall | 0.69 |
| f1 | float | F1 score | 0.71 |
| cv_mean | float | CV mean accuracy | 0.868 |
| cv_std | float | CV standard deviation | 0.009 |

---

## 5. Integration Data Files

### 5.1 resistance_epitope_overlaps.csv

**Location:** `results/integrated/resistance_epitope_overlaps.csv`

**Description:** All resistance-epitope position overlaps

**Rows:** 16,054 (overlap instances)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| overlap_id | int | Unique identifier | 1 |
| mutation | string | Resistance mutation | "K103N" |
| position | int | HXB2 position | 103 |
| drug_class | string | Drug class | "NNRTI" |
| fold_change | float | Maximum fold-change | 45.6 |
| epitope_id | int | Epitope identifier | 234 |
| epitope_seq | string | Epitope sequence | "ITLWQRPLV" |
| hla | string | HLA restriction | "A*02:01" |
| n_hla | int | Number of HLAs | 15 |
| protein | string | Protein | "RT" |

### 5.2 tradeoff_scores.csv

**Location:** `results/integrated/tradeoff_scores.csv`

**Description:** Calculated trade-off scores for overlaps

**Rows:** 16,054 (one per overlap)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| overlap_id | int | Overlap identifier | 1 |
| mutation | string | Mutation notation | "K103N" |
| drug_class | string | Drug class | "NNRTI" |
| epitope_seq | string | Epitope sequence | "ITLWQRPLV" |
| n_hla | int | HLA count | 15 |
| fold_change | float | Resistance fold-change | 45.6 |
| tradeoff_score | float | Computed score | 5.32 |
| score_rank | int | Rank by score | 7 |

### 5.3 constraint_landscape.csv

**Location:** `results/integrated/constraint_landscape.csv`

**Description:** Multi-pressure constraint map

**Rows:** 847 (constrained positions)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| position | int | HXB2 position | 103 |
| protein | string | Protein region | "RT" |
| drug_constraint | float | Drug resistance pressure | 2.3 |
| ctl_constraint | float | CTL pressure | 1.2 |
| ab_constraint | float | Antibody pressure | 0.3 |
| tropism_constraint | float | Tropism constraint | 0.0 |
| total_constraint | float | Sum of constraints | 3.8 |
| constraint_rank | int | Rank by total | 1 |
| radial_position | float | Hyperbolic radius | 0.42 |

### 5.4 vaccine_targets.csv

**Location:** `results/integrated/vaccine_targets.csv`

**Description:** Ranked vaccine target epitopes

**Rows:** 387 (candidate targets)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| rank | int | Priority ranking | 1 |
| epitope | string | Epitope sequence | "TPQDLNTML" |
| protein | string | HIV protein | "Gag" |
| hxb2_start | int | Start position | 180 |
| hxb2_end | int | End position | 188 |
| n_hla | int | HLA count | 25 |
| resistance_overlap | bool | Overlaps drug position | False |
| escape_velocity | float | Escape velocity | 0.28 |
| conservation | float | Conservation score | 0.92 |
| vaccine_score | float | Priority score | 2.238 |

---

## 6. Common Fields

### 6.1 Standard Field Types

| Field Type | Description | Format |
|------------|-------------|--------|
| Amino acid | Single letter code | A-Y, * |
| Codon | Three nucleotides | ATG, etc. |
| HXB2 position | Reference coordinate | Integer |
| HLA | HLA nomenclature | Gene*allele:protein |
| Drug class | Four classes | PI/NRTI/NNRTI/INI |
| p-value | Statistical significance | Scientific notation |
| Score | Computed metric | Float, typically 0-5 |

### 6.2 Missing Value Conventions

| Symbol | Meaning |
|--------|---------|
| NA | Not available / not tested |
| NaN | Not a number (calculation error) |
| -1 | Not applicable |
| "" | Empty string (categorical NA) |

### 6.3 File Encoding

All files are:
- **Format:** CSV (comma-separated values)
- **Encoding:** UTF-8
- **Line endings:** Unix (LF)
- **Decimal:** Period (.)
- **Date format:** YYYY-MM-DD

---

## Appendix: Complete File Inventory

### All Generated Files

```
results/
├── stanford_resistance/
│   ├── mutation_distances.csv         (90,269 rows)
│   ├── position_constraints.csv       (847 rows)
│   ├── cross_resistance_matrix.csv    (625 rows)
│   ├── primary_accessory_predictions.csv
│   ├── ANALYSIS_REPORT.md
│   ├── resistance_distance_correlation.png
│   ├── primary_accessory_separation.png
│   ├── cross_resistance_network.png
│   └── position_constraint_heatmap.png
│
├── ctl_escape_expanded/
│   ├── epitope_data.csv               (2,115 rows)
│   ├── hla_analysis.csv               (240 rows)
│   ├── cluster_assignments.csv        (2,115 rows)
│   ├── vaccine_targets_ctl.csv
│   ├── CTL_ANALYSIS_REPORT.md
│   ├── protein_radial_distribution.png
│   ├── hla_escape_landscape.png
│   ├── conservation_geometry_correlation.png
│   ├── epitope_clusters.png
│   └── vaccine_priority_ranking.png
│
├── catnap_neutralization/
│   ├── breadth_data.csv               (1,123 rows)
│   ├── bnab_sensitivity.csv           (16 rows)
│   ├── virus_susceptibility.csv       (2,960 rows)
│   ├── potency_by_class.csv           (5 rows)
│   ├── NEUTRALIZATION_REPORT.md
│   ├── breadth_distribution.png
│   ├── bnab_sensitivity.png
│   ├── antibody_clustering.png
│   ├── virus_susceptibility.png
│   └── potency_by_class.png
│
├── tropism/
│   ├── v3_data.csv                    (2,932 rows)
│   ├── position_importance.csv        (35 rows)
│   ├── classifier_metrics.csv         (5 rows)
│   ├── TROPISM_REPORT.md
│   ├── tropism_separation.png
│   ├── position_importance.png
│   └── classifier_performance.png
│
└── integrated/
    ├── resistance_epitope_overlaps.csv (16,054 rows)
    ├── tradeoff_scores.csv            (16,054 rows)
    ├── constraint_landscape.csv       (847 rows)
    ├── vaccine_targets.csv            (387 rows)
    ├── INTEGRATION_REPORT.md
    ├── tradeoff_landscape.png
    ├── constraint_map.png
    └── vaccine_targets.png
```

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
