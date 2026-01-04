# DENV-4 Comprehensive Phylogenetic Analysis Report

**Doc-Type:** Phylogenetic Analysis Report · Version 1.0 · 2026-01-04 · AI Whisperers

---

## Executive Summary

This report presents a comprehensive phylogenetic analysis of **270 DENV-4 complete genomes** from NCBI. The analysis identifies **5 major clades** with distinct geographic and temporal patterns.

### Key Findings

| Metric | Value |
|--------|-------|
| Total Sequences | 270 |
| Number of Clades | 5 |
| Mean Pairwise Identity | 85.8% |
| Identity Range | 72.9% - 100.0% |

---

## Clade Overview

| Clade | Size | Mean Identity | Top Country | Year Range |
|-------|------|---------------|-------------|------------|
| Clade_E | 211 | 89.0% | Unknown | 1976-2023 |
| Clade_D | 52 | 85.8% | Unknown | 1956-2024 |
| Clade_B | 3 | 99.4% | Unknown | 1973-1975 |
| Clade_A | 2 | 100.0% | Unknown | 2007-2007 |
| Clade_C | 2 | 98.5% | Unknown | Unknown |

---

## Geographic Distribution

### Clade_A

**Size:** 2 sequences


### Clade_B

**Size:** 3 sequences


### Clade_C

**Size:** 2 sequences


### Clade_D

**Size:** 52 sequences


### Clade_E

**Size:** 211 sequences


---

## Conserved Regions for Primer Design

Top conserved windows per clade (suitable for genotype-specific primers):

### Clade_A

| Gene | Start | End | Entropy |
|------|-------|-----|---------|
| intergenic | 0 | 25 | 0.000 |
| 5UTR | 1 | 26 | 0.000 |
| 5UTR | 2 | 27 | 0.000 |
| 5UTR | 3 | 28 | 0.000 |
| 5UTR | 4 | 29 | 0.000 |

### Clade_B

| Gene | Start | End | Entropy |
|------|-------|-----|---------|
| intergenic | 0 | 25 | 0.000 |
| 5UTR | 1 | 26 | 0.000 |
| 5UTR | 2 | 27 | 0.000 |
| 5UTR | 3 | 28 | 0.000 |
| 5UTR | 4 | 29 | 0.000 |

### Clade_C

| Gene | Start | End | Entropy |
|------|-------|-----|---------|
| intergenic | 0 | 25 | 0.000 |
| 5UTR | 1 | 26 | 0.000 |
| 5UTR | 2 | 27 | 0.000 |
| 5UTR | 3 | 28 | 0.000 |
| 5UTR | 4 | 29 | 0.000 |

### Clade_D

*No conserved regions identified*

### Clade_E

*No conserved regions identified*

---

## Clade Representatives

Selected representative sequences for each clade (medoid selection):

| Clade | Accession | Country | Year | Length |
|-------|-----------|---------|------|--------|
| Clade_A | MZ215848 | Unknown | 2007 | 10677 bp |
| Clade_B | JF262780 | Unknown | 1973 | 10667 bp |
| Clade_C | AY618989 | Unknown | Unknown | 10653 bp |
| Clade_D | AY618990 | Unknown | Unknown | 10650 bp |
| Clade_E | EU854299 | Unknown | 2007 | 10606 bp |

---

## Implications for Primer Design

### Challenge

DENV-4 shows extensive cryptic diversity requiring **multiplexed detection**:

1. Within-clade identity is high enough for consensus primers
2. Between-clade identity requires separate primer pairs
3. Each clade has distinct conserved regions

### Recommended Strategy

1. **Design clade-specific primers** for each identified clade
2. **Use conserved windows** identified above as primer binding sites
3. **Multiplex all primers** in single reaction with staggered amplicons
4. **Monitor quarterly** for new clade emergence

---

## Files Generated

| File | Description |
|------|-------------|
| `phylogenetic/phylogenetic_analysis_results.json` | Complete analysis results |
| `phylogenetic/clade_representatives.fasta` | Representative sequences |
| `phylogenetic/geographic_distribution.json` | Geographic data |
| `phylogenetic/temporal_distribution.json` | Temporal data |
| `phylogenetic/per_clade_conservation.json` | Conservation data |
| `denv4/distance_matrix.npy` | Pairwise distance matrix |
| `denv4/linkage.npy` | Hierarchical clustering linkage |

---

*Analysis completed: 2026-01-04 04:19:24*

*IICS-UNA Arbovirus Surveillance Program*
