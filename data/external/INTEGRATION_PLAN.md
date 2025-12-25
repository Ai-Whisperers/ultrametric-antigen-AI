# HIV Dataset Integration Plan

## Overview

This document maps downloaded HIV datasets to the p-adic hyperbolic codon analysis capabilities and proposes specific integration strategies.

---

## Dataset Summary

| Dataset | Source | Size | Key Content | Primary Use |
|---------|--------|------|-------------|-------------|
| HIV-data | GitHub | 120 MB | 13,000+ sequences (env, genome) | Sequence diversity training |
| HIV-1_Paper | GitHub | 4 MB | 33 patient sequences + mutations | Drug resistance validation |
| human_hiv_ppi | HuggingFace | 3 MB | 16,179 protein interactions | Interaction site analysis |
| HIV_V3_coreceptor | HuggingFace | 0.06 MB | 2,935 V3 sequences + tropism | Coreceptor tropism prediction |
| cview_gp120 | Zenodo | 1.7 MB | 712 aligned gp120 sequences | Glycan shield analysis |
| hiv-aids-dataset | Kaggle | 0.1 MB | Global epidemiological data | Context/validation |
| corgis_aids | CSV | 0.4 MB | 2,759 country-year records | Epidemiological trends |

---

## Integration Strategies

### 1. Expand CTL Escape Analysis

**Current State:** 6 epitopes, 9 mutations
**Target:** 100+ epitopes, 500+ mutations

**Data Sources:**
- `HIV-data/HIV1_ALL_2021_env_DNA.2000.fasta` (8,394 env sequences)
- HuggingFace `HIV_V3_coreceptor` (2,935 V3 sequences)

**Integration Steps:**
```python
# 1. Extract V3 loop from env sequences
# 2. Identify mutations relative to HXB2 reference
# 3. Map mutations to hyperbolic space using codon encoder
# 4. Correlate with tropism labels (CCR5/CXCR4)

from pathlib import Path
import torch

# Load V3 coreceptor data
v3_data = load_v3_coreceptor_dataset()

# For each V3 sequence with known tropism
for seq in v3_data:
    codons = sequence_to_codons(seq['sequence'])
    embeddings = encode_codons(codons, encoder)

    # Analyze hyperbolic distance patterns
    # Correlate with CCR5/CXCR4 tropism
```

**Expected Outcomes:**
- Identify hyperbolic signatures of coreceptor switching
- Map "escape zones" in codon space for tropism change
- Validate boundary-crossing hypothesis with larger dataset

---

### 2. Expand Drug Resistance Analysis

**Current State:** 18 mutations across 4 drug classes
**Target:** 200+ mutations with clinical outcomes

**Data Sources:**
- `HIV-1_Paper/Mapping_Results/*.csv` (amino acid profiles)
- `HIV-1_Paper/Individual_Representative_Sequences/` (FASTA)
- `HIV-data` (full genome sequences)

**Integration Steps:**
```python
# 1. Parse Nigeria patient amino acid profiles
# 2. Identify mutations vs HXB2 reference
# 3. Extract codon-level changes from FASTA
# 4. Calculate hyperbolic distances
# 5. Correlate with known resistance phenotypes

# Load Nigeria patient data
nigeria_data = load_nigeria_mapping_results()

# For each patient
for patient_id in nigeria_data:
    # Get mutations in PR, RT, IN regions
    pr_mutations = get_region_mutations(patient_id, 'PR')
    rt_mutations = get_region_mutations(patient_id, 'RT')
    in_mutations = get_region_mutations(patient_id, 'IN')

    # Calculate hyperbolic distances
    for mutation in pr_mutations + rt_mutations + in_mutations:
        distance = calculate_hyperbolic_distance(mutation)
        # Compare with Stanford DB resistance scores
```

**Data Structure (Nigeria CSV):**
```
region | ref.aa.positions | TCS.number | A | C | D | E | F | G | ...
PR     | 1                | 5          | 0 | 0 | 0 | 0 | 0 | 0 | ...
```

**Expected Outcomes:**
- Validate distance-resistance correlation with real patient data
- Identify accessory vs primary mutation patterns
- Geographic/subtype-specific resistance patterns

---

### 3. Protein-Protein Interaction Analysis (NEW)

**Data Source:** HuggingFace `human_hiv_ppi` (16,179 interactions)

**Concept:** Analyze codons at HIV-human protein interaction interfaces

**Data Fields:**
- `hiv_protein_name`: e.g., "gp120", "Tat", "Vpr"
- `human_protein_name`: e.g., "CD4", "CXCR4", "CCR5"
- `interaction_type`: e.g., "binds", "inhibits"
- `hiv_protein_sequence`: Full protein sequence
- `human_protein_sequence`: Full protein sequence

**Integration Steps:**
```python
# 1. Load PPI dataset
# 2. Focus on key interactions (gp120-CD4, gp120-CCR5, etc.)
# 3. Extract interaction interface residues
# 4. Encode codons at interface
# 5. Analyze hyperbolic properties of interface codons

from datasets import load_dataset

# Load PPI data
ppi_data = load_parquet('data/external/huggingface/human_hiv_ppi/data/train-*.parquet')

# Filter for envelope interactions
env_interactions = ppi_data[ppi_data['hiv_protein_name'].str.contains('gp120|gp41')]

# Analyze CD4 binding site codons
cd4_binding = env_interactions[env_interactions['human_protein_name'] == 'CD4']
```

**Expected Outcomes:**
- Map interaction hotspots in hyperbolic space
- Identify conserved vs variable interface positions
- Predict mutation tolerance at binding interfaces

---

### 4. Enhanced Glycan Shield Analysis

**Current State:** Basic glycan position analysis
**Target:** Full gp120 glycan landscape with tropism correlation

**Data Sources:**
- `cview_gp120/` (712 aligned gp120, 636 CCR5 + 76 CXCR4)
- `HIV-data/HIV1_ALL_2021_env_DNA.2000.fasta`

**Integration Steps:**
```python
# 1. Load aligned gp120 sequences
# 2. Identify N-glycosylation sites (N-X-S/T motif)
# 3. Map glycan positions to hyperbolic space
# 4. Compare CCR5 vs CXCR4 glycan patterns

# Load gp120 alignment
gp120_seqs = load_fasta('data/external/zenodo/cview_gp120/USE_CASE_DATA/*.fasta')
ccr5_ids = load_txt('CCR5_TITLES.txt')
cxcr4_ids = load_txt('CXCR4_TITLES.txt')

# Find N-glycosylation sites
for seq_id, sequence in gp120_seqs.items():
    glycan_sites = find_glycosylation_sites(sequence)
    tropism = 'CCR5' if seq_id in ccr5_ids else 'CXCR4'

    # Encode codons at glycan sites
    for site in glycan_sites:
        codon = get_codon_at_position(sequence, site)
        embedding = encode_codon(codon)
        # Analyze radial position (conserved = center, variable = edge)
```

**Expected Outcomes:**
- Glycan site conservation patterns in hyperbolic space
- Differential glycan shielding between CCR5/CXCR4 viruses
- Identify glycan "holes" as therapeutic targets

---

### 5. Sequence Diversity Training

**Data Source:** `HIV-data/` (13,000+ sequences)

**Purpose:** Train/validate codon encoder on full HIV diversity

**Sequence Types Available:**
| File | Sequences | Description |
|------|-----------|-------------|
| HIV1_ALL_2021_env_DNA.2000.fasta | 8,394 | HIV-1 envelope |
| HIV1_ALL_2021_genome_DNA.5000.fasta | 5,053 | HIV-1 full genome |
| HIV2_ALL_2021_env_DNA.2000.fasta | 218 | HIV-2 envelope |
| SIV_ALL_2021_env_DNA.2000.fasta | 207 | SIV envelope |

**Integration Steps:**
```python
# 1. Parse all FASTA files
# 2. Extract codon usage statistics
# 3. Build codon frequency matrices by gene region
# 4. Validate encoder on diverse subtypes

# Codon usage analysis
codon_usage = {}
for fasta_file in HIV_DATA_DIR.glob('*.fasta'):
    for record in parse_fasta(fasta_file):
        subtype = extract_subtype(record.header)
        codons = sequence_to_codons(record.sequence)
        update_codon_counts(codon_usage, subtype, codons)

# Compare subtype B (training) vs other subtypes
validate_encoder_on_subtypes(encoder, codon_usage)
```

**Expected Outcomes:**
- Encoder generalization to HIV-2 and SIV
- Subtype-specific codon bias patterns
- Improved encoder training with diverse data

---

### 6. Epidemiological Context

**Data Sources:**
- `kaggle/hiv-aids-dataset/` (6 CSV files, 170 countries)
- `csv/corgis_aids.csv` (2,759 country-year records)

**Purpose:** Contextual analysis, not direct codon analysis

**Potential Uses:**
- Correlate mutation patterns with epidemic dynamics
- Map drug resistance emergence to ART rollout
- Identify high-burden regions for targeted analysis

---

## Proposed New Analysis Scripts

### Script 1: `expand_escape_analysis.py`
```python
"""
Expand CTL escape analysis using V3 coreceptor dataset
- Load 2,935 V3 sequences with CCR5/CXCR4 labels
- Extract codons from V3 loop
- Calculate hyperbolic distances for tropism-switching mutations
- Generate tropism escape landscape
"""
```

### Script 2: `analyze_nigeria_resistance.py`
```python
"""
Analyze drug resistance patterns from Nigeria patient data
- Parse amino acid profiles from Mapping_Results/*.csv
- Map mutations to codon space
- Validate against existing resistance results
- Compare African vs Western mutation patterns
"""
```

### Script 3: `ppi_interface_analysis.py`
```python
"""
Analyze HIV-human protein interaction interfaces
- Load 16k+ interactions from HuggingFace dataset
- Focus on key interactions (gp120-CD4, gp120-CCR5)
- Encode interface codons
- Identify mutation-tolerant vs conserved positions
"""
```

### Script 4: `glycan_tropism_correlation.py`
```python
"""
Correlate glycan patterns with coreceptor tropism
- Load 712 aligned gp120 sequences
- Map N-glycosylation sites
- Compare CCR5 vs CXCR4 glycan landscapes
- Identify differential shielding patterns
"""
```

### Script 5: `validate_on_hiv2.py`
```python
"""
Validate codon encoder on HIV-2 and SIV sequences
- Load HIV-2 and SIV sequences
- Extract codon usage
- Test encoder generalization
- Compare hyperbolic properties across lentiviruses
"""
```

---

## Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW DATA SOURCES                            │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│ HIV-data    │ HIV-1_Paper │ human_hiv_ppi│ V3_coreceptor│ gp120 │
│ (sequences) │ (mutations) │ (interactions)│ (tropism)   │(align)│
└──────┬──────┴──────┬──────┴──────┬───────┴──────┬──────┴───┬────┘
       │             │             │              │          │
       ▼             ▼             ▼              ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                              │
│  • Sequence → Codons                                            │
│  • Amino acid → Codon mapping                                   │
│  • Alignment parsing                                            │
│  • Mutation extraction                                          │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CODON ENCODER (3-adic)                         │
│  • Encode 64 codons → 16D hyperbolic space                     │
│  • Hierarchy correlation: -0.832                                │
│  • Cluster accuracy: 79.7%                                      │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ANALYSES                                    │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│ CTL Escape  │ Drug        │ PPI         │ Tropism     │ Glycan │
│ (expanded)  │ Resistance  │ Interface   │ Switching   │ Shield │
│             │ (Nigeria)   │ (new)       │ (new)       │(expand)│
└─────────────┴─────────────┴─────────────┴─────────────┴────────┘
```

---

## Priority Implementation Order

1. **High Priority (Immediate Value)**
   - `expand_escape_analysis.py` - V3 tropism analysis
   - `analyze_nigeria_resistance.py` - Real patient validation

2. **Medium Priority (New Capabilities)**
   - `ppi_interface_analysis.py` - Protein interactions
   - `glycan_tropism_correlation.py` - Enhanced glycan analysis

3. **Low Priority (Validation/Generalization)**
   - `validate_on_hiv2.py` - Cross-lentivirus validation
   - Epidemiological correlations

---

## Expected Impact

| Analysis | Current | With New Data |
|----------|---------|---------------|
| CTL Escape Mutations | 9 | 500+ |
| Drug Resistance Mutations | 18 | 200+ |
| Protein Interactions | 0 | 16,000+ |
| gp120 Sequences | 0 | 712 |
| V3 Tropism Examples | 0 | 2,935 |
| Sequence Diversity | 64 codons | 13,000+ sequences |

---

## Next Steps

1. Create data loading utilities for each dataset
2. Implement `expand_escape_analysis.py`
3. Validate with Nigeria patient data
4. Document findings in updated ANALYSIS_REPORT.md
