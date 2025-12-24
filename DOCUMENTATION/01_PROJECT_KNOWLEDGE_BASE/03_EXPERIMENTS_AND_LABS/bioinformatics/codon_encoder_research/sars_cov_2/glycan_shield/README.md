# SARS-CoV-2 RBD-ACE2 Handshake Analysis

**Doc-Type:** Project README · Version 1.0 · Updated 2025-12-19 · Author AI Whisperers

---

## Overview

This directory contains the complete analysis of SARS-CoV-2 spike protein RBD-ACE2 binding interface using the p-adic geometric framework. The goal is to identify **asymmetric therapeutic targets** that disrupt viral binding while preserving host protein function.

---

## Key Findings

### Validated Hypotheses

| Hypothesis | Prediction | AlphaFold3 Result | Status |
|:-----------|:-----------|:------------------|:-------|
| N439/N440 is critical handshake | Tightest convergence (0.147) | Mutations destabilize RBD | VALIDATED |
| ACE2 unaffected by RBD mutation | 0% host shift | pTM unchanged (0.91) | VALIDATED |
| RBD destabilized by phosphomimic | 20% viral shift | pTM -4.9%, 4% disorder | VALIDATED |
| Y449 is alternative target | High asymmetry | Highest PAE increase (+19%) | VALIDATED |

### Therapeutic Candidates

1. **N439D + N440D double phosphomimic** - 12.7% interface disruption
2. **Y449D phosphomimic** - 19.0% interface disruption
3. **Peptide inhibitor**: `Ac-VIAWNDNLDDKVGG-NH2`

---

## Directory Structure

```
glycan_shield/
├── README.md                          # This file
├── CONJECTURE_SPIKE_GLYCANS.md        # Initial hypothesis
├── ANALYSIS_RESULTS.md                # Glycan shield findings
├── HANDSHAKE_ANALYSIS_FINDINGS.md     # Detailed handshake results
├── ALPHAFOLD3_VALIDATION.md           # Structure validation report
│
├── 01_spike_sentinel_analysis.py      # Glycan site analysis
├── 02_handshake_interface_analysis.py # Interface mapping
├── 03_deep_handshake_sweep.py         # 19 modification types
├── 04_alphafold3_validation_jobs.py   # Generate AF3 inputs
│
├── spike_analysis_results.json        # Glycan analysis data
├── handshake_analysis_results.json    # Interface mapping data
├── deep_sweep_results.json            # Full modification scan data
│
├── alphafold3_jobs/                   # AlphaFold3 input files
│   ├── job1a_rbd_ace2_wildtype.json
│   ├── job1b_rbd_S439D_ace2.json
│   ├── job1c_rbd_S440D_ace2.json
│   ├── job1d_rbd_double_mutant_ace2.json
│   └── job2*_peptide_*.json
│
└── alphafold3_predictions/            # AlphaFold3 results
    └── folds_2025_12_19_07_07/
        ├── sarscov2_rbd_ace2_wildtype/
        ├── sarscov2_rbd_s439d_ace2/
        ├── sarscov2_rbd_s440d_ace2/
        ├── sarscov2_rbd_s439d_s440d_ace2/
        └── sarscov2_rbd_y449d_ace2/
```

---

## Analysis Pipeline

### Stage 1: Glycan Shield Sentinel Analysis

**Script**: `01_spike_sentinel_analysis.py`

Analyzed all 22 N-glycosylation sites on spike protein to determine if they function as immune shields (like HIV) or functional machinery.

**Result**: All sites showed <10% geometric shift - below Goldilocks zone. Spike glycans are **functional machinery** for host hijacking, not immune shields.

### Stage 2: Handshake Interface Mapping

**Script**: `02_handshake_interface_analysis.py`

Mapped 34 RBD contact residues × 31 ACE2 contact residues = 1,054 pairs. Computed hyperbolic distance between geometric centroids.

**Result**: Identified N439/N440 doublet as tightest handshake (distance 0.147).

### Stage 3: Deep Modification Sweep

**Script**: `03_deep_handshake_sweep.py`

Tested 19 modification types across all handshake pairs:
- Phosphomimics (S→D, T→D, Y→D)
- Deglycosylation (N→Q)
- Citrullination (R→Q)
- Acetylation (K→Q)
- Oxidation (M→Q, C→S, W→F)
- And 10 more...

**Result**: S→D phosphomimics show best asymmetric disruption profile.

### Stage 4: AlphaFold3 Validation

**Script**: `04_alphafold3_validation_jobs.py`

Generated structure predictions for:
- Wildtype RBD-ACE2
- Single mutants (N439D, N440D, Y449D)
- Double mutant (N439D + N440D)

**Result**: All predictions validated the asymmetric perturbation hypothesis.

---

## How to Run

### Prerequisites

```bash
# From repository root
pip install numpy scipy torch

# Ensure hyperbolic_utils is available
export PYTHONPATH=$PYTHONPATH:research/bioinformatics/rheumatoid_arthritis/scripts
```

### Run Analysis

```bash
cd research/bioinformatics/sars_cov_2/glycan_shield

# Stage 1: Glycan analysis
python 01_spike_sentinel_analysis.py

# Stage 2: Handshake mapping
python 02_handshake_interface_analysis.py

# Stage 3: Deep sweep
python 03_deep_handshake_sweep.py

# Stage 4: Generate AlphaFold3 jobs
python 04_alphafold3_validation_jobs.py
```

---

## Data Sources

### Structural Data

- **PDB 6M0J**: SARS-CoV-2 RBD-ACE2 crystal structure
- **UniProt P0DTC2**: Spike protein sequence
- **UniProt Q9BYF1**: ACE2 sequence

### Encoder

- **3-adic codon encoder** (V5.11.3)
- Location: `research/genetic_code/models/codon_encoder_3adic.pt`

---

## Related Documentation

### P-adic Framework

- Theory: `research/p-adic-genomics/theory/MATHEMATICAL_FOUNDATIONS.md`
- Discoveries: `research/p-adic-genomics/DISCOVERIES.md`
- Validation: `research/p-adic-genomics/validations/SARS_COV2_CASE_STUDY.md`

### Other Applications

- Rheumatoid Arthritis: `research/bioinformatics/rheumatoid_arthritis/`
- Alzheimer's Tau: `research/bioinformatics/neurodegeneration/alzheimers/`
- HIV Glycan Shield: `research/bioinformatics/hiv/glycan_shield/`

---

## Key Metrics

### Geometric Framework

| Metric | Description | Threshold |
|:-------|:------------|:----------|
| Handshake Distance | Hyperbolic distance between centroids | <0.20 = convergent |
| Centroid Shift | Modification-induced geometry change | >15% = significant |
| Asymmetry Score | Viral shift - Host shift | >10% = therapeutic |

### AlphaFold3 Validation

| Metric | Description | Interpretation |
|:-------|:------------|:---------------|
| pTM | Predicted TM-score | >0.7 = high confidence |
| iPTM | Interface pTM | <0.5 = uncertain complex |
| PAE | Predicted Aligned Error | Lower = better |
| Disorder | Predicted disorder fraction | Higher = destabilized |

---

## Citation

If using this analysis, please cite:

```
P-adic Geometric Framework for Viral-Host Interface Analysis
AI Whisperers, 2025
https://github.com/[repo]/research/p-adic-genomics
```

---

## Changelog

| Date | Version | Description |
|:-----|:--------|:------------|
| 2025-12-19 | 1.0 | Complete analysis with AlphaFold3 validation |
| 2025-12-18 | 0.1 | Initial glycan shield analysis |
