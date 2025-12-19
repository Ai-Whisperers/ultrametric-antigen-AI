# P-adic Genomics Framework: Cross-Reference Index

**Doc-Type:** Navigation Index · Version 2.0 · Updated 2025-12-19 · Author AI Whisperers

---

## Quick Navigation

### Core Documentation

| Document | Location | Description |
|:---------|:---------|:------------|
| **DISCOVERIES.md** | `p-adic-genomics/` | Consolidated findings across all applications |
| **VALIDATION_PROOF.md** | `p-adic-genomics/` | **Mathematical validation of 3-adic framework** |
| **DISEASE_APPLICATION_ROADMAP.md** | `p-adic-genomics/` | Future application planning |
| **PROPOSAL_ACCESSIBLE_THERAPEUTICS.md** | `p-adic-genomics/` | Therapeutic development strategy |
| README.md | `p-adic-genomics/` | Project overview |

### Theory

| Document | Location | Description |
|:---------|:---------|:------------|
| MATHEMATICAL_FOUNDATIONS.md | `p-adic-genomics/theory/` | P-adic number theory for genomics |
| PTM_MODEL.md | `p-adic-genomics/theory/` | Goldilocks zone formalization |

### Validation Case Studies

| Disease | Location | Status |
|:--------|:---------|:-------|
| **Rheumatoid Arthritis** | `p-adic-genomics/validations/RA_CASE_STUDY.md` | VALIDATED |
| **SARS-CoV-2** | `p-adic-genomics/validations/SARS_COV2_CASE_STUDY.md` | VALIDATED (AlphaFold3) |
| **Alzheimer's Tau** | `neurodegeneration/alzheimers/FINDINGS.md` | VALIDATED (additive model) |

### Framework Validation

| Test | Script | Result |
|:-----|:-------|:-------|
| **Ultrametric Property** | `validations/padic_biology_validation.py` | 100% valid |
| **Synonymous Clustering** | `validations/encoder_cross_validation.py` | 2.59x separation |
| **Wobble Effect** | `validations/encoder_cross_validation.py` | 35% smaller |
| **Phosphomimic Geometry** | `validations/encoder_cross_validation.py` | 6.74 shift |

---

## Application Directories

### Rheumatoid Arthritis (Autoimmunity)

**Location:** `bioinformatics/rheumatoid_arthritis/`

| File | Description |
|:-----|:------------|
| `scripts/hyperbolic_utils.py` | Core encoding functions (shared) |
| `scripts/01-17*.py` | Analysis pipeline |
| `discoveries/DISCOVERY_HLA_RA_PREDICTION.md` | HLA binding predictions |
| `CONSOLIDATED_FINDINGS.md` | Summary of RA analysis |

### SARS-CoV-2 (Viral-Host Interaction)

**Location:** `bioinformatics/sars_cov_2/glycan_shield/`

| File | Description |
|:-----|:------------|
| `README.md` | Project overview |
| `01_spike_sentinel_analysis.py` | Glycan shield analysis |
| `02_handshake_interface_analysis.py` | Interface mapping |
| `03_deep_handshake_sweep.py` | 19 modification types |
| `04_alphafold3_validation_jobs.py` | AlphaFold3 job generator |
| `ALPHAFOLD3_VALIDATION.md` | Structure validation report |
| `HANDSHAKE_ANALYSIS_FINDINGS.md` | Detailed results |
| `alphafold3_validation_summary.json` | Machine-readable validation data |
| `alphafold3_predictions/` | AlphaFold3 structure files |

### HIV (Glycan Shield)

**Location:** `bioinformatics/hiv/glycan_shield/`

| File | Description |
|:-----|:------------|
| Analysis scripts | Glycan sentinel analysis |
| AlphaFold3 inputs | Structure prediction jobs |

### Alzheimer's Disease (Neurodegeneration)

**Location:** `bioinformatics/neurodegeneration/alzheimers/`

| File | Description |
|:-----|:------------|
| `DESIGN_TAU_ANALYSIS.md` | Design document |
| `FINDINGS.md` | Key discoveries |
| `01_tau_phospho_sweep.py` | Single-site analysis |
| `02_tau_mtbr_interface.py` | Tau-tubulin handshakes |
| `03_tau_vae_trajectory.py` | Phosphorylation trajectory |
| `data/tau_phospho_database.py` | Phospho-site database |
| `results/` | Analysis outputs |

---

## Key Concepts Cross-Reference

### Goldilocks Zone (15-30% Shift)

| Application | Interpretation | Files |
|:------------|:---------------|:------|
| RA | Immunogenic threshold | `RA_CASE_STUDY.md` |
| HIV | Inverse (removal exposes epitopes) | `DISCOVERIES.md` |
| Tau | Dysfunction transition zone | `FINDINGS.md` |

### Handshake Analysis

| Application | Proteins | Files |
|:------------|:---------|:------|
| SARS-CoV-2 | RBD ↔ ACE2 | `HANDSHAKE_ANALYSIS_FINDINGS.md` |
| Alzheimer's | Tau ↔ Tubulin | `02_tau_mtbr_interface.py` |

### Asymmetric Perturbation

| Application | Goal | Validation |
|:------------|:-----|:-----------|
| SARS-CoV-2 | Disrupt viral, preserve host | AlphaFold3 pTM unchanged for ACE2 |
| Tau | Understand dysfunction | Interface distance metrics |

---

## AlphaFold3 Validations

| Application | Predictions | Location |
|:------------|:------------|:---------|
| HIV | Glycan shield | `hiv/glycan_shield/alphafold3_inputs/` |
| SARS-CoV-2 | RBD mutations | `sars_cov_2/glycan_shield/alphafold3_predictions/` |
| RA | HLA-peptide | `rheumatoid_arthritis/results/alphafold3/` |

---

## Encoder Resources

| Resource | Location | Description |
|:---------|:---------|:------------|
| Trained encoder | `genetic_code/data/codon_encoder_3adic.pt` | 3-adic weights |
| Encoder utils | `bioinformatics/rheumatoid_arthritis/scripts/hyperbolic_utils.py` | Shared functions |
| VAE model | `src/models/ternary_vae.py` | V5.11.3 architecture |
| **Validation scripts** | `p-adic-genomics/validations/` | Cross-validation framework |

---

## Validation Script Details

| Script | Purpose | Key Results |
|:-------|:--------|:------------|
| `padic_biology_validation.py` | Test ultrametric property across 7 encoding schemes | 100% validity |
| `encoder_cross_validation.py` | Validate trained encoder against genetic code | 2/3 tests pass |

**Encoding Schemes Tested:**
1. Amino Acid Chemistry (hydrophobic/polar/charged)
2. Amino Acid Size (small/medium/large)
3. Secondary Structure Propensity (helix/sheet/coil)
4. Nucleotide Chemistry (purine/pyrimidine/other)
5. Codon Position (1st/2nd/3rd)
6. Methylation State (unmet/5mC/5hmC)
7. Phosphorylation State (unphospho/phospho/mimic)

---

## Changelog

| Date | Version | Description |
|:-----|:--------|:------------|
| 2025-12-19 | 2.0 | Added VALIDATION_PROOF.md and validation scripts |
| 2025-12-19 | 1.0 | Initial index with SARS-CoV-2 and Tau applications |
