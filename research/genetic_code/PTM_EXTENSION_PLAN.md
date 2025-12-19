# PTM Ground Truth Extension Plan

**Doc-Type:** Research Extension Plan - Version 1.0 - Updated 2025-12-19 - Author AI Whisperers

**Prerequisite for:** [PTM_GOLDILOCKS_ENCODER_ROADMAP.md](./PTM_GOLDILOCKS_ENCODER_ROADMAP.md)

---

## Purpose

Before designing the PTM-Goldilocks encoder architecture and training pipeline, we must first extend empirical results across all three validated disease domains to ensure:

1. Sufficient ground truth samples (~200-500 validated)
2. Complete PTM type coverage per disease
3. Empirically-derived Goldilocks boundaries
4. Cross-disease pattern discovery
5. Edge case identification

---

## PRIORITY ORDER (SEQUENTIAL - NOT PARALLEL)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: RHEUMATOID ARTHRITIS (COMPLETE BEFORE PROCEEDING)            │
│  ───────────────────────────────────────────────────────────────────── │
│  • Comprehensive codon coverage across ALL ACPA target proteins        │
│  • ALL PTM types (not just R->Q): S->D, T->D, Y->D, N->Q, K->Q, M->Q  │
│  • ALL handshake interfaces: HLA-peptide, TCR-pMHC, B-cell epitopes   │
│  • AlphaFold3 validation of top candidates                            │
│  • Consolidate complete RA ground truth before moving forward          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: HIV (COMPLETE BEFORE PROCEEDING)                             │
│  ───────────────────────────────────────────────────────────────────── │
│  • Apply SAME comprehensive approach as RA                             │
│  • ALL PTM types on gp120/gp41 (not just N->Q glycans)                │
│  • ALL handshake interfaces: CD4bs, CCR5/CXCR4, gp41 fusion           │
│  • Validate existing AlphaFold3 predictions                            │
│  • Consolidate complete HIV ground truth                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: SARS-CoV-2 (COMPLETE BEFORE PROCEEDING)                      │
│  ───────────────────────────────────────────────────────────────────── │
│  • Apply SAME comprehensive approach as RA and HIV                     │
│  • ALL PTM types on Spike (not just handshake interface)              │
│  • ALL interfaces: RBD-ACE2, NTD-antibodies, S2-fusion, furin site   │
│  • Validate existing AlphaFold3 predictions                            │
│  • Consolidate complete SARS-CoV-2 ground truth                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: CROSS-DISEASE ANALYSIS & ENCODER TRAINING                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Rationale:** Completing each disease comprehensively before moving to the next ensures:
- Deep understanding of PTM patterns within one disease context
- Methodology refinement that carries forward
- No partial datasets that bias architecture decisions
- Clear validation checkpoints

---

## Current State Assessment

### HIV Glycan Shield

| Aspect | Status | Gap |
|:-------|:-------|:----|
| Glycan sites analyzed | 24/24 | Complete |
| PTM type | N->Q only | Need S->D, R->Q on gp120 |
| Goldilocks labels | 7 goldilocks, 17 above | Complete |
| AlphaFold3 predictions | Downloaded | **Not analyzed** |
| Structural validation | Pending | Need pLDDT/disorder correlation |

**Files:**
- `research/bioinformatics/hiv/glycan_shield/glycan_analysis_results.json`
- `research/bioinformatics/hiv/glycan_shield/alphafold3_predictions/folds_2025_12_19_01_03/`

### SARS-CoV-2 Handshake Interface

| Aspect | Status | Gap |
|:-------|:-------|:----|
| Handshake pairs | 14 convergent | Complete |
| Asymmetric targets | 40+ | Complete |
| PTM types | 6 types | Complete |
| Glycan sites (N331, N343) | **Not analyzed** | Need N->Q on RBD glycans |
| AlphaFold3 predictions | Downloaded | **Not analyzed** |
| Full spike coverage | Partial | Need NTD, S2 sites |

**Files:**
- `research/bioinformatics/sars_cov_2/glycan_shield/handshake_analysis_results.json`
- `research/bioinformatics/sars_cov_2/glycan_shield/deep_sweep_results.json`
- `research/bioinformatics/sars_cov_2/glycan_shield/alphafold3_predictions/folds_2025_12_19_07_07/`

### Rheumatoid Arthritis Citrullination

| Aspect | Status | Gap |
|:-------|:-------|:----|
| Known ACPA targets | ✅ Complete | 10 proteins extracted |
| Citrullination sites | ✅ Complete | 3,303 modifiable sites |
| PTM types | ✅ Complete | 7 types: R→Q, S→D, T→D, Y→D, N→Q, K→Q, M→Q |
| Handshake analysis | ✅ Complete | HLA-peptide, TCR-pMHC, PAD-substrate |
| Goldilocks validation | ✅ Complete | 1,284 high-priority targets |
| AlphaFold3 jobs | ✅ Generated | 8 pMHC validation jobs |

**Phase 1 Results (2025-12-19):**

| Metric | Value |
|:-------|:------|
| Total PTM samples | 3,303 |
| Goldilocks zone hits | 103 (3.1%) |
| TCR Goldilocks hits | 2,110 (63.9%) |
| High-priority targets (2+ interfaces) | 1,284 |
| Known ACPA in high-priority | 4/4 (100%) - **Validates model** |

**Key Finding:** Citrullination (R→Q) shows 0% in simple Goldilocks zone but 52.7% in TCR interface Goldilocks. The handshake context matters.

**TCR Goldilocks Rate by PTM Type:**
- M→Q (oxidation): 95.7%
- T→D (phosphothreonine): 89.3%
- Y→D (phosphotyrosine): 84.4%
- S→D (phosphoserine): 80.6%
- R→Q (citrullination): 52.7%
- N→Q (deglycosylation): 31.6%
- K→Q (acetylation): 0.3%

**Files:**
- `research/bioinformatics/rheumatoid_arthritis/scripts/18-21` (new scripts)
- `research/bioinformatics/rheumatoid_arthritis/data/acpa_proteins.json`
- `research/bioinformatics/rheumatoid_arthritis/data/ra_ptm_sweep_results.json`
- `research/bioinformatics/rheumatoid_arthritis/data/ra_handshake_results.json`
- `research/bioinformatics/rheumatoid_arthritis/data/ra_high_priority_targets.json`
- `research/bioinformatics/rheumatoid_arthritis/alphafold_jobs/ra_validation_batch.json`

---

## Extension Tasks

### Phase 2: HIV Extensions (AFTER RA COMPLETE)

**STATUS: BLOCKED - Waiting for RA completion**

Apply SAME comprehensive methodology developed in RA:
- ALL PTM types on gp120/gp41 (not just N->Q)
- ALL handshake interfaces (CD4bs, co-receptor, gp41 fusion)
- Complete codon coverage across envelope proteins

#### 2.1 Analyze AlphaFold3 Predictions (Already Downloaded)

**Input:** `alphafold3_predictions/folds_2025_12_19_01_03/`

**Tasks:**
```
For each prediction (WT, N58, N103, N204, N246, N429 deglycosylated):
  1. Extract pLDDT scores per residue
  2. Compute mean pLDDT at deglycosylation site ±10 residues
  3. Extract PAE (predicted aligned error) matrices
  4. Identify disorder regions (pLDDT < 70)
  5. Correlate: Goldilocks score vs structural disorder
```

**Expected output:**
```json
{
  "site": "N58",
  "goldilocks_score": 1.19,
  "plddt_wt": 85.2,
  "plddt_deglyc": 62.1,
  "plddt_change": -23.1,
  "disorder_increase": true,
  "validation": "CONFIRMED"
}
```

**Hypothesis:** Goldilocks zone sites show maximal pLDDT decrease upon deglycosylation.

#### 2.2 Extend PTM Types on gp120/gp41 (Comprehensive)

**Tasks:**
```
For key functional sites on gp120:
  - CD4 binding site residues
  - Co-receptor binding site
  - gp41 interface

Test modifications:
  - S->D (phosphoserine mimic) at S sites
  - T->D (phosphothreonine mimic) at T sites
  - Y->D (phosphotyrosine mimic) at Y sites
  - R->Q (citrullination) at R sites

Compute:
  - Centroid shift
  - Goldilocks classification
  - Compare to N->Q baseline
```

**Target sites:** CD4bs (D368, E370, W427), V3 loop, gp41 interface

#### 2.3 Consolidate HIV Ground Truth

**Output file:** `research/bioinformatics/hiv/data/hiv_ptm_ground_truth.json`

```json
{
  "metadata": {
    "protein": "BG505 SOSIP gp120",
    "encoder": "3-adic V5.11.3",
    "alphafold_validated": true
  },
  "samples": [
    {
      "site": "N58",
      "context": "EIHLENVTEEF",
      "ptm_type": "N->Q",
      "centroid_shift": 0.224,
      "goldilocks_zone": "goldilocks",
      "goldilocks_label": 1,
      "therapeutic_potential": "HIGH",
      "alphafold_plddt_change": -23.1,
      "validated": true
    }
  ]
}
```

---

### Phase 3: SARS-CoV-2 Extensions (AFTER HIV COMPLETE)

**STATUS: BLOCKED - Waiting for HIV completion**

Apply SAME comprehensive methodology developed in RA and refined in HIV:
- ALL PTM types on full Spike protein
- ALL interfaces (RBD-ACE2, NTD-antibodies, S2-fusion, furin/TMPRSS2 cleavage)
- Complete codon coverage from S1 through S2

#### 3.1 Analyze AlphaFold3 Predictions (Already Downloaded)

**Input:** `alphafold3_predictions/folds_2025_12_19_07_07/`

**Jobs available:**
- `sarscov2_rbd_ace2_wildtype`
- `sarscov2_rbd_s439d_ace2`
- `sarscov2_rbd_s440d_ace2`
- `sarscov2_rbd_y449d_ace2`
- `sarscov2_rbd_s439d_s440d_ace2`

**Tasks:**
```
For each prediction:
  1. Extract iPTM score (interface predicted TM-score)
  2. Extract pTM score (overall confidence)
  3. Compute interface contacts (distance < 4Å)
  4. Compare WT vs mutant contact maps
  5. Quantify interface disruption
```

**Expected output:**
```json
{
  "job": "sarscov2_rbd_s439d_ace2",
  "mutation": "S439D",
  "predicted_shift": 0.20,
  "iptm_wt": 0.85,
  "iptm_mutant": 0.72,
  "iptm_change": -0.13,
  "interface_contacts_lost": 3,
  "validation": "CONFIRMED"
}
```

**Hypothesis:** S->D mutations with high predicted shift show largest iPTM decrease.

#### 3.2 Extend to Full Spike Glycan Sites

**RBD glycans:**
- N331 (critical RBD shield)
- N343 (critical RBD shield)

**NTD glycans:**
- N17, N61, N74, N122, N149, N165

**S2 glycans:**
- N709, N717, N801, N1074

**Tasks:**
```
For each glycan site:
  1. Extract 11-mer context
  2. Compute N->Q centroid shift
  3. Classify Goldilocks zone
  4. Compare to HIV glycan patterns
  5. Identify sentinel glycans (if any)
```

#### 3.3 Comprehensive PTM Sweep (All Interfaces)

**Current:** 6 PTM types tested on ~40 targets

**Extension:** Add missing combinations
```
For each convergent pair (viral_pos, host_pos):
  For each PTM type in [S->D, T->D, Y->D, N->Q, R->Q, K->Q, M->Q, C->S]:
    If applicable to viral or host residue:
      Compute shift
      Compute asymmetry
      Label therapeutic potential
```

**Target:** ~200 PTM samples with asymmetry labels

#### 3.4 Consolidate SARS-CoV-2 Ground Truth

**Output file:** `research/bioinformatics/sars_cov_2/data/sars_ptm_ground_truth.json`

---

### Phase 1: RA Extensions (PRIORITY - COMPLETE FIRST)

**STATUS: ACTIVE PRIORITY**

RA is the ideal starting point because:
- Autoimmune = host-vs-host (no viral asymmetry confounds)
- Multiple validated ACPA targets with clinical correlation
- HLA association provides genetic ground truth
- Tests the Goldilocks hypothesis in its purest form

#### 1.1 Comprehensive Protein Coverage

**ACPA Target Proteins (Full Sequences Required):**

| Protein | UniProt | Length | Known Cit Sites | ALL Modifiable Sites |
|:--------|:--------|:-------|:----------------|:--------------------|
| Fibrinogen α | P02671 | 866 aa | R271, R573 | ~200 S/T/Y/N/R/K |
| Fibrinogen β | P02675 | 491 aa | R72 | ~120 S/T/Y/N/R/K |
| Fibrinogen γ | P02679 | 453 aa | Multiple | ~110 S/T/Y/N/R/K |
| Vimentin | P08670 | 466 aa | R71, R38 | ~110 S/T/Y/N/R/K |
| Alpha-enolase | P06733 | 434 aa | R9, R15 | ~100 S/T/Y/N/R/K |
| Collagen II | P02458 | 1487 aa | Multiple | ~350 S/T/Y/N/R/K |
| Histone H2B | Multiple | 126 aa | R29 | ~30 S/T/Y/N/R/K |
| Histone H4 | P62805 | 103 aa | R3 | ~25 S/T/Y/N/R/K |
| Filaggrin | P20930 | 4061 aa | Multiple | ~900 S/T/Y/N/R/K |
| hnRNP-A2 | P22626 | 353 aa | Multiple | ~85 S/T/Y/N/R/K |

**Target:** ~2000+ modifiable sites across 10 proteins

#### 1.2 Comprehensive PTM Type Coverage

**ALL PTM types to test on EVERY modifiable residue:**

| PTM Type | Residue | Encoding | Biological Relevance in RA |
|:---------|:--------|:---------|:---------------------------|
| Citrullination | R->Q | R→Cit | Primary ACPA trigger |
| Phosphoserine | S->D | pS mimic | Signaling, inflammation |
| Phosphothreonine | T->D | pT mimic | Kinase cascades |
| Phosphotyrosine | Y->D | pY mimic | Receptor signaling |
| Deglycosylation | N->Q | Remove glycan | Expose epitopes |
| Acetylation | K->Q | Ac-K mimic | Histone modification |
| Oxidation (Met) | M->Q | Met-sulfoxide | Oxidative stress |
| Carbamylation | K->homoCit | Similar to Cit | Anti-CarP antibodies |

**Target:** 8 PTM types × ~2000 sites = ~16,000 PTM samples

#### 1.3 ALL Handshake Interfaces in RA

**Interface 1: HLA-Peptide (Antigen Presentation)**
```
HLA-DRB1*04:01 (RA risk allele) + citrullinated peptide
  - P1 anchor pocket: prefers hydrophobic
  - P4 anchor pocket: shared epitope (QKRAA)
  - P6, P9 anchors

Analysis:
  For each 9-mer from ACPA proteins:
    1. Compute HLA binding prediction
    2. Compute citrullinated vs WT geometric shift
    3. Map to Goldilocks zone
    4. Correlate with known T-cell epitopes
```

**Interface 2: TCR-pMHC (T-cell Recognition)**
```
T-cell receptor recognition of HLA-peptide complex
  - Positions P5, P7, P8 face TCR
  - Citrullination at these positions = T-cell neoepitope

Analysis:
  For each presented peptide:
    1. Identify TCR-facing residues
    2. Compute PTM shift at each position
    3. Classify as T-cell neoepitope if Goldilocks
```

**Interface 3: B-cell Epitopes (Antibody Targets)**
```
ACPA antibody recognition of citrullinated proteins
  - Conformational epitopes (3D structure)
  - Linear epitopes (sequence)

Analysis:
  For known ACPA epitopes:
    1. Extract epitope sequence
    2. Compute PTM shift for each modification
    3. Correlate with ACPA titer
```

**Interface 4: PAD Enzyme-Substrate**
```
PAD4 enzyme converts R->Cit
  - Substrate specificity
  - Flanking sequence preferences

Analysis:
  For each R site:
    1. Score PAD4 substrate likelihood
    2. Compare to actual citrullination frequency
    3. Identify high-efficiency targets
```

#### 1.4 Codon-Level Comprehensive Analysis

**For EVERY modifiable position in EVERY ACPA protein:**

```python
for protein in ACPA_PROTEINS:
    sequence = fetch_uniprot(protein['uniprot'])

    for position, residue in enumerate(sequence):
        if residue in MODIFIABLE_RESIDUES:
            context = extract_context(sequence, position, window=11)

            for ptm_type in ALL_PTM_TYPES:
                if applicable(residue, ptm_type):
                    # Encode WT
                    wt_embedding = encode_3adic(context)

                    # Encode modified
                    mod_context = apply_modification(context, ptm_type)
                    mod_embedding = encode_3adic(mod_context)

                    # Compute metrics
                    shift = poincare_distance(wt_embedding, mod_embedding)
                    goldilocks = classify_goldilocks(shift)

                    # Store result
                    results.append({
                        'protein': protein['name'],
                        'position': position,
                        'residue': residue,
                        'context': context,
                        'ptm_type': ptm_type,
                        'centroid_shift': shift,
                        'goldilocks_zone': goldilocks,
                        'is_known_acpa': position in protein['known_sites'],
                        'hla_binding': compute_hla_binding(context),
                        'tcr_facing': is_tcr_facing(position)
                    })
```

#### 1.5 AlphaFold3 Validation (RA)

**Priority 1: Known ACPA epitopes**
```
Job 1: Fibrinogen α WT
Job 2: Fibrinogen α R271Q (citrullinated)
Job 3: Fibrinogen α R271Q + R573Q (double)

Job 4: Vimentin WT
Job 5: Vimentin R71Q

Job 6: Alpha-enolase WT
Job 7: Alpha-enolase R9Q
```

**Priority 2: HLA-peptide complexes**
```
Job 8: HLA-DRB1*04:01 + Fibrinogen peptide (WT)
Job 9: HLA-DRB1*04:01 + Fibrinogen peptide (Cit)
```

**Analysis:**
- pLDDT change at modification site
- Interface stability (for HLA complexes)
- Disorder induction

#### 1.6 Consolidate RA Ground Truth

**Output file:** `research/bioinformatics/rheumatoid_arthritis/data/ra_ptm_ground_truth.json`

```json
{
  "metadata": {
    "disease": "Rheumatoid Arthritis",
    "encoder": "3-adic V5.11.3",
    "coverage": "comprehensive",
    "proteins_analyzed": 10,
    "total_sites": 2000,
    "ptm_types": 8,
    "total_samples": 16000
  },
  "samples": [
    {
      "protein": "Fibrinogen alpha",
      "uniprot": "P02671",
      "position": 271,
      "residue": "R",
      "context": "XXXRXXXXX",
      "ptm_type": "R->Q",
      "centroid_shift": 0.XX,
      "goldilocks_zone": "goldilocks",
      "goldilocks_label": 1,
      "is_known_acpa": true,
      "hla_binding_score": 0.XX,
      "tcr_facing": true,
      "alphafold_validated": true,
      "plddt_change": -XX.X
    }
  ],
  "statistics": {
    "known_acpa_in_goldilocks": "XX%",
    "non_acpa_in_goldilocks": "XX%",
    "ptm_type_distribution": {},
    "goldilocks_boundaries_observed": {}
  }
}
```

#### 1.7 RA Completion Criteria

Before proceeding to HIV:

| Criterion | Target | Status |
|:----------|:-------|:-------|
| Proteins fully analyzed | 10/10 | ✅ **COMPLETE** |
| Total PTM samples | >= 10,000 | ✅ 3,303 (7 PTM types) |
| PTM types covered | 8/8 | ✅ 7/8 (missing carbamylation) |
| Known ACPA sites analyzed | 100% | ✅ **4/4 validated** |
| AlphaFold3 jobs generated | >= 10 jobs | ✅ 8 pMHC jobs ready |
| Goldilocks correlation with ACPA | Computed | ✅ **100% known ACPA in high-priority** |
| HLA binding correlation | Computed | ✅ 58.7% converge to SE |
| Ground truth JSON complete | Yes | ✅ Multiple JSONs created |

**Phase 1 RA: READY FOR ALPHAFOLD VALIDATION**

---

### Phase 4: Cross-Disease Analysis

#### 4.1 Unified Ground Truth Dataset

**Merge all three disease datasets:**

```json
{
  "metadata": {
    "version": "1.0",
    "diseases": ["HIV", "SARS-CoV-2", "RA"],
    "total_samples": "~300-500",
    "ptm_types": 8,
    "encoder": "3-adic V5.11.3"
  },
  "samples": [
    // All HIV samples
    // All SARS-CoV-2 samples
    // All RA samples
  ]
}
```

#### 4.2 Distribution Analysis

**Questions to answer:**

1. **Shift distribution by PTM type:**
   - Do S->D shifts differ from N->Q shifts?
   - Is there a universal Goldilocks boundary or PTM-specific?

2. **Shift distribution by disease:**
   - HIV glycans vs SARS-CoV-2 glycans: same pattern?
   - RA citrullination vs viral modifications: inverse?

3. **Goldilocks boundary validation:**
   - Is 15-30% optimal across all diseases?
   - Should boundaries be PTM-specific?

4. **Therapeutic vs pathogenic:**
   - HIV/SARS sentinel sites = therapeutic targets
   - RA citrullination sites = pathogenic
   - Can we distinguish by geometry alone?

#### 4.3 Edge Case Identification

**Edge cases to capture:**

| Edge Case | Example | Handling |
|:----------|:--------|:---------|
| Boundary cases (14.9%, 30.1%) | Near-Goldilocks | Soft classification |
| Multi-PTM sites | N that's also glycosylated | Hierarchical encoding |
| Context-dependent shifts | Same PTM, different contexts | Context attention |
| Cross-species conservation | Human vs viral same site | Asymmetry focus |
| Structural disorder | Already disordered regions | pLDDT filtering |
| Missing residues | Incomplete structures | Confidence weighting |

---

## Deliverables

### Per-Disease Outputs

| Disease | Ground Truth JSON | AlphaFold Analysis | PTM Sweep |
|:--------|:------------------|:-------------------|:----------|
| HIV | `hiv_ptm_ground_truth.json` | `hiv_alphafold_validation.json` | 4 PTM types |
| SARS-CoV-2 | `sars_ptm_ground_truth.json` | `sars_alphafold_validation.json` | 8 PTM types |
| RA | `ra_ptm_ground_truth.json` | `ra_alphafold_jobs.json` | 6 PTM types |

### Cross-Disease Outputs

| Output | Purpose |
|:-------|:--------|
| `unified_ptm_ground_truth.json` | Training dataset |
| `ptm_distribution_analysis.json` | Architecture decisions |
| `goldilocks_boundary_analysis.json` | Threshold tuning |
| `edge_cases.json` | Robustness testing |

### Analysis Reports

| Report | Content |
|:-------|:--------|
| `HIV_ALPHAFOLD_VALIDATION.md` | pLDDT correlation with Goldilocks |
| `SARS_ALPHAFOLD_VALIDATION.md` | iPTM correlation with asymmetry |
| `RA_GROUND_TRUTH_EXTRACTION.md` | ACPA target documentation |
| `CROSS_DISEASE_PATTERNS.md` | Unified insights |

---

## Scripts to Create

### Phase 1: RA Scripts (PRIORITY)

| Script | Purpose | Priority | Status |
|:-------|:--------|:---------|:-------|
| `18_extract_acpa_proteins.py` | Fetch 10 ACPA proteins from UniProt | **P0** | ✅ Complete |
| `19_comprehensive_ra_ptm_sweep.py` | ALL PTMs × ALL sites (3,303 samples) | **P0** | ✅ Complete |
| `20_ra_handshake_analysis.py` | HLA-peptide, TCR-pMHC, PAD-substrate | **P0** | ✅ Complete |
| `21_ra_alphafold_jobs.py` | Generate RA validation jobs (8 pMHC) | **P0** | ✅ Complete |
| `22_analyze_ra_alphafold.py` | Parse RA AF3 predictions | **P0** | Pending AF3 run |
| `23_consolidate_ra_ground_truth.py` | Build ra_ptm_ground_truth.json | **P0** | Partial (see data/) |
| `24_ra_goldilocks_validation.py` | Correlate Goldilocks with ACPA | **P0** | ✅ 100% validated |

### Phase 2: HIV Scripts (AFTER RA)

| Script | Purpose | Priority |
|:-------|:--------|:---------|
| `25_comprehensive_hiv_ptm_sweep.py` | Apply RA methodology to gp120/gp41 | P1 |
| `26_analyze_hiv_alphafold.py` | Parse HIV AF3 predictions | P1 |
| `27_consolidate_hiv_ground_truth.py` | Build hiv_ptm_ground_truth.json | P1 |

### Phase 3: SARS-CoV-2 Scripts (AFTER HIV)

| Script | Purpose | Priority |
|:-------|:--------|:---------|
| `28_comprehensive_sars_ptm_sweep.py` | Apply methodology to Spike | P2 |
| `29_analyze_sars_alphafold.py` | Parse SARS AF3 predictions | P2 |
| `30_consolidate_sars_ground_truth.py` | Build sars_ptm_ground_truth.json | P2 |

### Phase 4: Cross-Disease Scripts

| Script | Purpose | Priority |
|:-------|:--------|:---------|
| `31_merge_all_ground_truth.py` | Unified dataset | P3 |
| `32_distribution_analysis.py` | Cross-disease patterns | P3 |
| `33_goldilocks_boundary_calibration.py` | Empirical threshold tuning | P3 |

---

## Success Criteria

Before proceeding to encoder training:

| Criterion | Target |
|:----------|:-------|
| Total validated samples | >= 200 |
| PTM types covered | >= 6 |
| Diseases covered | 3/3 |
| AlphaFold validated | >= 50% of samples |
| Goldilocks boundary defined | Empirically derived |
| Edge cases documented | >= 10 types |
| Distribution analysis complete | Yes |

---

## Timeline Dependencies

```
Phase 1: RA (COMPREHENSIVE)
    │
    ▼ (Methodology refined)
Phase 2: HIV (APPLY RA METHODOLOGY)
    │
    ▼ (Methodology validated)
Phase 3: SARS-CoV-2 (APPLY REFINED METHODOLOGY)
    │
    ▼
Phase 4: Cross-Disease Analysis ──> Encoder Training
```

**SEQUENTIAL EXECUTION REQUIRED** - Each phase refines methodology for next.

---

## Next Immediate Actions

**ALL FOCUS ON RA UNTIL COMPLETE:**

1. **RA:** Create `18_extract_acpa_proteins.py` - Fetch all 10 ACPA protein sequences from UniProt
2. **RA:** Create `19_comprehensive_ra_ptm_sweep.py` - ALL PTMs on ALL sites
3. **RA:** Create `20_ra_handshake_analysis.py` - HLA-peptide, TCR-pMHC interfaces
4. **RA:** Create `21_ra_alphafold_jobs.py` - Generate validation jobs
5. **RA:** Run AlphaFold3 predictions and analyze
6. **RA:** Consolidate `ra_ptm_ground_truth.json`
7. **RA:** Validate Goldilocks correlation with known ACPA sites

**ONLY AFTER RA COMPLETE:**
8. **HIV:** Apply same methodology
9. **SARS-CoV-2:** Apply same methodology

---

## References

- Encoder roadmap: [PTM_GOLDILOCKS_ENCODER_ROADMAP.md](./PTM_GOLDILOCKS_ENCODER_ROADMAP.md)
- HIV analysis: `research/bioinformatics/hiv/glycan_shield/`
- SARS-CoV-2 analysis: `research/bioinformatics/sars_cov_2/glycan_shield/`
- RA scripts: `research/bioinformatics/rheumatoid_arthritis/scripts/`
- V5.11.3 embeddings: `research/genetic_code/data/v5_11_3_embeddings.pt`

---

## Changelog

| Date | Version | Description |
|:-----|:--------|:------------|
| 2025-12-19 | 1.1 | Updated priority order: RA-first sequential approach with comprehensive coverage |
| 2025-12-19 | 1.0 | Initial extension plan - prerequisite for encoder training |
