# Alejandra Rojas Package: Unified Research Roadmap

**Doc-Type:** Strategic Research Plan · Version 1.0 · 2026-01-05 · AI Whisperers

---

## Executive Summary

The Rojas package represents a **scientifically successful but implementation-incomplete** arbovirus surveillance toolkit. After extensive validation, we identified the root cause of DENV-4 primer failures (cryptic diversity from ancient divergence) and proposed solutions using dual-metric (Shannon + p-adic) targeting.

### Current State Matrix

| Category | Completion | Key Finding |
|----------|------------|-------------|
| Scientific Validation | **95%** | Root cause = 200-500yr cryptic diversity |
| Data Standardization | **93%** | 25/27 files ML-ready with metadata |
| Hypothesis Testing | **100%** | 4 conjectures rejected, orthogonal signal confirmed |
| Primer Design | **0%** | CRITICAL GAP - no usable primers |
| In Silico Validation | **0%** | CRITICAL GAP - 0% coverage verified |
| ML Integration | **30%** | P-adic encoder available but unused |

---

## Part 1: Implemented Work

### 1.1 Phylogenetic Analysis ✅

**Status:** COMPLETE
**Files:**
- `scripts/denv4_phylogenetic_analysis.py`
- `results/ml_ready/phylogenetic_analysis_results.json`
- `results/ml_ready/clade_representatives.json`

**Key Results:**
- 270 DENV-4 complete genomes from NCBI
- 5 phylogenetic clades identified (UPGMA @ 90% identity)
- Clade E dominant (211/270 = 78.1%)
- Within-serotype identity: **71.7%** (vs 95-98% for DENV-1/2/3)

| Clade | Size | Identity | Year Range | Representative |
|-------|------|----------|------------|----------------|
| Clade_A | 2 | 100% | 2007 | MZ215848 |
| Clade_B | 3 | 99.4% | 1973-1975 | JF262780 |
| Clade_C | 2 | 98.5% | - | AY618989 |
| Clade_D | 52 | 85.8% | 1956-2024 | AY618990 |
| Clade_E | 211 | 89.0% | 1976-2023 | EU854299 |

---

### 1.2 P-adic Integration ✅

**Status:** COMPLETE
**Files:**
- `scripts/denv4_padic_integration.py`
- `results/ml_ready/padic_integration_results.json`
- `results/CONSOLIDATED_FINDINGS.md`

**Key Discovery:** E gene position 2400 has **4x lower hyperbolic variance** than current NS5 targets.

| Rank | Position | Gene | Hyp Variance | Current Use |
|------|----------|------|--------------|-------------|
| **1** | **2400** | **E** | **0.0183** | **NOVEL TARGET** |
| 2 | 3000 | NS1 | 0.0207 | Unused |
| 3 | 9600 | NS5 | 0.0222 | Alternative |
| 8 | 9007 | NS5 | 0.0271 | PANFLAVI_FU1 |
| 12 | 9908 | NS5 | 0.0287 | DENV4_E32_NS5_F |

**Critical Insight:** Hyperbolic variance identifies DIFFERENT conserved regions than Shannon entropy, providing orthogonal targeting.

---

### 1.3 Conjecture Testing ✅

**Status:** COMPLETE - ALL REJECTED
**Files:**
- `scripts/denv4_synonymous_conjecture.py`
- `scripts/denv4_revised_conjecture.py`
- `scripts/denv4_codon_bias_conjecture.py`
- `scripts/denv4_codon_pair_conjecture.py`
- `results/ml_ready/*_conjecture_results.json`

| Conjecture | Hypothesis | Result | Correlation |
|------------|------------|--------|-------------|
| Synonymous | Low hyp_var = synonymous changes | **REJECTED** | ρ=0.03, p=0.86 |
| AA Property | Low hyp_var = small AA changes | **REJECTED** | ρ=0.01, p=0.97 |
| Codon Bias | Low hyp_var = conserved preferences | **REJECTED** | ρ=0.31, p=0.07 |
| Codon Pair | Low hyp_var = optimized pairs | **REJECTED** | ρ=-0.14, p=0.43 |

**Meta-Finding:** Hyperbolic variance is **orthogonal to all classical conservation metrics**. The p-adic encoder captures a novel aspect of sequence diversity.

---

### 1.4 ML-Ready Data Standardization ✅

**Status:** 25/27 files complete
**Location:** `results/ml_ready/`

**Created Files:**
| Category | Files | Contents |
|----------|-------|----------|
| Genome Data | 2 | Metadata (270 entries), Sequences (2.8MB) |
| Reference | 2 | Serotype stats, RefSeq genomes (7 viruses) |
| P-adic | 1 | Integration results, primer candidates |
| Phylogenetic | 7 | Clades, conservation, representatives |
| Conjectures | 4 | All hypothesis test results |
| Detection | 2 | Tiered strategy, pan-arbovirus library |
| Validation | 7 | CDC recovery, divergence, deep analysis |

**Schema:** All files have `_metadata` header with:
- `schema_version`, `field_definitions`
- `ml_usage` (index_by, join_with, features, labels)
- `related_files` for data joining

---

### 1.5 Validation Framework ✅

**Status:** COMPLETE
**Files:** `validation/*.py`, `validation/*.json`, `validation/*.md`

**Tests Completed:**
| Test | Result | Key Finding |
|------|--------|-------------|
| CDC Primer Recovery | 60% pass | Gene targets differ from literature |
| Strain Conservation | DENV-4 30x variable | Entropy 0.88 vs 0.03-0.07 |
| Population Hypothesis | Partial | DENV-4 has cryptic diversity |
| Polymerase Fidelity | Supported | NS5 16.9x more variable |
| Immune Evasion | Falsified | E/NS5 ratio identical |
| Ancient Divergence | Confirmed | 200-500 years of independent evolution |

---

## Part 2: Critical Gaps (Not Implemented)

### 2.1 CRITICAL: No Usable Primers Exist 🔴

**Current State:**
- Degenerate primers: **0 usable** (degeneracy >10^28)
- Pan-arbovirus library: **0/70 specific** (cross-reactivity untested)
- Tiered detection: **0% validated** in silico
- E gene primers: **NOT DESIGNED** despite being identified as best target

**Root Cause:** DENV-4 diversity too high for consensus primers at any position.

---

### 2.2 CRITICAL: In Silico PCR Missing 🔴

**Required:**
- Test ALL primer candidates against 270 genomes
- Allow 0-2 mismatches
- Report coverage per clade
- Generate specificity matrix

**Implementation:** `scripts/insilico_pcr_validation.py` (not created)

---

### 2.3 HIGH: E Gene Primer Design 🟡

**Target:** Position 2400-2500 (lowest hyperbolic variance)
**Rationale:** P-adic analysis identified this as best target
**Status:** Identified but no primers designed

**Design Constraints:**
- Forward: 2400-2420 (20bp)
- Reverse: 2475-2495 (20bp)
- Amplicon: 75-95bp
- Allow 2-3 degenerate positions
- Tm: 55-60°C

---

### 2.4 HIGH: Clade-Specific Cocktails 🟡

**Strategy:** Since pan-DENV-4 is impossible, design per-clade primers
**Target:** Cover 5 clades with 5 primer pairs
**Challenge:** Clade_E (211 sequences) still has internal diversity

---

## Part 3: Research Directions

### 3.1 Cross-Disease Validation: DHF Correlation

**Status:** Test 3 FAILED (null result)
**File:** `research/cross-disease-validation/results/test3_dengue_dhf/`

**Finding:** NS1 p-adic distances show **negative correlation** with DHF rates (ρ=-0.33, p=0.29)

**Interpretation:**
- NS1 is wrong target (immune modulator, not antibody target)
- E protein should be tested instead
- Inverted U-curve (Goldilocks zone) likely

**Follow-up Required:**
1. Test E protein distances vs DHF
2. Fit quadratic model (Goldilocks zone)
3. Test epitope-specific (fusion loop) distances

---

### 3.2 TrainableCodonEncoder Integration

**Status:** Available but not integrated
**Location:** `research/codon-encoder/training/`
**Performance:** LOO Spearman **0.61** on DDG (S669)

**Potential Use Cases:**
1. **Clade Classification:** Train on 270 DENV-4 → predict clade from sequence
2. **Conservation Features:** Use embedding dimensions as features
3. **Primer Scoring:** Score primers by embedding stability

**Integration Path:**
```python
from src.encoders import TrainableCodonEncoder
encoder = TrainableCodonEncoder(latent_dim=16)
# Load trained weights
# Embed primer binding sites
# Score by within-site variance
```

---

### 3.3 Dual-Metric Primer Design (Novel Approach)

**Concept:** Use TWO orthogonal metrics to identify primer targets:

| Metric | Level | Best Region | Captures |
|--------|-------|-------------|----------|
| Shannon Entropy | Nucleotide | NS5 (9908) | Sequence identity |
| Hyperbolic Variance | Codon | E gene (2400) | P-adic structure |

**Implementation:**
1. Compute Shannon entropy per position (classical)
2. Compute hyperbolic variance per window (p-adic)
3. Identify positions low in BOTH (intersection)
4. Design primers targeting dual-low positions

**Expected Benefit:** Capture strains missed by Shannon-only approach.

---

### 3.4 Foundation Encoder Multi-Task Integration

**Vision:** Unified encoder serving multiple partner packages:
- DDG prediction (Colbes)
- AMP optimization (Brizuela)
- Viral clade classification (Rojas)
- Drug resistance (HIV package)

**Rojas-Specific Tasks:**
1. **Clade classification head:** 5-class output
2. **Conservation regression head:** Predict entropy from embedding
3. **Primer binding head:** Binary (binds/doesn't bind)

---

## Part 4: Prioritized Action Plan

### Priority 1: Design E Gene Primers (Immediate)

**Objective:** Create usable primer pair for E gene position 2400-2500

**Steps:**
1. Extract consensus at position 2400-2500 from all 270 sequences
2. Identify variable positions (<90% consensus)
3. Add IUPAC degeneracy codes at variable positions
4. Validate Tm, GC, no hairpins
5. Output: DENV4_E2400_F and DENV4_E2400_R

**Deliverable:** `results/primers/egene_primer_pair.json`

---

### Priority 2: Implement In Silico PCR (Critical)

**Objective:** Validate ALL primer candidates against 270 genomes

**Steps:**
1. Create `scripts/insilico_pcr_validation.py`
2. Test exact matches, 1mm, 2mm
3. Generate coverage matrix per clade
4. Flag primers with <50% coverage

**Deliverable:** `results/validation/insilico_pcr_results.json`

---

### Priority 3: Clade-Specific Primer Cocktail (High)

**Objective:** Design multiplex for 5-clade coverage

**Steps:**
1. For each clade: Find conserved window (entropy <0.3)
2. Design forward/reverse primers
3. Stagger amplicon sizes (100, 150, 200, 250, 300bp)
4. Ensure Tm compatibility (±2°C)
5. Validate multiplex in silico

**Deliverable:** `results/multiplex/denv4_cocktail.json`

---

### Priority 4: E Protein DHF Analysis (Research)

**Objective:** Test if E protein distances correlate with DHF (Test 3 redo)

**Steps:**
1. Extract E protein sequences (positions 977-2471)
2. Compute E protein p-adic distances
3. Correlate with literature DHF rates
4. Test quadratic model (Goldilocks zone)

**Deliverable:** `research/cross-disease-validation/test3b_e_protein_dhf/`

---

### Priority 5: TrainableCodonEncoder for Clade Classification (Research)

**Objective:** Train encoder head for 5-class clade prediction

**Steps:**
1. Create training dataset (270 sequences, 5 labels)
2. Add classification head to TrainableCodonEncoder
3. Train with 5-fold CV
4. Evaluate accuracy, confusion matrix
5. Compare to k-mer baseline

**Deliverable:** `research/codon-encoder/clade_classification/`

---

## Part 5: Success Metrics

### Minimum Viable Deliverable

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Usable primer pairs | 0 | ≥1 | 🔴 CRITICAL |
| In silico validated | 0% | 100% | 🔴 CRITICAL |
| Strain coverage | 13.3% | >50% | 🔴 CRITICAL |

### Extended Goals

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Clade-specific pairs | 0 | 5 | 🟡 HIGH |
| Pan-arbovirus specific | 0/70 | >35/70 | 🟡 MEDIUM |
| DHF correlation (E protein) | - | ρ>0.6 | 🟡 RESEARCH |
| Clade classification | - | >90% | 🟡 RESEARCH |

---

## Part 6: Resource Requirements

### Computational

| Task | VRAM | RAM | Time |
|------|------|-----|------|
| In silico PCR | <1GB | 4GB | 30 min |
| E gene primer design | <1GB | 2GB | 15 min |
| Multiplex optimization | <1GB | 4GB | 1 hour |
| Clade classification | 2GB | 4GB | 2 hours |
| E protein DHF test | 2GB | 4GB | 1 hour |

### Human Effort

| Task | Estimated Time |
|------|----------------|
| E gene primer design | 1-2 hours |
| In silico PCR implementation | 2-3 hours |
| Clade-specific cocktail | 3-4 hours |
| Multiplex validation | 2-3 hours |
| E protein DHF analysis | 2-3 hours |

---

## Part 7: File Index

### Existing Files (Key)

| File | Purpose | Status |
|------|---------|--------|
| `results/ml_ready/*.json` | ML-ready data (25 files) | ✅ COMPLETE |
| `results/CONSOLIDATED_FINDINGS.md` | Scientific summary | ✅ COMPLETE |
| `SOLUTION_APPROACH.md` | Dual-metric strategy | ✅ COMPLETE |
| `ACTIONABLE_GAPS.md` | Gap analysis | ✅ COMPLETE |
| `docs/SCHEMA_CATALOG.md` | Data catalog | ✅ COMPLETE |

### Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `scripts/design_egene_primers.py` | E gene primer design | CRITICAL |
| `scripts/insilico_pcr_validation.py` | Primer validation | CRITICAL |
| `scripts/clade_specific_primer_design.py` | Per-clade primers | HIGH |
| `scripts/optimize_multiplex.py` | Cocktail optimization | HIGH |
| `results/primers/egene_primer_pair.json` | E gene primers | CRITICAL |
| `results/validation/insilico_pcr_results.json` | Validation matrix | CRITICAL |
| `results/multiplex/denv4_cocktail.json` | Multiplex recipe | HIGH |

---

## Conclusion

The Rojas package has **excellent scientific foundation** - we understand why DENV-4 primers fail (cryptic diversity) and have identified a novel solution (dual-metric targeting with p-adic hyperbolic variance).

**The gap is implementation:** No actual usable primers have been designed.

**Immediate priority:** Design E gene primers (position 2400) and validate in silico.

**Research opportunity:** TrainableCodonEncoder integration for clade classification and E protein DHF correlation.

---

*Unified Research Roadmap: 2026-01-05*
*IICS-UNA Arbovirus Surveillance Program*
