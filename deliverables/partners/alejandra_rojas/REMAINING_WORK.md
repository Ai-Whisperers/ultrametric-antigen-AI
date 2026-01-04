# Alejandra Rojas Package: Remaining Work

**Doc-Type:** Gap Analysis & Roadmap · Version 1.0 · 2026-01-04 · AI Whisperers

---

## Executive Summary

The validation phase has exceeded expectations, discovering the root cause of DENV-4 primer failures (cryptic diversity from ancient diversification). However, **implementation of actionable solutions is incomplete**.

### Current State

| Category | Completed | Remaining |
|----------|-----------|-----------|
| Scientific Validation | 90% | Root cause identified |
| Documentation | 85% | Minor gaps |
| Implementation | 10% | Major gaps |
| Primer Design | 0% | Not started |

---

## Phase 1: COMPLETED - Scientific Validation

### 1.1 CDC Primer Recovery ✅
- 60% full recovery (met threshold)
- Gene target corrections documented
- Literature annotations found unreliable

### 1.2 Strain Conservation Analysis ✅
- 80 genomes analyzed (20 per serotype)
- DENV-4 entropy 30x higher confirmed
- Forward primer: 90% positions variable
- Reverse primer: 75% positions variable

### 1.3 Hypothesis Testing ✅

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| Serotype Competition | FALSIFIED | DENV-4 reaches 75% locally |
| Polymerase Fidelity | SUPPORTED | NS5 16.9x more variable |
| Immune Evasion | FALSIFIED | E/NS5 ratio identical |
| Neutral Evolution | SUPPORTED | dN/dS proxy ~1.0 |

### 1.4 Deep Molecular Analysis ✅
- ALL NS5 domains >90% variable
- Catalytic GDD motif variable (entropy 1.063)
- DENV-4 within-serotype identity: 71.7%
- Divergence estimate: 200-500 years

### 1.5 Root Cause Identified ✅
**DENV-4 contains cryptic diversity** - multiple deeply divergent lineages classified as single serotype.

---

## Phase 2: INCOMPLETE - Ground Truth & Comparison

### 2.1 Ground Truth Primer Database ❌
**Status:** Not started
**Priority:** HIGH

**Required Actions:**
1. Download CDC published primer sequences (not just from literature)
2. Curate PAHO regional primers for Latin America
3. Curate WHO recommended primers
4. Create standardized database format
5. Document primer metadata (target gene, amplicon size, publication)

**Files to Create:**
- `data/primers/cdc_primers.json`
- `data/primers/paho_primers.json`
- `data/primers/who_primers.json`
- `data/primers/PRIMER_DATABASE_SCHEMA.md`

### 2.2 Hyperbolic vs Euclidean Comparison ❌
**Status:** Not started
**Priority:** MEDIUM

**Required Actions:**
1. Compute Euclidean distances between codon embeddings
2. Compute hyperbolic (Poincaré) distances
3. Compare correlation with conservation
4. Test if hyperbolic geometry adds predictive value

**Hypothesis:** P-adic structure may improve primer binding site identification.

**Note:** P-adic conservation correlation was null (ρ=-0.012), suggesting p-adic encodes genetic code structure (universal) not virus-specific evolution (context-specific). Hyperbolic comparison may show same result.

---

## Phase 3: NOT STARTED - Implementation

### 3.1 DENV-4 Phylogenetic Analysis 🔴
**Status:** Not started
**Priority:** CRITICAL

**Required Actions:**
1. Download ALL DENV-4 sequences from NCBI (~270)
2. Multiple sequence alignment (MUSCLE or MAFFT)
3. Build phylogenetic tree (IQ-TREE or RAxML)
4. Identify major clades/genotypes
5. Map geographic distribution per clade
6. Select representative sequences per clade

**Files to Create:**
- `data/denv4/all_denv4_sequences.fasta`
- `data/denv4/denv4_alignment.fasta`
- `data/denv4/denv4_tree.nwk`
- `data/denv4/clade_assignments.json`
- `results/denv4_phylogeny/CLADE_REPORT.md`

### 3.2 Genotype-Specific Primer Design 🔴
**Status:** Not started
**Priority:** CRITICAL

**Required Actions:**
1. Align sequences within each genotype
2. Compute per-position entropy within each clade
3. Identify conserved windows (entropy <0.3)
4. Design primers for each genotype:
   - Forward: 18-22 bp, 40-60% GC, Tm 58-62°C
   - Reverse: same constraints
   - Amplicon: 80-150 bp
5. Ensure Tm compatibility (±2°C) across all pairs
6. Add degeneracy where needed (IUPAC codes)

**Files to Create:**
- `scripts/design_denv4_primers.py`
- `results/primers/genotype_I_primers.json`
- `results/primers/genotype_II_primers.json`
- `results/primers/genotype_III_primers.json`
- `results/primers/PRIMER_DESIGN_REPORT.md`

### 3.3 In Silico Validation 🔴
**Status:** Not started
**Priority:** CRITICAL

**Required Actions:**
1. In silico PCR against all DENV-4 NCBI sequences
2. Check for cross-reactivity with DENV-1/2/3
3. Check for off-target human genome hits
4. Compute expected sensitivity per genotype
5. Compute expected specificity (no cross-amplification)

**Files to Create:**
- `scripts/insilico_pcr_validation.py`
- `results/validation/insilico_pcr_results.json`
- `results/validation/INSILICO_VALIDATION_REPORT.md`

### 3.4 Multiplex Optimization ⚠️
**Status:** Strategy documented, implementation not started
**Priority:** HIGH

**Required Actions:**
1. Stagger amplicon sizes (100, 150, 200 bp)
2. Ensure no primer-dimer formation
3. Optimize concentration ratios
4. Design internal positive control

**Files to Create:**
- `scripts/optimize_multiplex.py`
- `results/multiplex/OPTIMIZATION_REPORT.md`
- `results/multiplex/final_multiplex_recipe.json`

---

## Phase 4: NOT STARTED - Quality & Automation

### 4.1 Unit Tests ❌
**Status:** Not started
**Priority:** MEDIUM

**Required Actions:**
1. Test primer design constraints
2. Test entropy computation
3. Test in silico PCR logic
4. Test multiplex optimization

**Files to Create:**
- `tests/test_primer_design.py`
- `tests/test_entropy_analysis.py`
- `tests/test_insilico_pcr.py`

### 4.2 Automated Primer Update Pipeline ⚠️
**Status:** Protocol documented, not automated
**Priority:** LOW

**Required Actions:**
1. Script to download new NCBI sequences quarterly
2. Re-compute binding site conservation
3. Flag primers with <90% perfect matches
4. Alert for redesign

**Files to Create:**
- `scripts/quarterly_primer_update.py`
- `configs/update_schedule.yaml`

---

## Success Metrics Gap Analysis

| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| Specific primers per virus | >5 | 0 | 🔴 CRITICAL |
| Strain coverage (DENV-4) | >95% | Unknown | 🔴 CRITICAL |
| False negative rate | <5% | Unknown | 🔴 CRITICAL |
| Hypothesis tests completed | 5 | 5 | ✅ COMPLETE |
| Literature discrepancies found | >3 | 4 | ✅ COMPLETE |
| Statistical validation scripts | 5 | 0 | ⚠️ MEDIUM |

---

## Recommended Roadmap

### Week 1: Phylogenetic Foundation
1. Download all DENV-4 sequences
2. Build phylogenetic tree
3. Assign clades
4. Map geography

### Week 2: Primer Design
1. Design genotype-specific primers
2. Ensure Tm compatibility
3. Add degeneracy

### Week 3: Validation
1. In silico PCR
2. Cross-reactivity check
3. Sensitivity/specificity computation

### Week 4: Multiplex & Documentation
1. Optimize multiplex conditions
2. Create final recipe
3. Update all documentation
4. Create user guide

---

## Files Index

### Completed This Session
| File | Purpose |
|------|---------|
| `validation/VALIDATION_MASTER_REPORT.md` | All findings synthesis |
| `docs/MULTIPLEXED_DETECTION_STRATEGY.md` | 5 detection approaches |
| `validation/DENV4_MOLECULAR_ANALYSIS_REPORT.md` | Deep NS5 analysis |
| `validation/ALTERNATIVE_HYPOTHESES_REPORT.md` | Hypothesis testing |
| `validation/test_alternative_hypotheses.py` | Hypothesis tests |
| `validation/test_ns5_deep_analysis.py` | Domain analysis |
| `validation/test_evolutionary_divergence.py` | Divergence timing |

### To Be Created
| File | Purpose | Priority |
|------|---------|----------|
| `data/denv4/clade_assignments.json` | Genotype mapping | CRITICAL |
| `scripts/design_denv4_primers.py` | Primer design | CRITICAL |
| `scripts/insilico_pcr_validation.py` | Validation | CRITICAL |
| `results/primers/genotype_*_primers.json` | Final primers | CRITICAL |
| `tests/test_*.py` | Quality assurance | MEDIUM |

---

## Conclusion

The scientific foundation is **excellent** - we've identified why DENV-4 primers fail and defined the solution (multiplexed genotype-specific detection).

The gap is **implementation**: no actual primers have been designed yet.

**Minimum viable deliverable:** Genotype-specific DENV-4 primers validated in silico.

---

*Gap analysis completed: 2026-01-04*
*IICS-UNA Arbovirus Surveillance Program*
