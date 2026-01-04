# Validation Methodology for Arbovirus Primer Design

**Doc-Type:** Scientific Methodology · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Purpose

This document defines the scientific validation methodology for the pan-arbovirus primer design toolkit. It addresses:
1. Current assumptions and their limitations
2. Rigorous test fixture requirements
3. Falsification criteria for hypothesis testing
4. Ground truth sources for validation

---

## Current Assumptions Analysis

### Assumption 1: Random Demo Sequences Approximate Real Conservation

**Current State:**
- Demo mode generates random sequences with fixed conserved regions at positions 100, 500, 5000, 9000
- All viruses share identical conserved motifs

**Problem:**
- Random sequences show 100% cross-reactivity because the only similar regions are the hardcoded ones
- This produces 0 specific primers (observed in `library_summary.json`)
- Does not capture real evolutionary divergence between serotypes

**Scientific Issue:**
Cross-reactivity depends on sequence identity gradients:
- Dengue serotypes: 62-68% amino acid identity
- Flavivirus family (Dengue/Zika): 40-55% identity
- Flavivirus vs Alphavirus (Dengue/Chikungunya): 15-25% identity

**Fix Required:**
Generate sequences with realistic identity profiles based on known phylogenetic distances.

---

### Assumption 2: 70% Cross-Reactivity Threshold

**Current State:**
- `max_cross_reactivity: 0.7` threshold for primer specificity
- Primers with >70% match to non-target viruses are rejected

**Problem:**
- For closely related serotypes (DENV-1 vs DENV-2 at 65% identity), this threshold may be too strict
- For distantly related viruses (DENV vs CHIKV at 25% identity), this threshold is too lenient

**Scientific Issue:**
PCR specificity depends on:
- 3' end mismatch position (last 6 nt critical)
- GC content of mismatch region
- Annealing temperature vs Tm differential
- Primer length

A single 3' terminal mismatch often provides specificity even at 95% overall identity.

**Fix Required:**
- Implement position-weighted similarity scoring
- Penalize matches in 3' region more heavily
- Use empirical specificity data from published pan-flavivirus assays

---

### Assumption 3: P-adic Stability Correlates with Conservation

**Current State:**
- Higher p-adic valuation = more evolutionarily stable
- Stability score uses mean valuation across primer codons

**Hypothesis:**
Codons with high 3-adic valuation (divisible by powers of 3) occur at conserved positions because:
- Synonymous codons cluster by valuation
- Functional constraints preserve high-valuation positions

**Validation Required:**
Test on real aligned sequences:
1. Extract p-adic valuations for all codon positions
2. Correlate with Shannon entropy (conservation metric)
3. Expected: negative correlation (high valuation → low entropy)

---

## Rigorous Test Fixtures

### Fixture 1: Synthetic Phylogenetic Sequences

Generate sequences with controlled identity matrices:

```python
IDENTITY_MATRIX = {
    ('DENV-1', 'DENV-2'): 0.65,
    ('DENV-1', 'DENV-3'): 0.63,
    ('DENV-1', 'DENV-4'): 0.62,
    ('DENV-2', 'DENV-3'): 0.66,
    ('DENV-2', 'DENV-4'): 0.64,
    ('DENV-3', 'DENV-4'): 0.63,
    ('DENV-*', 'ZIKV'): 0.45,
    ('DENV-*', 'CHIKV'): 0.22,
    ('DENV-*', 'MAYV'): 0.24,
    ('ZIKV', 'CHIKV'): 0.18,
    ('CHIKV', 'MAYV'): 0.62,
}
```

Algorithm:
1. Start with real DENV-1 reference (NC_001477)
2. Mutate to target identity using codon-aware substitutions
3. Preserve conserved functional domains (UTRs, RdRp active site)
4. Add insertion/deletion variation at known variable sites

### Fixture 2: Known Published Primers as Ground Truth

Use validated CDC/PAHO primers as positive controls:

| Primer Set | Target | Forward | Reverse | Validated |
|------------|--------|---------|---------|-----------|
| CDC DENV-1 | NS5 | 5'-CAAAAGGAAGTCGTGCAATA-3' | 5'-CTGAGTGAATTCTCTCTACTGAACC-3' | Yes |
| CDC DENV-2 | NS1 | 5'-GCAGATCTCTGATGAATAACCAAC-3' | 5'-TTTGTGGAATGAAGTGCAGATCTG-3' | Yes |
| Lanciotti ZIKV | Env | 5'-AARTACACATACCARAACAAAGTGGT-3' | 5'-TCCRCTCCCYCTYTGGTCTTG-3' | Yes |

Test criteria:
- Our algorithm should assign high scores to these primers
- If rejected, analyze why (threshold too strict?)

### Fixture 3: Known Cross-Reactive Primers as Negative Controls

Include primers known to cross-react:

| Primer | Intended Target | Actual Cross-Reactivity |
|--------|-----------------|-------------------------|
| Panflavivirus NS5 | All Flavivirus | DENV, ZIKV, YFV, JEV, WNV |
| Generic Alphavirus | Alphaviruses | CHIKV, MAYV, VEEV, EEEV |

Test criteria:
- Our algorithm should detect and flag these as non-specific
- Cross-reactivity score should exceed threshold

---

## Falsification Criteria

### Hypothesis 1: P-adic Geometry Improves Primer Stability Prediction

**Null hypothesis:** Random conservation scoring performs equally well.

**Test:**
1. Generate random stability scores for same primer set
2. Compare correlation with actual sequence conservation (from real alignments)
3. p < 0.05 required for p-adic method superiority

**Falsification:** If random scoring achieves comparable correlation, p-adic approach adds no value.

### Hypothesis 2: Hyperbolic Embeddings Predict Serotype Dominance

**Null hypothesis:** Euclidean embeddings perform equally well.

**Test:**
1. Embed same sequences in Euclidean space
2. Compare trajectory forecasts against historical data
3. Use RMSE for position prediction accuracy

**Falsification:** If Euclidean RMSE ≤ Hyperbolic RMSE, hyperbolic geometry adds no value.

### Hypothesis 3: Conserved Regions Have Higher P-adic Valuation

**Null hypothesis:** No correlation between valuation and conservation.

**Test:**
1. Obtain real Dengue alignments (NCBI RefSeq)
2. Compute Shannon entropy per position
3. Compute mean p-adic valuation per position (across all codons covering that position)
4. Spearman correlation test

**Falsification:** If ρ > -0.2 (weak or positive correlation), hypothesis is rejected.

---

## Ground Truth Data Sources

### NCBI Virus Sequences

| Virus | RefSeq Accession | Genome Length | Source |
|-------|------------------|---------------|--------|
| DENV-1 | NC_001477 | 10,735 nt | NCBI |
| DENV-2 | NC_001474 | 10,723 nt | NCBI |
| DENV-3 | NC_001475 | 10,707 nt | NCBI |
| DENV-4 | NC_002640 | 10,649 nt | NCBI |
| ZIKV | NC_012532 | 10,794 nt | NCBI |
| CHIKV | NC_004162 | 11,826 nt | NCBI |
| MAYV | NC_003417 | 11,429 nt | NCBI |

### Published Primer Validation Studies

| Citation | Primers Tested | Species Coverage | Sensitivity |
|----------|----------------|------------------|-------------|
| Lanciotti 2008 | CDC pan-DENV | DENV 1-4 | 100% |
| Santiago 2018 | Multiplex DENV/ZIKV | DENV 1-4, ZIKV | 98% |
| PAHO 2016 | Regional surveillance | All arboviruses | 95% |

---

## Implementation Plan

### Phase 1: Reference Sequence Integration (Required First)

1. Download RefSeq genomes for all 7 viruses
2. Store in `data/reference/` as individual FASTA files
3. Create alignment-derived identity matrix
4. Update demo sequence generator to use phylogenetically-informed mutations

### Phase 2: Ground Truth Primer Database

1. Curate published primer sequences (CDC, PAHO, WHO)
2. Store in structured format with validation metadata
3. Implement automatic comparison in test suite

### Phase 3: Statistical Validation Framework

1. Implement falsification tests as unit tests
2. Set up continuous validation against NCBI updates
3. Document pass/fail criteria for each hypothesis

---

## Success Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Specific primers per virus | 0 | >5 | Use realistic sequences |
| CDC primer recovery rate | Unknown | >80% | Include ground truth |
| P-adic vs conservation ρ | Unknown | < -0.3 | Real sequence validation |
| Hyperbolic vs Euclidean RMSE | Unknown | <0.8 ratio | Trajectory benchmarks |

---

## References

1. Lanciotti RS et al. (2008) Genetic and serologic properties of Zika virus. Emerg Infect Dis.
2. Santiago GA et al. (2018) Analytical and clinical performance of the CDC real-time RT-PCR assay for detection and typing of dengue virus.
3. PAHO (2016) Guidelines for arbovirus surveillance in the Americas.
4. WHO (2009) Dengue: guidelines for diagnosis, treatment, prevention and control.

---

*Prepared for IICS-UNA Arbovirus Surveillance Program*
