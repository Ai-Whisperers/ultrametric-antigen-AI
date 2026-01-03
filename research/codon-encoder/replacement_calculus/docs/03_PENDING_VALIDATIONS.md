# Replacement Calculus: Pending Validations

**Doc-Type:** Research Roadmap · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Overview

This document tracks remaining validation experiments for the Replacement Calculus framework.

---

## Completed Validations

| ID | Validation | Result | Status |
|----|------------|--------|--------|
| V1 | P-adic valuation-based groupoid | 21.4% accuracy | COMPLETE |
| V2 | Embedding distance-based groupoid | 33.2% accuracy, F1=0.41 | COMPLETE |
| V3 | BLOSUM62 substitution matrix validation | 97.8% recall, 26.3% precision | COMPLETE |

---

## Pending Validations

### V4: Hybrid Morphism Validity

**Priority**: HIGH

**Hypothesis**: Combining embedding distance with physicochemical properties improves precision without sacrificing recall.

**Method**:
1. Compute embedding distance between amino acid centroids
2. Add physicochemical constraints:
   - Charge similarity: |charge(A) - charge(B)| ≤ 1
   - Size similarity: |MW(A) - MW(B)| ≤ 50 Da
   - Hydrophobicity: |GRAVY(A) - GRAVY(B)| ≤ 1.0
3. Morphism valid only if BOTH criteria pass
4. Validate against BLOSUM62

**Expected Outcome**: Higher precision (fewer false positives for radical substitutions)

**Files to Create**:
- `integration/hybrid_groupoid.py`

---

### V5: Gene Ontology Functional Validation

**Priority**: HIGH

**Hypothesis**: Escape paths predict functional annotation transfer.

**Method**:
1. Use Gene Ontology (GO) molecular function annotations
2. For each amino acid pair, check if they share GO terms
3. Compare path existence vs GO term sharing

**Ground Truth**:
- Same enzymatic function → should have path
- Different functions → should NOT have path

**Data Sources**:
- UniProt GO annotations
- QuickGO API

**Files to Create**:
- `integration/go_validation.py`

---

### V6: Evolutionary Rate Correlation

**Priority**: MEDIUM

**Hypothesis**: Path costs correlate with evolutionary substitution rates (dN/dS).

**Method**:
1. Get PAM matrices (PAM1, PAM30, PAM120, PAM250)
2. Compute correlation between path cost and PAM scores
3. Test if low-cost paths correspond to frequent evolutionary substitutions

**Expected Outcome**: Negative correlation (low cost = high evolutionary frequency)

**Files to Create**:
- `integration/pam_validation.py`

---

### V7: Protein Stability (DDG) Validation

**Priority**: MEDIUM

**Hypothesis**: Path costs predict mutation stability effects.

**Method**:
1. Use S669 DDG dataset (already downloaded)
2. For each mutation (A→B at position X), compute path cost
3. Correlate path cost with experimental DDG

**Connection to Previous Work**:
- This connects to Dr. Colbes validation
- Tests if groupoid structure encodes thermodynamic effects

**Files to Create**:
- `integration/ddg_validation.py`

---

### V8: Multi-Scale Groupoid Hierarchy

**Priority**: LOW

**Hypothesis**: Groupoid structure extends to protein domains and pathways.

**Method**:
1. Build codon-level groupoid (current)
2. Build amino acid-level groupoid (current)
3. Build domain-level groupoid (protein domains as LocalMinima)
4. Build pathway-level groupoid (metabolic pathways)
5. Test if morphisms compose across scales

**Expected Outcome**: Hierarchical groupoid structure mirrors biological organization

**Files to Create**:
- `integration/multiscale_groupoid.py`

---

### V9: Codon Usage Bias Validation

**Priority**: LOW

**Hypothesis**: Within-group (synonymous) codon preferences reflect p-adic structure.

**Method**:
1. Get organism-specific codon usage tables
2. Within each amino acid group, rank codons by usage frequency
3. Compare ranking to p-adic valuation ordering

**Expected Outcome**:
- High-expression genes prefer high-valuation codons (more stable)
- Confirms p-adic = information resilience hypothesis

**Files to Create**:
- `integration/codon_usage_validation.py`

---

## Falsification Studies (Completed)

These studies informed the framework design:

| Study | Finding | Implication |
|-------|---------|-------------|
| TEGB Falsification | P-adic anti-correlates with thermodynamics | P-adic ≠ stability |
| Multi-Prime Falsification | ALL primes (2,3,4,5,7,11,13) anti-correlate | Not prime-specific |
| Full S669 Analysis | Physicochemistry dominates (volume 75%) | Embeddings need physics |

See `falsification/COMBINED_FALSIFICATION_RESULTS.md` for details.

---

## Implementation Priority

| Priority | Validation | Effort | Impact |
|----------|------------|--------|--------|
| 1 | V4: Hybrid morphism | Medium | High |
| 2 | V5: GO functional | High | High |
| 3 | V7: DDG stability | Medium | Medium |
| 4 | V6: PAM evolutionary | Low | Medium |
| 5 | V9: Codon usage | Low | Low |
| 6 | V8: Multi-scale | High | Low |

---

## Dependencies

```
V4 (Hybrid) ──┬──> V5 (GO)
              │
              └──> V7 (DDG)

V6 (PAM) ───────> V9 (Codon Usage)

V8 (Multi-scale) depends on V4, V5, V7 completion
```

---

## Data Requirements

| Validation | Data Needed | Source | Status |
|------------|-------------|--------|--------|
| V4 | Amino acid properties | Built-in | Ready |
| V5 | GO annotations | UniProt/QuickGO | Download needed |
| V6 | PAM matrices | NCBI | Download needed |
| V7 | S669 DDG dataset | ProThermDB | Downloaded |
| V8 | Domain annotations | Pfam | Download needed |
| V9 | Codon usage tables | Kazusa | Download needed |

---

## Success Criteria

| Validation | Minimum Success | Target |
|------------|-----------------|--------|
| V4 | Precision > 50% | Precision > 70% |
| V5 | Accuracy > 60% | Accuracy > 75% |
| V6 | Spearman r > 0.3 | Spearman r > 0.5 |
| V7 | Spearman r > 0.2 | Spearman r > 0.4 |
| V8 | Cross-scale composition | Hierarchical structure |
| V9 | Correlation with usage | P-adic = resilience confirmed |

---

## Timeline Estimate

No specific timeline - work proceeds as priorities dictate. Tasks are ordered by impact and dependency, not calendar.
