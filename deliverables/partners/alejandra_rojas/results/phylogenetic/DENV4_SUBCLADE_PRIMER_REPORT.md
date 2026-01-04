# DENV-4 Sub-clade Analysis for Primer Design

**Doc-Type:** Sub-clade Primer Report · Version 1.0 · 2026-01-04 · AI Whisperers

---

## Executive Summary

Initial phylogenetic analysis showed that 270 DENV-4 sequences cluster into 5 major clades, but the two largest (97% of sequences) have **NO conserved regions** suitable for primer design.

This sub-clade analysis performs **adaptive sub-clustering** to identify smaller, more homogeneous groups where conserved primer binding sites exist.

### Key Results

| Metric | Value |
|--------|-------|
| Total Sequences | 270 |
| Sequences with Primers | 36 (13.3%) |
| Sequences without Primers | 234 (86.7%) |
| Primer-Suitable Sub-clades | 1 |
| Total Sub-clades | 9 |

---

## Sub-clade Summary

| Sub-clade | Size | Entropy | Has Primers | Top Gene Region |
|-----------|------|---------|-------------|-----------------|
| Clade_E.1.3.2.1 | 88 | 1.558 | No | - |
| Clade_E.1.3.2.3 | 58 | 1.408 | No | - |
| Clade_E.3.2 | 36 | 0.405 | Yes | NS5 |
| Clade_D.2.1.1.1 | 12 | 1.078 | No | - |
| Clade_D.1.1.2 | 9 | 0.960 | No | - |
| Clade_D.2.2.2 | 9 | 1.388 | No | - |
| Clade_E.2.2.2 | 9 | 0.702 | No | - |
| Clade_D.3 | 7 | 1.462 | No | - |
| Clade_D.1.2 | 5 | 1.261 | No | - |

---

## Primer Candidates

Top primer candidates by sequence coverage:

| Clade | Gene | Position | Length | GC% | Tm | Entropy | Coverage |
|-------|------|----------|--------|-----|-----|---------|----------|
| Clade_E.3.2 | NS5 | 9908-9933 | 25bp | 44% | 72°C | 0.294 | 36 |

---

## Recommended Multiplex Strategy

Based on this analysis, the recommended approach is:

### Tier 1: High-Coverage Primers

Use primers from the largest primer-suitable sub-clades to cover the majority of sequences.

- **Clade_E.3.2** (36 seqs): NS5 9908-9933

### Tier 2: Rare Variant Coverage

For sequences not covered by Tier 1, consider:

1. **Degenerate primers** with IUPAC codes for variable positions
2. **Nested PCR** with outer pan-DENV-4 primers
3. **Amplicon sequencing** for novel variants

---

## Gap Analysis: Uncovered Sequences

**234 sequences** remain uncovered by current primer candidates.

These likely represent:
- Highly divergent or recombinant strains
- Novel genotypes not in clustering reference set
- Sylvatic strains with different evolutionary history

**Recommended action:** Analyze uncovered sequences separately for potential new primer targets.

---

*Analysis completed: 2026-01-04 04:22:20*

*IICS-UNA Arbovirus Surveillance Program*