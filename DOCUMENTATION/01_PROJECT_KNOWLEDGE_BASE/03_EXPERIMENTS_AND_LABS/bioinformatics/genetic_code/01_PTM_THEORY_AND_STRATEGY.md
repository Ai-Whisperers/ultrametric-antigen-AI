# PTM Extension: Theory & Strategy

**Doc-Type:** Research Strategy
**Parent:** [Bioinformatics Executive Summary](../BIOINFORMATICS_EXECUTIVE_SUMMARY.md)

---

## Purpose

Before designing the PTM-Goldilocks encoder architecture, we must extend empiric results across all three validated disease domains to ensure:

1.  Sufficient ground truth samples (~200-500 validated)
2.  Complete PTM type coverage per disease
3.  Empirically-derived Goldilocks boundaries
4.  Cross-disease pattern discovery

## Priority Order (Sequential)

We strictly follow a sequential execution model to ensure lessons learned in one domain apply to the next.

1.  **PHASE 1: RHEUMATOID ARTHRITIS (RA)**

    - _Why:_ Autoimmune = purest "host-vs-host" test case. Validated ACPA targets.
    - _Goal:_ Consolidate complete RA ground truth.

2.  **PHASE 2: HIV**

    - _Why:_ Inverse Goldilocks (using glycans to hide).
    - _Goal:_ Validate "Sentinel Glycan" theory.

3.  **PHASE 3: SARS-CoV-2**

    - _Why:_ Asymmetric Interface (Handshake theory).
    - _Goal:_ Validate viral-host interaction modeling.

4.  **PHASE 4: CROSS-DISEASE ANALYSIS**
    - _Goal:_ Train the unified encoder.

## Cross-Disease Analysis Plan

### Unified Ground Truth Dataset

We will merge all three disease datasets into a single JSON structure:

- **Metadata**: Encoder version, total samples, PTM types.
- **Samples**: Normalized list of all modifications with their calculated geometric shifts.

### Key Questions

1.  **Universal vs. Specific boundaries**: Is the 15-30% "Goldilocks Zone" universal, or does it shift for viral vs. autoimmune contexts?
2.  **PTM Type equivalence**: Does a Phosphorylation (S->D) create the same geometric distortion as Citrullination (R->Q)?
3.  **Edge Cases**: How do we handle multi-PTM sites or boundary cases (e.g., 14.9% shift)?
