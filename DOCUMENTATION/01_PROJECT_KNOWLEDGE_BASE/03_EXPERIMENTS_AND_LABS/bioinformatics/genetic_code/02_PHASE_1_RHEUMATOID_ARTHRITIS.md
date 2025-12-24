# PTM Extension: Phase 1 - Rheumatoid Arthritis

**Status:** ACTIVE PRIORITY
**Goal:** Establish the ground truth for "Host-vs-Host" autoimmunity.

---

## Why RA First?

- **Host-vs-Host**: No viral asymmetry to confound results.
- **Validated Targets**: ACPA (Anti-Citrullinated Protein Antibody) targets are clinically known.
- **Genetic Link**: HLA-DRB1 association provides a clear "Shared Epitope" ground truth.

## Methodology

We define a comprehensive sweep of **10 ACPA proteins** (Fibrinogen, Vimentin, Enolase, etc.).

### 1. Comprehensive PTM Sweep

We test **ALL** modifiable residues against **ALL** relevant PTM types:

- **Citrullination (R->Q)**: The primary ACPA trigger.
- **Phosphomimics (S->D, T->D, Y->D)**: To test geometric similarity.
- **Deglycosylation (N->Q)**: To test inverse effects.

### 2. Handshake Interfaces

We analyze the geometric shift in the context of three critical interfaces:

1.  **HLA-Peptide**: How the peptide fits into the MHC groove (P1, P4 pockets).
2.  **TCR-pMHC**: How the T-cell receptor "sees" the presented peptide (P5, P7, P8).
3.  **B-cell Epitopes**: How antibodies bind the surface.

## Current Status (Dec 2025)

- **Proteins Analyzed**: 10/10 (Complete)
- **Known ACPA Detection**: 100% of known sites fall in the high-priority "Goldilocks" list.
- **Key Finding**: Citrullination alone isn't enough; it must happen at a TCR-facing residue to trigger immunity.

## Next Steps

1.  Run AlphaFold3 validation on the identified high-priority targets.
2.  Consolidate `ra_ptm_ground_truth.json`.
