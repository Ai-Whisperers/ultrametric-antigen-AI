# HIV Vulnerability Mapping: Partner Documentation

**Access Level**: Tier 1 (Partner)
**Status**: Validated predictions with reproducible protocols

---

## Executive Summary

Our computational platform analyzes HIV sequences using geometric methods to identify exploitable vulnerabilities in viral immune evasion. All predictions are falsifiable through independent validation.

---

## Validated Discoveries

### 1. Integrase Achilles Heel
HIV integrase (Pol_IN) is geometrically isolated from all other viral proteins.
- **Isolation score**: 3.24 (45% higher than next protein)
- **Vulnerability zones**: 13/13 protein pairs show gaps
- **Therapeutic target**: LEDGF interface modifications

### 2. Seven Sentinel Glycans
Specific glycosylation sites whose removal optimally exposes bnAb epitopes.
- **Sites**: N58, N429, N103, N204, N107, N271, N265
- **Validation**: AlphaFold3 confirms (r = -0.89)
- **Application**: Immunogen design for vaccines

### 3. Elite Controller Mechanism
HLA-B27/B57 alleles impose high "escape barriers" on HIV epitopes.
- **Barrier scores**: 4.18 - 4.40 for protective alleles
- **Mechanism**: Geometric constraints on mutation space
- **Application**: CTL-based therapeutic design

### 4. Drug Class Hierarchy
Antiretroviral classes ranked by evolutionary constraint:
- **NRTI**: 4.06 (highest barrier)
- **INSTI**: 4.30 (high barrier)
- **NNRTI**: 3.59 (moderate)
- **PI**: 3.52 (lower)

### 5. 49 Vulnerability Zones
Complete map of gaps between HIV proteins.
- **Severe**: 6 gaps (distance > 3.5)
- **Moderate**: 22 gaps (2.5 - 3.5)
- **Mild**: 21 gaps (< 2.5)

### 6. Accessory Protein Convergence
NC-Vif pair shares evasion architecture (proximity = 0.565).

### 7. Cascade Reveal Strategy
Single codon-level intervention reaches 89% of evasion mechanisms.

---

## Contents

| Directory | Description |
|:----------|:------------|
| [PREDICTIONS.md](./PREDICTIONS.md) | Detailed numerical predictions |
| [VALIDATION_PROTOCOLS/](./VALIDATION_PROTOCOLS/) | Step-by-step reproduction guides |
| [DATA/](./DATA/) | Sanitized JSON datasets |
| [SEQUENCES/](./SEQUENCES/) | AlphaFold3 input sequences |

---

## Methodology Overview

Our platform uses:
- **Geometric encoding** of codon sequences
- **Distance metrics** in non-Euclidean space
- **Hierarchical analysis** across protein/peptide/glycan levels

Specific algorithms, training procedures, and hyperparameters are proprietary.

---

## Validation Approach

All predictions can be independently verified:

1. **AlphaFold3**: Submit provided sequences, observe pLDDT correlations
2. **Los Alamos HIV DB**: Cross-reference escape frequencies
3. **Stanford HIVdb**: Compare resistance pathway complexity
4. **IEDB**: Validate epitope-HLA associations

---

## What's Protected

| Shared | Protected |
|:-------|:----------|
| Numerical predictions | Encoding scheme |
| Validation protocols | Model architecture |
| Input sequences | Training procedure |
| Score interpretations | Hyperparameters |

---

## Partnership Opportunities

- Vaccine immunogen design (sentinel glycans)
- Drug combination optimization (49-gap map)
- Elite controller mechanism studies
- Personalized therapy prediction

---

## Contact

For Tier 2 access (methodology overview, custom predictions): See [../pitch_tier2/CONTACT.md](../pitch_tier2/CONTACT.md)

---

*Proprietary computational platform. Predictions shared for validation and partnership.*
