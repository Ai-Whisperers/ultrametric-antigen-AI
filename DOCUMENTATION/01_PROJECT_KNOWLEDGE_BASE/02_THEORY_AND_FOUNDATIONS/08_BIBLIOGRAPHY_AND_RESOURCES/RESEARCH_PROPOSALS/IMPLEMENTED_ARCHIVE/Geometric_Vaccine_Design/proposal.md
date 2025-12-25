<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Proposal – Geometric Vaccine Design & Nanoparticle Scaffolding"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Proposal – Geometric Vaccine Design & Nanoparticle Scaffolding

## 1. Background & Motivation

The **COMPREHENSIVE_RESEARCH_REPORT.md** (Section 2.1) highlights that nanoparticle scaffolds such as ferritin, mi3, and virus‑like particles (VLPs) dramatically improve HIV‑Env antigen presentation. Translating these geometric principles into a differentiable loss term for the Ternary VAE can enable the generation of vaccine candidates with native‑like antigen spacing.

## 2. Objectives

1. Extract 3‑D coordinates of scaffold‑antigen complexes from the library (`RESEARCH_LIBRARY/03_REVIEW_INBOX/HIV_RESEARCH_2024/`).
2. Define a **GeometricAlignmentLoss** that penalises deviations from target inter‑epitope distances using a p‑adic distance matrix.
3. Integrate this loss into the VAE training pipeline and evaluate generated designs against experimental RMSD benchmarks.

## 3. Methodology

| Step                        | Action                                                                      | Tools / Files                                      |
| --------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------- |
| **A. Data Curation**        | Collect PDB files for ferritin‑Env, mi3‑Env, VLP‑Env.                       | `data/geometric_vaccine/` (to be created)          |
| **B. Distance Computation** | Compute pairwise Euclidean and p‑adic distances between displayed epitopes. | `src/utils/geometry.py` (new)                      |
| **C. Loss Definition**      | Implement `GeometricAlignmentLoss` in `src/losses/geometric_loss.py`.       | Existing loss module will be extended              |
| **D. Training Integration** | Add loss term to VAE training script (`src/train.py`).                      | Modify training loop to include `geom_loss_weight` |
| **E. Evaluation**           | Compare generated scaffold RMSD to reference (< 2 Å).                       | `tests/integration/test_geometric_vaccine.py`      |

## 4. Expected Outcomes

- A VAE capable of producing antigen‑nanoparticle configurations meeting geometric criteria.
- Quantitative improvement in RMSD over baseline models.
- Publication‑ready dataset of synthetic scaffold designs.

## 5. Implementation Checklist

- [ ] Create `data/geometric_vaccine/` and populate with PDBs.
- [ ] Add `src/utils/geometry.py` with distance utilities.
- [ ] Extend `src/losses/geometric_loss.py` with new class.
- [ ] Update `src/train.py` to accept `--geom_loss_weight`.
- [ ] Write unit tests in `tests/unit/test_geometry.py`.
- [ ] Document the workflow in `DOCUMENTATION/03_PRESENTATION_TIERS/Tier2_Carlos_Brizuela_Technical_Presentation.md`.

## 6. References

- **COMPREHENSIVE_RESEARCH_REPORT.md** – Sections 2.1 & 2.2.
- **CARLOS_BRIZUELA_FULL_PROFILE_AND_RESEARCH_REPORT.md** – Table of 30 papers (geometric vaccine entries).
- **RESEARCH_LIBRARY/03_REVIEW_INBOX/COMBINED_REVIEW_STRATEGY.md** – methodology for extracting structural data.
