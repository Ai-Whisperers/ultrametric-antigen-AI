<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

---
title: "Proposal – Autoimmunity & Codon Adaptation"
date: 2025-12-24
authors:
  - AI Whisperers
version: "0.1"
license: PolyForm-Noncommercial-1.0.0
---

# Proposal – Autoimmunity & Codon Adaptation

## 1. Background & Motivation

The **COMPREHENSIVE_RESEARCH_REPORT.md** (Section 2.8) and the **autoimmunity_and_codons** sub‑folder describe how host immune pressure can shape viral codon usage. Modeling this relationship can improve predictions of viral evolution and vaccine escape.

## 2. Objectives

1. Quantify codon‑usage bias in auto‑immune‑associated viral strains.
2. Extend the `CodonEncoder` with a regularisation term that penalises deviations from host‑favoured codons under immune pressure.
3. Validate the model by reproducing known escape mutations in HIV and other pathogens.

## 3. Methodology

| Step                        | Action                                                                                                              | Tools / Files                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **A. Data Collection**      | Gather viral genome sequences from patients with documented autoimmune markers (CD4/CD8 ratios, cytokine profiles). | `data/autoimmunity_viral/` (to be created)     |
| **B. Codon Bias Analysis**  | Compute Relative Synonymous Codon Usage (RSCU) and compare against host codon preferences.                          | `src/analysis/codon_bias_autoimmunity.py`      |
| **C. Encoder Extension**    | Add `AutoimmuneCodonRegularizer` to `src/encoders/codon_encoder.py`.                                                | Extend existing encoder                        |
| **D. Training Integration** | Include regulariser loss in VAE training (`src/train.py`).                                                          | Modify loss weighting                          |
| **E. Validation**           | Test on held‑out viral strains; measure prediction of escape mutations (precision > 0.75).                          | `tests/integration/test_autoimmunity_codon.py` |

## 4. Expected Outcomes

- A measurable link between immune pressure and codon adaptation captured in the latent space.
- Improved ability to forecast viral escape pathways.
- Publication‑ready figures showing codon‑usage shifts under different CD4/CD8 ratios.

## 5. Implementation Checklist

- [ ] Create `data/autoimmunity_viral/` with sequence files.
- [ ] Implement analysis script.
- [ ] Extend `CodonEncoder` with regulariser.
- [ ] Update training pipeline.
- [ ] Write unit and integration tests.
- [ ] Document in the technical presentation deck.

## 6. References

- **COMPREHENSIVE_RESEARCH_REPORT.md** – Section 2.8.
- **CARLOS_BRIZUELA_FULL_PROFILE_AND_RESEARCH_REPORT.md** – discussion of codon‑immune interactions.
- **RESEARCH_LIBRARY/03_REVIEW_INBOX/COMBINED_REVIEW_STRATEGY.md** – data processing workflow.
