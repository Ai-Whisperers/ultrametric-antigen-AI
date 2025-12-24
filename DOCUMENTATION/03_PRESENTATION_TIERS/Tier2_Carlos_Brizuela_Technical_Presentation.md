# Tier 2 Presentation – Technical Deep‑Dive

**Audience:** Researchers, developers, and domain experts who will implement the Ternary VAE components.

---

## 1. Repository Structure (Technical)

| Folder                                     | Purpose                                                                                                                                  |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `src/`                                     | Core VAE code, codon‑encoder, custom loss functions, training scripts.                                                                   |
| `data/`                                    | Raw and processed datasets (HIV geometric vaccine models, drug‑interaction matrices, peptide libraries, codon‑p‑adic similarity tables). |
| `DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/` | Theory, bibliography, and research library – the scientific backbone.                                                                    |
| `DOCUMENTATION/03_PRESENTATION_TIERS/`     | Presentation assets (this Tier 1 & Tier 2 markdown).                                                                                     |
| `scripts/`                                 | Utility scripts for data preprocessing, model evaluation, and report generation.                                                         |

## 2. Consolidated Knowledge Source

- **File:** `CARLOS_BRIZUELA_FULL_PROFILE_AND_RESEARCH_REPORT.md`
  - Contains Carlos Brizuela’s full academic profile, curated publication list, and a **comprehensive research report** focused on HIV‑related topics.
  - **Key validated sections** (see Tier 1 for summary): geometric vaccine design, drug‑interaction penalties, nucleoside‑analogue vaccine concepts, antibiotic‑retroviral synergy, and p‑adic codon‑space exploration.

## 3. Technical Insights for Implementation

### 3.1 Geometric Vaccine Design

- **Reference:** Ferritin, mi3, VLP scaffolds (Section *Geometric HIV Vaccines*).
- **Implementation Idea:** Encode scaffold symmetry as a regularization term in the VAE latent space using a **p‑adic distance matrix** that penalizes deviations from native‑like antigen orientations.
- **Data Needed:** 3‑D coordinates of scaffold‑antigen complexes (available in `data/geometric_vaccine/`).

### 3.2 Drug‑Interaction Penalty

- **Reference:** Lenacapavir ↔ Sertraline interaction (CYP3A4 inhibition).
- **Implementation Idea:** Add a **scalar penalty** to the loss proportional to predicted CYP3A4 inhibition strength for each drug‑pair in the training batch.
- **Data Needed:** Interaction matrix `data/drug_interactions/cyp3a4_inhibition.csv`.

### 3.3 Nucleoside‑Analogue Vaccine Concept

- **Reference:** FIV + Tenofovir DNA vaccine.
- **Implementation Idea:** Treat nucleoside‑analogue sequences as **auxiliary conditioning vectors** that guide the decoder toward immunogenic motifs.
- **Data Needed:** Tenofovir‑derived peptide libraries `data/tenofovir_peptides/`.

### 3.4 Antibiotic‑Retroviral Synergy

- **Reference:** β‑lactams, macrolides, fluoroquinolones modulating viral replication.
- **Implementation Idea:** Encode antibiotic exposure as **binary feature flags** in the model input; evaluate their impact on latent representations during training.
- **Data Needed:** Antibiotic‑retroviral effect table `data/antibiotic_effects.csv`.

### 3.5 p‑Adic Codon‑Space Exploration

- **Reference:** p‑adic metric for codon similarity (Section *Codon‑Space Exploration*).
- **Implementation Idea:** Build a **codon similarity matrix** using the p‑adic distance; feed it to a **codon‑encoder** layer that maps nucleotide triplets to a continuous embedding.
- **Data Needed:** Pre‑computed matrix `data/codon_padic_matrix.npy`.

## 4. Proposed Development Roadmap (30 days)

| Day   | Milestone                                                                                  |
| ----- | ------------------------------------------------------------------------------------------ |
| 1‑3   | Pull the latest repo, verify folder layout, run existing unit tests.                       |
| 4‑7   | Ingest all datasets listed above; write preprocessing scripts (`scripts/preprocess_*.py`). |
| 8‑12  | Implement **GeometricLoss** module (`src/losses/geometric_loss.py`).                       |
| 13‑16 | Implement **DrugInteractionPenalty** (`src/losses/drug_interaction.py`).                   |
| 17‑20 | Add **CodonEncoder** layer (`src/encoders/codon_encoder.py`).                              |
| 21‑24 | Integrate antibiotic feature flags into the training pipeline (`src/train.py`).            |
| 25‑27 | End‑to‑end training on a reduced dataset; generate validation metrics.                     |
| 28‑30 | Write a technical report and update the presentation deck (Tier 2).                        |

## 5. Validation & Verification Plan

- **Unit Tests:** Each new loss/encoder gets a dedicated test in `tests/unit/`.
- **Integration Test:** Train on a synthetic dataset (10 k samples) and verify that loss values decrease as expected.
- **Scientific Validation:** Compare generated antigen geometry against the reference structures (RMSD < 2 Å).
- **Performance Benchmark:** Ensure training time increase ≤ 15 % compared to baseline VAE.

---

**Prepared for the development team – all source data paths are relative to the repository root.**
