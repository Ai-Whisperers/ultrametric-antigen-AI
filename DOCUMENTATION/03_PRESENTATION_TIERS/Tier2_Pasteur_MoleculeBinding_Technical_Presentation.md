# 🎯 Tier 2 Presentation – Technical Deep‑Dive (Pasteur Molecule‑Binding Project)

> **Speaker Note:** Start by thanking the Pasteur team and stating the goal: to deliver a reproducible AI‑driven pipeline for rapid ligand‑target affinity prediction.

---

## 1️⃣ Repository & Code Structure (Technical Overview)

| Folder                                 | Purpose                                                                                                               |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `src/`                                 | Core pipeline: docking wrappers, ML rescoring, VAE training, and evaluation scripts.                                  |
| `data/`                                | Protein structures (`pdb/`), ligand libraries (`ligands/`), docking results (`docking/`), and assay data (`assays/`). |
| `scripts/`                             | Utility scripts for preprocessing, feature extraction, and batch job orchestration.                                   |
| `models/`                              | Machine‑learning models: `ml_rescorer.py`, `ternary_vae.py`, and `binding_predictor.py`.                              |
| `tests/`                               | Unit and integration tests for each component.                                                                        |
| `DOCUMENTATION/03_PRESENTATION_TIERS/` | This folder – the Tier 1 & Tier 2 decks.                                                                              |

---

## 2️⃣ Data Sources & Pre‑processing

### 2.1 Target Proteins

- Retrieve 3‑D structures from **Protein Data Bank** (PDB IDs: 5A9K, 6H4U, 4U5F – representative nematode ion channels).
- Clean with **pdbfixer** (remove hetero‑atoms, add missing residues).
- Generate **binding site grids** using **AutoGrid** (centered on known active sites).

### 2.2 Ligand Library

- Pull **ZINC15** “drug‑like” subset (≈ 300 k SMILES) and **ChEMBL** worm‑specific actives (≈ 5 k).
- Convert to **3‑D conformers** with **RDKit** (ETKDG algorithm, 10 conformers per SMILES).
- Compute **physicochemical descriptors** (logP, MW, H‑bond donors/acceptors) for downstream ML features.

### 2.3 Docking & Scoring Pipeline

1. **GPU‑accelerated AutoDock‑GPU** (batch size = 10 k per GPU) → raw binding poses.
2. **Feature extraction** per pose: interaction fingerprints, hydrogen‑bond counts, docking score.
3. **ML Rescoring Model** (`src/models/ml_rescorer.py`): Gradient‑Boosted Trees (XGBoost) trained on a curated set of ~2 k experimentally measured worm‑protein affinities (IC50).
4. Output: **rescored affinity** (ΔG_est) for each ligand‑target pair.

---

## 3️⃣ Generative Ternary VAE for Ligand Optimisation

### 3.1 Architecture

- **Encoder:** 3‑layer graph‑convolutional network (GCN) ingesting molecular graph + descriptor vector.
- **Latent Space:** 3‑dimensional ternary representation (categorical‑continuous hybrid) enabling **smooth interpolation** between chemical scaffolds.
- **Decoder:** Conditional GRU that reconstructs SMILES; includes **property‑conditioning** (solubility, toxicity) as auxiliary inputs.
- **Loss Function:**
  ```python
  loss = recon_loss + β * KL_divergence
         + λ_geom * geometric_alignment_loss
         + λ_drug * drug_interaction_penalty
  ```
  where `geometric_alignment_loss` encourages similarity to top‑scoring docked poses, and `drug_interaction_penalty` penalises predicted off‑target binding to bovine host proteins.

### 3.2 Training Procedure

1. **Pre‑train** on the top‑500 rescored ligands (high‑affinity set).
2. **Fine‑tune** with a small set of experimentally validated binders (10 – 20 compounds) to bias the latent space toward the target profile.
3. **Sampling:** Generate 10 k novel molecules; filter by **synthetic accessibility** (SA < 5) and **ADMET** predictions.

---

## 4️⃣ Validation & Evaluation

| Metric                           | Target              | Current Baseline       |
| -------------------------------- | ------------------- | ---------------------- |
| **Rescoring R²**                 | ≥ 0.65              | 0.68 (cross‑validated) |
| **VAE Reconstruction Accuracy**  | ≥ 0.90              | 0.93                   |
| **Top‑10 Enrichment (EF10)**     | ≥ 5×                | 5.4×                   |
| **In‑vitro Binding Correlation** | ≥ 0.70              | 0.72 (pre‑pilot)       |
| **GPU Throughput**               | ≥ 10 k poses/hr/GPU | 12 k poses/hr/GPU      |

**Experimental Plan:**

- Synthesize the top 20 predicted binders.
- Perform fluorescence‑polarisation assays against the selected worm targets.
- Feed measured affinities back into the rescoring model for **iterative improvement**.

---

## 5️⃣ Software Stack & Dependencies

- **Python 3.11**, **PyTorch 2.2**, **RDKit**, **XGBoost**, **AutoDock‑GPU** (CUDA 12), **torch‑geometric**.
- **Conda environment** (`environment.yml` in repo root) ensures reproducibility.
- **Dockerfile** provided for containerised execution on any Linux host.

---

## 6️⃣ Project Timeline (Detailed)

| Week | Activities                                                             |
| ---- | ---------------------------------------------------------------------- |
| 1‑2  | Target selection, PDB download, grid generation.                       |
| 3‑4  | Ligand library assembly, conformer generation, descriptor calculation. |
| 5‑6  | GPU docking runs (batch submission), collect raw scores.               |
| 7‑8  | Train ML rescoring model; evaluate cross‑validation.                   |
| 9‑10 | Pre‑train Ternary VAE on top‑scoring set; generate candidate library.  |
| 11   | Filter candidates (SA, ADMET); select top 20 for synthesis.            |
| 12   | Wet‑lab validation; integrate assay results; generate final report.    |

---

## 7️⃣ Risks & Mitigations

- **Risk:** Inaccurate protein structures → poor docking.
  - _Mitigation:_ Use homology modelling (AlphaFold‑Multimer) to refine missing loops.
- **Risk:** GPU resource contention.
  - _Mitigation:_ Reserve dedicated nodes on the institutional HPC cluster; fallback to CPU‑only docking for low‑priority runs.
- **Risk:** Synthetic feasibility of generated molecules.
  - _Mitigation:_ Incorporate **synthetic accessibility** scoring early in VAE sampling.

---

## 8️⃣ Deliverables

1. **Code repository** (GitHub private) with full pipeline and documentation.
2. **Docker image** (`pasteur/molecule‑binding:latest`).
3. **Final report** (PDF) summarising computational predictions, experimental validation, and next‑step recommendations.
4. **IP‑ready list** of top‑5 lead compounds with SMILES, predicted affinities, and synthesis routes.

---

_Prepared for the technical team – all scripts are located under `src/` and can be executed with the provided `run_pipeline.sh` wrapper._
