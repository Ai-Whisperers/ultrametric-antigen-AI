# Agent Implementation Guide: Carlos Brizuela (AMP MVP)

> **Objective:** Build the "Hyperbolic AMP Navigator" and "Toxicity Optimizer".

## 1. Context

Carlos Brizuela specializes in Antimicrobial Peptides (AMPs). We are building tools to:

1.  Visualize AMPs in our Hyperbolic Latent Space.
2.  Optimize AMPs to be highly active but low toxicity using his NSGA-II algorithm adapted to our VAE.

## 2. Infrastructure Status

- **Ingest Script:** `scripts/ingest/ingest_starpep.py` (Exists).
- **Toxicity Trainer:** `scripts/training/train_toxicity_regressor.py` (Exists & Verified).
- **Model:** `src/models/ternary_vae.py` (Exists).

## 3. Step-by-Step Implementation for Agents

### Step A: Data Ingestion

1.  **Action:** Run ingestion on the full StarPepDB dataset.
    ```bash
    python scripts/ingest/ingest_starpep.py --input data/raw/starpep.csv --output data/processed/starpep_hyperbolic.pt
    ```
2.  **Verification:** Ensure output tensor has shape `[N, 16]` and metadata contains `activity` and `hemolytic` columns.

### Step B: Train Toxicity Predictor

1.  **Action:** Train the regressor on the processed data.
    ```bash
    python scripts/training/train_toxicity_regressor.py --input data/processed/starpep_hyperbolic.pt --epochs 50
    ```
2.  **Goal:** Achieve Validation Loss < 0.1 (Binary Cross Entropy).

### Step C: Build the "Navigator" Notebook

1.  **File:** Create `notebooks/partners/brizuela_amp_navigator.ipynb`.
2.  **Content:**
    - Load VAE Encoder and Toxicity Regressor.
    - Plot the 2D interaction (UMAP of 16D latent) of "Toxic vs Non-Toxic" peptides.
    - **Interactive Widget:** Allow user to input a sequence, see its position, and "nudge" it towards the safe cluster.

### Step D: Zero-Shot Optimization Experiment

1.  **File:** Create `scripts/optimization/latent_nsga2.py`.
2.  **Logic:**
    - Define Objective 1: Maximize `VAE_Reconstruction_Likelihood` (Stability).
    - Define Objective 2: Minimize `Toxicity_Regressor_Output` (Safety).
    - Run NSGA-II on the 16 Latent Dimensions.
    - Decode Pareto Front points back to sequences.

## 4. Deliverable

- A ZIP file containing the `amp_navigator` notebook and the `best_peptides.csv` from Step D.
