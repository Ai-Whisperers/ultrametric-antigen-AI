# Agent Implementation Guide: Alejandra Rojas (Arbovirus MVP)

> **Objective:** Build the "Dengue Serotype Forecaster" and "Primer Stability Map".

## 1. Context

Alejandra Rojas (IICS-UNA) manages Arboviral surveillance. She needs:

1.  **Forecasting:** Predicting which Dengue Serotype (1-4) will dominate next season.
2.  **Diagnostics:** Designing primers that won't fail when the virus mutates.

## 2. Infrastructure Status

- **Sliding Window:** `scripts/analysis/sliding_window_embedder.py` (Exists & Verified).
- **Data Source:** NCBI Virus (Requires download of Paraguayan sequences).

### Step A: Data Ingestion (NCBI)

1.  **Action:** Download chronological Dengue genomes.
    ```bash
    datasets download virus genome taxon 12637 --geo-location "Paraguay" --include genome
    ```
2.  **Action:** Unzip and clean FASTA headers to extract Collection Date.

### Step B: Generate Trajectories

1.  **Action:** Run the sliding window embedder.
    ```bash
    python scripts/analysis/sliding_window_embedder.py --input data/raw/dengue_paraguay.fasta --output data/processed/dengue_trajectories.pt --window 300
    ```
2.  **Result:** A dataset where each virus is a "Path" in hyperbolic space.

### Step C: The "Forecaster"

1.  **File:** `notebooks/partners/rojas_serotype_forecast.ipynb`.
2.  **Logic:**
    - Plot Centroids of DENV-1, DENV-2, DENV-4 over time (2011-2024).
    - Calculate the "Hyperbolic Momentum" (Velocity Vector).
    - **Prediction:** Project the vector forward 6 months. Does it move towards a "Severe" cluster?

### Step D: The "Primer Map"

1.  **File:** `scripts/analysis/primer_stability_scanner.py`.
2.  **Logic:**
    - Calculate Variance of Hyperbolic position for every 20nt window across all 2024 samples.
    - Identify windows with Variance < Threshold.
    - Export these sequences as "Safe Primer Candidates".

## 4. Deliverable

- **Interactive Dashboard:** A Plotly/Dash map showing the "Viral Vector" moving through time.
- **Primer List:** CSV of top 10 stable regions for rRT-PCR design.
