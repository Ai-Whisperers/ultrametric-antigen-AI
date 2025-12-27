# Agent Implementation Guide: Dr. José Colbes (Protein Optimization MVP)

> **Objective:** Build the "Geometric Rotamer Scoring" function.

## 1. Context

Dr. Colbes works on the "Side Chain Packing Problem". Current heuristics (SCWRL) are stuck in local optima.
We propose a **Geometric Energy Term ($E_{geom}$)** derived from our P-adic metric to penalize unstable rotamers that "look" optimized in Euclidean space but are fractured in Hyperbolic space.

## 2. Infrastructure Status

- **Model:** `TernaryVAEV5_11` (Requires re-training or transfer learning for Rotamer angles).
- **Missing:** `ingest_pdb_rotamers.py`.

## 3. Step-by-Step Implementation for Agents

### Step A: Data Ingestion (PDB)

1.  **Action:** Create `scripts/ingest/ingest_pdb_rotamers.py`.
    - **Input:** List of PDB IDs (from `data/raw/colbes_targets.txt`).
    - **Logic:** Download PDB -> Extract Side Chain Dihedrals ($\chi_1, \chi_2, \chi_3, \chi_4$).
    - **Output:** Tensor `[N_Residues, 4]` (Angles normalized to $-\pi, \pi$).

### Step B: The "Geometric Filter" Experiment

1.  **Hypothesis:** Rare rotamers have "High Hyperbolic Velocity" (instability).
2.  **Action:** Create `scripts/analysis/rotamer_stability.py`.
    - Load the Rotamer Tensor.
    - Embed into Hyperbolic Space.
    - Compute Distance to "Rotamer Cluster Centroids".
    - **Correlation Test:** Compare `HyperbolicDistance` vs `RosettaEnergyScore`.

### Step C: The "Scoring Function" Deliverable

1.  **File:** `notebooks/partners/colbes_scoring_function.ipynb`.
2.  **Content:**
    - Show a specific case (e.g., T4 Lysozyme core).
    - Highlight a residue where Rosetta says "Stable" but our VAE says "Unstable".
    - Prove (via literature or MD simulation logs) that this residue _is_ actually problematic.

## 4. Deliverable

- A PDF Report: "The Hidden Geometry of Rotamer Packing: A 3-Adic Perspective".
