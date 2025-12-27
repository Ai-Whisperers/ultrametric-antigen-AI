# Partnership Implementation Plan: Expert MVP Execution

> **Actionable Roadmap for "The Expert MVP Strategy"**

This document details the technical implementation steps to execute the MVP ideas for our three key partners. It focuses on **Data Acquisition** (via API/Ingest), **Analysis Implementation**, and **Deliverable Generation**.

---

## 📅 Phased Execution Overview

| Phase | Partner             | Focus Domain                  | Key Deliverable                        | Est. Timeline |
| :---- | :------------------ | :---------------------------- | :------------------------------------- | :------------ |
| **1** | **Carlos Brizuela** | Antimicrobial Peptides (AMPs) | `AMP_Hyperbolic_Navigator.ipynb`       | Week 1        |
| **2** | **Dr. José Colbes** | Protein Optimization          | `Geometric_Rotamer_Scoring_Report.pdf` | Week 2        |
| **3** | **Alejandra Rojas** | Arboviruses (Dengue/Zika)     | `Dengue_Serotype_Forecast_Dashboard`   | Week 3        |

---

## 🧬 Phase 1: Carlos Brizuela (Antimicrobial Peptides)

### 1.1 Data Acquisition (StarPepDB)

- **Source:** StarPepDB (Graph Database of AMPs).
- **Existing Tool:** `scripts/ingest/ingest_starpep.py` (Verify coverage).
- **Action:** Ensure the ingest script captures **Hemolytic Activity** and **MIC (Minimum Inhibitory Concentration)** fields, as these are critical for the multi-objective optimization MVP.

### 1.2 New Implementations

- **Script:** `scripts/analysis/amp_hyperbolic_navigator.py`
  - **Logic:**
    1.  Load `starpep_data.json`.
    2.  Encode sequences into 5D Hyperbolic Space using `models.ternary_vae`.
    3.  Compute "Geodesic Paths" from toxic AMPs to non-toxic clusters.
- **Script:** `scripts/optimization/latent_nsga2.py`
  - **Logic:** Implement NSGA-II algorithm where the decision variables are the _latent coordinates_ $(z_1, ..., z_5)$ rather than the discrete amino acids.

---

## 🧩 Phase 2: Dr. José Colbes (Protein Optimization)

### 2.1 Data Acquisition (RCSB PDB)

We need high-resolution protein structures to calculate side-chain angles (chi angles).

- **Source:** RCSB PDB Data API.
- **API Strategy:**
  - **Endpoint:** `https://data.rcsb.org/rest/v1/core/entry/{pdb_id}` (Metadata) + File Download.
  - **Library:** `Biopython` (`Bio.PDB`) or `biotite`.
  - **Target Dataset:** "Hard-to-fold" benchmark set (e.g., CASP targets or specific proteins from his 2016 paper).

### 2.2 New Implementations

- **Script:** `scripts/ingest/ingest_pdb_structures.py`
  - **Command:** `python scripts/ingest/ingest_pdb_structures.py --pdb_ids "1CRN,1TIM,..."`
  - **Logic:** Download `.cif` or `.pdb` files. Parsers atoms to extract `N, CA, C, CB, ...` coordinates.
- **Script:** `scripts/analysis/rotamer_padic_score.py`
  - **Logic:**
    1.  Calculate side-chain dihedral angles ($\chi_1, \chi_2$).
    2.  Compute the **3-adic distance** of the amino acid sequence context.
    3.  Correlate "Rare/Unstable Rotamers" with "High P-adic Shift".

---

## 🦟 Phase 3: Alejandra Rojas (Arboviruses)

### 3.1 Data Acquisition (NCBI Virus)

We need complete genomes of Dengue (DENV-1, 2, 3, 4), Zika, and Chikungunya, specifically chronological series from Paraguay/South America.

- **Source:** NCBI Virus / GenBank.
- **API Strategy:**
  - **Tool:** `NCBI Datasets` CLI (Recommended) or `Bio.Entrez`.
  - **Query:** `Organism: Dengue virus (taxid:12637)`, `Geo Location: Paraguay/South America`, `Collection Date: 2011-2025`.
  - **Command:**
    ```bash
    datasets download virus genome taxon 12637 --geo-location "Paraguay" --include genome,biosample
    ```

### 3.2 New Implementations

- **Script:** `scripts/ingest/ingest_arboviruses.py`
  - **Logic:** Wrapper around `datasets` CLI or Entrez to fetch FASTA sequences and metadata (date, location).
- **Script:** `scripts/analysis/arbovirus_hyperbolic_trajectory.py`
  - **Logic:**
    1.  Embed genomes into Hyperbolic Space.
    2.  Group by `Date` (Month/Year).
    3.  Calculate the "Velocity Vector" of the clade centroid.
    4.  Predict next-month trajectory.
- **Script:** `scripts/analysis/primer_stability_scanner.py`
  - **Logic:** Scan genome with rolling window. Identify regions with minimal hyperbolic movement over the last 10 years.

---

## 🛠 Shared Infrastructure Requirements

1.  **Bioinformatics Env:** Ensure `biopython`, `numpy`, `scipy` are installed.
2.  **NCBI Tooling:** Install `ncbi-datasets-cli` via `winget` or Conda.
3.  **Visualization:** Update `dash` dashboard to support "Protein 3D View" (for Colbes) and "Geographic Map" (for Rojas).

---

_Plan Pending User Approval for Execution._
