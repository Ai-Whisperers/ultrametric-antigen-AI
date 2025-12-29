# Experimental Evidence: The Proof of Concept

**98 Experiments, 200,000+ Sequences, One Conclusion**

---

## 1. Executive Summary of Results

Our findings validate the **Ternary VAE with P-adic Geometry (v5.12)** as the optimal architecture for biological sequence modeling.

| Metric                          | Result      | vs Baseline (Euclidean)         |
| :------------------------------ | :---------- | :------------------------------ |
| **Structure (Spearman $\rho$)** | **+0.4504** | +0.37 (Baseline)                |
| **Accuracy (Reconstruction)**   | **96.7%**   | 62% (Baseline)                  |
| **Drug Resist. Prediction**     | **+216%**   | Using ESM-2 Embeddings          |
| **Vaccine Targets**             | **387**     | Identified via Goldilocks Score |

---

## 2. The Comprehensive Experiment Phase (Run Catalog)

We conducted **98 controlled experiments** across 4 phases to isolate the best components.

### Phase 1: Architecture Search

- **Winner**: `TropicalVAE` (Accuracy) & `SimpleHyperbolic` (Structure).
- **Finding**: Complex "Dual-VAEs" failed due to gradient conflicts. Simple models with strong geometry won.

### Phase 2: Loss Function Landscape

- **Winner**: `Triplet P-adic` + `Monotonic Radial`.
- **Finding**: Radial loss acts as a scaffold (Global Tree), while Triplet loss handles local clustering.

### Phase 3: Training Dynamics (The Breakthrough)

- **Winner**: **Cyclical Beta Schedule** (0.0 $\to$ 0.1 $\to$ 0.0).
- **Finding**: Constant Beta leads to collapse. The "heartbeat" schedule allows the model to alternate between memorization and organization.

---

## 3. Deep Dive: HIV Discoveries (from 200k sequences)

Using the Stanford HIVDB and LANL CATNAP datasets, the model independently rediscovered known biological facts and found new ones.

### A. Novel Tropism Determinant

- **Discovery**: **Position 22** is the top predictor of CCR5 vs CXCR4 tropism.
- **Status**: Novel finding. Conventional wisdom focuses on the V3 loop (Pos 11, 25).
- **Impact**: Suggests allosteric control mechanism outside the loop.

### B. Geometric Drug Resistance

- **Correlation**: $r=0.41$ between **Hyperbolic Radius** and Drug Resistance.
- **Mechanism**: Resistant variants live at the "boundary" (high radius), while sensitive wildtypes live at the "origin".
- **Clinical Utility**: We can predict resistance potential purely from geometric coordinates.

### C. The "Goldilocks Zone" for Autoimmunity

- **Finding**: Epitopes with $d \approx 0.33$ from human self-peptides are the most dangerous.
- **Why**: Too close = Tolerance. Too far = Clearance. Middle = Autoimmunity/Mimicry.

---

## 4. Deep Dive: ESM-2 Language Models (The "Supercharger")

In late 2024, we integrated Meta's **ESM-2** (650M) protein language model embeddings.

### Results on Problem Drugs (TPV, DRV, DTG)

| Drug                   | Baseline Improvement | ESM-2 Improvement | Relative Gain |
| :--------------------- | :------------------- | :---------------- | :------------ |
| **Darunavir (DRV)**    | +0.093               | **+0.294**        | **+216%**     |
| **Dolutegravir (DTG)** | +0.173               | **+0.311**        | **+80%**      |
| **Tipranavir (TPV)**   | +0.079               | **+0.138**        | **+75%**      |

**Conclusion**: Learned embeddings (ESM-2) vastly outperform One-Hot encoding for resistance prediction, likely because they capture evolutionary constraints implicitly.

---

## 5. Artifacts & Code

- **Catalog**: `results/experiment_catalog.csv`
- **Validation Suite**: `tests/validate_all_phases.py`
- **ESM Integration**: `scripts/api_integration/esm2_embedder.py`
