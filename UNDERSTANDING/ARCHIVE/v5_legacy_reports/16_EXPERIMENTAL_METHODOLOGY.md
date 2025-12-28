# Experimental Methodology & Data Sources

**Date:** 2025-12-27
**Scope:** Methodological details for the 98-experiment campaign (v1-v5.12).

---

## 1. Data Sources

Our analysis relies on four primary high-quality HIV datasets, totaling over 440,000 sequence records.

### A. Stanford HIV Drug Resistance Database (HIVDB)

- **Size**: >200,000 sequences (Pol gene: Protease, RT, Integrase).
- **Features**: Paired genotype-phenotype data.
- **Labeling**: Drug resistance scores (0-5 scale) for PIs, NRTIs, NNRTIs.
- **Usage**: Training the VAE reconstruction task and validating geometric drug resistance correlation.

### B. LANL CATNAP (Neutralization Assay)

- **Size**: 189,879 records.
- **Features**: HIV-1 Env pseudovirus sequences + IC50 neutralization values against broadly neutralizing antibodies (bnAbs).
- **Usage**: Validating the "Glycan Shield" geometry and immune escape predictions.

### C. Los Alamos CTL Epitopes

- **Size**: 2,115 optimal epitopes.
- **Features**: Defined genotype + HLA restriction (e.g., HLA-A*02, HLA-B*57).
- **Usage**: Mapping T-cell pressure to latent radius. (Finding: HLA-B\*57 epitopes cluster closer to the origin).

### D. Synthetic Ternary Universe

- **Size**: 19,683 operations ($3^9$ total space).
- **Features**: Complete enumeration of all possible 3-adic mutations for length-9 sequences.
- **Usage**: "Unit Testing" the VAE. We check if the model can perfectly reconstruct the entire mathematical universe before testing on biological data.

---

## 2. Preprocessing Pipeline

### Step 1: Codon Encoding

Biology is ternary. We convert raw nucleotides to base-10 indices (0-63) representing codons.

```python
# DNA -> Codon Index
"AUG" -> 35 (Methionine/Start)
"UGA" -> 14 (Opal Stop)
```

### Step 2: Sequence Alignment

- **Tool**: HMMER3 / MAFFT.
- **Reference**: HXB2 (Standard HIV-1 reference).
- **Cleaning**: Removed sequences with >5% ambiguity codes (N, R, Y).

### Step 3: P-adic Pre-calculation

For every pair of codons in the dataset, we pre-calculate the 3-adic distance.

```math
d_3(x, y) = 3^{-v_3(|x-y|)}
```

- $v_3$ measures the "depth" of the difference (Position 1, 2, or 3).
- This matrix serves as the **Ground Truth** for the P-adic Ranking Loss.

---

## 3. The Experimental Environment

### Hardware

- **GPU**: NVIDIA A100 (40GB) / RTX 4090.
- **Training Time**: ~15 mins per experiment (cyclical beta) vs 2 hours (old static beta).

### Metrics & Evaluation

We evaluated every model on three axes:

#### Axis 1: Reconstruction (Accuracy)

- **Metric**: `Accuracy %`.
- **Definition**: Can the VAE output the exact same codon sequence it took as input?
- **Passing**: >95% (Production), >60% (Research/Structure).

#### Axis 2: Structure (Spearman Correlation)

- **Metric**: `Spearman ρ`.
- **Definition**: Correlation between the _Learned Hyperbolic Distance_ and the _Ground Truth P-adic Distance_.
- **Passing**: > +0.30 (Strong biological signal).

#### Axis 3: Clinical Utility ( Downstream Tasks)

- **Metric**: `Resistance Prediction Score` (Validation).
- **Definition**: Train a simple linear classifier on top of the fixed latent space to predict Drug Resistance.
- **Passing**: > 80% F1-Score.

---

## 4. The Validation Harness (Test Suite)

Before accepted as "Working," every architectural change passed the `validate_all_phases.py` suite:

1.  **Triangle Inequality Test**: Verifies that the latent space obeys metric triangle inequality ($d(a,c) \le d(a,b) + d(b,c)$).
2.  **Gradient Orthogonality Test**: Checks if $\nabla_{recon}$ and $\nabla_{struct}$ are aligned ($cos \theta > 0$) or fighting ($cos \theta < 0$).
3.  **Boundary Collapse Check**: Ensures points aren't just stuck at $r=0$ (Gaussian collapse) or $r=1$ (Vanishing gradients).

---

## 5. Summary of how specific results were obtained

### "Position 22 Tropism"

- **Method**: Random Forest feature importance on the VAE latent dimensions.
- **Data**: 2,932 sequences with labeled Tropism (CCR5 vs CXCR4).

### "Goldilocks Zone"

- **Method**: Calculated geometric distance between Viral Epitopes and Human Self-Peptides.
- **Finding**: Autoimmune mimics reside in the "Goldilocks Zone" (Dist $\approx 0.3-0.4$)—close enough to confuse, far enough to evade negative selection.

### "Vaccine Targets"

- **Method**: Computed the "Hyperbolic Radius" of all 2,115 CTL epitopes.
- **Finding**: The most effective epitopes (low escape) cluster near the origin ($r < 0.3$), representing the "ancestral core" of the virus.
