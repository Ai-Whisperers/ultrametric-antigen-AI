# Spectral Analysis

**Doc-Type:** Research Index · Version 1.0 · Updated 2025-12-16

---

## Overview

Mathematical analysis connecting the Ternary VAE's learned embedding space to p-adic geometry, spectral theory, and the Riemann hypothesis.

---

## Key Discoveries

### 1. Radial Exponent (r^0.63)
The embedding distances follow a power-law scaling with exponent ~0.63, suggesting deep geometric structure.

### 2. Prime Capacity Structure
The 21 natural clusters in the embedding space match the degeneracy structure of the genetic code (unexpected emergence).

### 3. P-Adic Ultrametric
The learned space satisfies the ultrametric inequality: d(A,B) ≤ max(d(A,C), d(B,C))

---

## Scripts

| Script | Purpose |
|--------|---------|
| `01_extract_embeddings.py` | Extract latent space from trained VAE |
| `02_compute_spectrum.py` | Compute eigenvalue spectrum of distance matrix |
| `02_compute_spectrum_fast.py` | Optimized spectrum computation |
| `03_compare_zeta.py` | Compare eigenvalues to Riemann zeta zeros |
| `04_padic_spectral_analysis.py` | P-adic spectral analysis |
| `05_exact_padic_analysis.py` | Exact p-adic arithmetic |
| `07_adelic_analysis.py` | Adelic structure analysis |
| `08_alternative_spectral_operators.py` | Alternative operator constructions |
| `09_binary_ternary_decomposition.py` | Binary/ternary number theory |
| `10_semantic_amplification_benchmark.py` | Semantic structure tests |
| `11_variational_orthogonality_test.py` | Orthogonality verification |

---

## Running

```bash
cd scripts
python 01_extract_embeddings.py
python 02_compute_spectrum.py
python 03_compare_zeta.py
```

---

## Connection to Genetic Code

The spectral structure discovered here provides the mathematical foundation for the genetic code mapping in `../genetic_code/`.

---

**Status:** Mathematical analysis complete, Riemann connection documented
