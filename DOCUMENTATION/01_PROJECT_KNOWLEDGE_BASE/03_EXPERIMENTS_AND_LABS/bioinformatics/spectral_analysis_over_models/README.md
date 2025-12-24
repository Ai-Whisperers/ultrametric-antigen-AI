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

## Laboratory (Code)

Active research code has been moved to:
`research/bioinformatics/spectral_analysis_over_models/scripts/`

| Script                     | Purpose                                        |
| -------------------------- | ---------------------------------------------- |
| `01_extract_embeddings.py` | Extract latent space from trained VAE          |
| `02_compute_spectrum.py`   | Compute eigenvalue spectrum of distance matrix |
| ...                        | ...                                            |

## Running

Navigate to the research folder:

```bash
cd research/bioinformatics/spectral_analysis_over_models/scripts
python 01_extract_embeddings.py
```

---

## Connection to Genetic Code

The spectral structure discovered here provides the mathematical foundation for the genetic code mapping in `../genetic_code/`.

---

**Status:** Mathematical analysis complete, Riemann connection documented
