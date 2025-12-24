# experiments/bioinformatics

> **Goal:** Validate the geometric theory on real biological data.

## Subdirectories

### 1. `codon_encoder_research/`

- **Focus:** The Genetic Code (64 -> 21 mapping).
- **Key Finding:** The genetic code is synonymous-optimized to minimize 3-adic distance.

### 2. `p-adic-genomics/`

- **Focus:** Whole genomes.
- **Key Finding:** Viral genomes are fractals. Their dimensionality is not 2D, but ~1.58D (Sierpinski triangle).

### 3. `spectral_analysis_over_models/`

- **Focus:** Analyzing the "eigenvalues" of the learned representation.
- **Key Finding:** The spectrum of the Ternary VAE matches the spectrum of natural proteins (Zipf's Law), whereas standard VAEs look like white noise.

## How to Run

Research scripts are organized by disease/topic.

```bash
# Example: Running the Spike Sentinel Analysis for SARS-CoV-2
python experiments/bioinformatics/codon_encoder_research/sars_cov_2/glycan_shield/01_spike_sentinel_analysis.py
```
