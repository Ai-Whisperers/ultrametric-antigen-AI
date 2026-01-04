# DENV-4 P-adic Hyperbolic Integration Analysis

**Doc-Type:** Research Analysis · Version 1.0 · 2026-01-04 · AI Whisperers

---

## Executive Summary

This analysis integrates phylogenetic data from 270 DENV-4 genomes with p-adic/hyperbolic
codon encoding to identify conserved regions using a fundamentally different approach than
traditional Shannon entropy.

### Key Finding

**Shannon entropy and hyperbolic variance identify DIFFERENT conserved regions:**

| Metric | Best Region | Variance/Entropy | Gene |
|--------|-------------|------------------|------|
| **Hyperbolic Variance** | Position 2400 | 0.0183 | E (Envelope) |
| Shannon Entropy | Position 9908 | 0.29 | NS5 |

The p-adic codon encoder identifies structural proteins (E, NS1) as most conserved at
the codon level, while Shannon entropy identifies NS5 polymerase regions.

---

## Why This Matters

### Shannon Entropy Limitations

Traditional primer design uses Shannon entropy to find regions with low nucleotide
variability. However, this approach:

- Treats all nucleotide substitutions equally
- Ignores codon degeneracy (synonymous mutations)
- Misses structural conservation at the amino acid level

### Hyperbolic Variance Advantages

The p-adic codon encoder embeds codons into hyperbolic space where:

- Synonymous codons cluster together
- Amino acid properties are preserved in the geometry
- Conservation reflects functional constraints, not just sequence identity

---

## Results: 270 DENV-4 Genomes

### Region Analysis

| Region | Position | Hyp. Variance | Interpretation |
|--------|----------|---------------|----------------|
| NS5_conserved | 9908-9933 | 0.0718 | **Higher** variance than expected |
| PANFLAVI_FU1 | 9007-9033 | 0.0503 | Pan-flavivirus target |
| PANFLAVI_cFD2 | 9196-9222 | 0.0680 | Pan-flavivirus target |
| NS5_5prime | 9000-9100 | 0.0238 | Moderate |
| NS5_middle | 9500-9600 | 0.0282 | Moderate |
| NS5_3prime | 10000-10100 | 0.0221 | Low (good candidate) |

**Surprising finding:** The NS5 region previously identified as "conserved" by Shannon
entropy (position 9908) has HIGHER hyperbolic variance (0.0718) than other regions.

### Top Primer Candidates (Hyperbolic Metric)

| Rank | Position | Gene Region | Hyp. Variance | Notes |
|------|----------|-------------|---------------|-------|
| 1 | 2400 | E (Envelope) | 0.0183 | Structural protein |
| 2 | 3000 | NS1 | 0.0207 | Immune target |
| 3 | 9600 | NS5 | 0.0222 | Polymerase |
| 4 | 600 | prM/M | 0.0235 | Membrane protein |
| 5 | 0 | 5'UTR | 0.0252 | Untranslated |
| 6 | 7800 | NS4B/NS5 | 0.0253 | Junction |
| 7 | 2700 | NS1 | 0.0260 | Immune target |
| 8 | 9000 | NS5 | 0.0271 | Polymerase |
| 9 | 8700 | NS5 | 0.0272 | Polymerase |
| 10 | 1200 | prM | 0.0272 | Membrane |

### Gene Region Assignments (DENV-4 RefSeq NC_002640)

| Gene | Start | End |
|------|-------|-----|
| 5'UTR | 1 | 101 |
| C (Capsid) | 102 | 476 |
| prM/M | 477 | 976 |
| E (Envelope) | 977 | 2471 |
| NS1 | 2472 | 3527 |
| NS2A | 3528 | 4184 |
| NS2B | 4185 | 4574 |
| NS3 | 4575 | 6431 |
| NS4A | 6432 | 6806 |
| NS4B | 6807 | 7558 |
| NS5 | 7559 | 10271 |
| 3'UTR | 10272 | 10649 |

---

## Genome-wide Variance Profile

| Statistic | Value |
|-----------|-------|
| Windows scanned | 36 |
| Window size | 75 nt (25 codons) |
| Step size | 300 nt |
| Variance range | 0.0183 - 0.0573 |
| Mean variance | 0.0347 |

### Variance Distribution by Gene

| Region | Mean Variance | Interpretation |
|--------|---------------|----------------|
| E (Envelope) | 0.018-0.026 | **Most conserved** |
| NS1 | 0.021-0.026 | Very conserved |
| prM/M | 0.024-0.045 | Variable |
| NS5 | 0.022-0.042 | Mixed |
| NS3 | 0.029-0.049 | High variance |

---

## Implications for DENV-4 Detection

### Why 86.7% of Sequences Can't Be Targeted

Previous Shannon entropy analysis found that 86.7% of DENV-4 sequences cannot be
targeted with consensus primers because:

1. High nucleotide diversity (71.7% mean identity vs 98% for other serotypes)
2. No conserved windows across all clades at the nucleotide level

### P-adic Perspective

The hyperbolic variance analysis reveals:

1. **Structural proteins are more conserved at codon level** than NS5
2. The E gene (position 2400) has 4x lower variance than NS5_conserved
3. This suggests functional constraints maintain codon structure even when
   nucleotide sequences diverge

### Recommended Strategy

**For universal DENV-4 detection:**

1. Target E gene region (position 2400) - lowest hyperbolic variance
2. Use degenerate primers that preserve codon structure
3. Combine with pan-flavivirus primers for redundancy

**For clade-specific detection:**

1. Continue using NS5 region for Clade_E.3.2 (13.3% coverage)
2. Use tiered detection (Tier 1: specific, Tier 2: pan-flavi + sequencing)

---

## Methodology

### TrainableCodonEncoder Architecture

```
Input: 12-dim one-hot (4 bases × 3 positions)
  ↓
MLP: 12 → 64 → 64 → 16 (with LayerNorm, SiLU, Dropout)
  ↓
exp_map_zero: Tangent space → Poincaré ball
  ↓
Output: 16-dim hyperbolic embedding
```

### Loss Components During Training

1. **Radial Loss:** Target radius by hierarchy level
2. **P-adic Structure:** Hyperbolic distances match p-adic distances
3. **Cohesion:** Synonymous codons cluster
4. **Separation:** Different AAs separate
5. **Property:** AA distances correlate with physicochemical properties

### Hyperbolic Variance Calculation

```python
# For each genome window:
embeddings = encoder(codon_indices)  # (n_codons, 16)
centroid = embeddings.mean(dim=0)
variance = poincare_distance(embeddings, centroid).mean()

# Cross-sequence variance:
all_window_embs = [embed(seq[start:end]) for seq in sequences]
cross_var = mean_distance_from_centroid(all_window_embs)
```

---

## Validation Status

| Aspect | Status |
|--------|--------|
| Encoder trained | Yes (LOO Spearman 0.61 on DDG) |
| Hyperbolic distances | Validated against p-adic structure |
| Conservation signal | Identified (E gene position 2400) |
| Primer design | Pending wet-lab validation |

---

## Files Generated

```
results/padic_integration/
├── padic_integration_results.json   # Full analysis data
└── PADIC_INTEGRATION_REPORT.md      # This report
```

---

## References

1. TrainableCodonEncoder: `src/encoders/trainable_codon_encoder.py`
2. P-adic geometry: `src/geometry/poincare.py`
3. DENV-4 sequences: `data/cache/denv4_sequences.json` (270 genomes)
4. Phylogenetic analysis: `results/phylogenetic/`

---

*Analysis performed with p-adic codon encoder framework*
*IICS-UNA Arbovirus Surveillance Program*
