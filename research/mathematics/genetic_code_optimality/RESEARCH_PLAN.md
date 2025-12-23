# Track 1: Genetic Code Optimality

**Doc-Type:** Research Plan · Version 1.0 · Updated 2025-12-23 · Author AI Whisperers

Testing whether the 64→21 genetic code degeneracy reflects optimal p-adic error-correction.

---

## Core Question

Is the genetic code an optimal (or near-optimal) sphere-packing in p-adic space?

---

## The Genetic Code Structure

```
64 codons (4^3 nucleotide triplets)
    ↓
21 outputs (20 amino acids + STOP)
    ↓
~3:1 average degeneracy
```

**degeneracy_pattern** - Not uniform; ranges from 1 (Met, Trp) to 6 (Leu, Ser, Arg)

**wobble_position** - Third codon position most degenerate (synonymous mutations cluster here)

---

## Hypotheses

### H1: Tight Synonymous Clustering

Synonymous codons (same amino acid) occupy minimal p-adic balls.

**prediction** - Intra-class p-adic variance << inter-class p-adic variance
**metric** - Ratio of within-group to between-group p-adic distances

### H2: Graceful Degradation

Single-nucleotide mutations to biochemically similar amino acids have smaller p-adic distances.

**prediction** - Correlation between p-adic distance and biochemical dissimilarity (hydrophobicity, charge, size)
**metric** - Spearman correlation with standard substitution matrices (BLOSUM62, PAM250)

### H3: Theoretical Bound Achievement

The natural code achieves or approaches theoretical packing bounds.

**prediction** - Natural code packing density > 95th percentile of random codes
**metric** - Comparison against Singleton-like bounds for ultrametric spaces

### H4: Wobble Position Optimality

Third position degeneracy minimizes error propagation.

**prediction** - Wobble mutations have systematically smaller p-adic distances than non-wobble
**metric** - Mean p-adic distance by mutation position (1st, 2nd, 3rd)

---

## Experimental Design

### Phase 1: Baseline Measurements

```
For each of 64 codons:
    1. Encode to 16-dim Poincare ball via trained encoder
    2. Record embedding coordinates
    3. Compute pairwise p-adic distances (64x64 matrix)
    4. Group by amino acid (21 groups)
```

### Phase 2: Packing Metrics

| Metric | Computation |
|--------|-------------|
| Covering radius | max_codon(min_AA(d(codon, AA_centroid))) |
| Packing radius | min(inter-class distances) / 2 |
| Packing density | Volume of codon balls / Total space volume |
| Kissing number | Average neighbors within packing radius |

### Phase 3: Random Code Comparison

Generate N=10,000 random 64→21 mappings preserving degeneracy structure:
- Same number of codons per amino acid
- Random assignment within constraint

Compute packing metrics for each, establish null distribution.

### Phase 4: Theoretical Bounds

Derive or adapt known bounds:
- Singleton bound (ultrametric analog)
- Hamming bound (ultrametric analog)
- Plotkin bound (ultrametric analog)

Compare natural code against bounds.

---

## Data Requirements

**encoder_weights** - `research/genetic_code/data/codon_encoder_3adic.pt`
**codon_table** - Standard genetic code (NCBI translation table 1)
**biochemical_properties** - Amino acid physicochemical properties
**substitution_matrices** - BLOSUM62, PAM250 for validation

---

## Expected Outputs

```
genetic_code_optimality/
├── RESEARCH_PLAN.md          # This document
├── scripts/
│   ├── 01_encode_all_codons.py
│   ├── 02_compute_packing_metrics.py
│   ├── 03_generate_random_codes.py
│   ├── 04_theoretical_bounds.py
│   └── 05_statistical_analysis.py
├── results/
│   ├── codon_embeddings.npy
│   ├── pairwise_distances.npy
│   ├── packing_metrics.json
│   ├── null_distribution.npy
│   └── figures/
└── FINDINGS.md               # Final analysis
```

---

## Success Criteria

| Outcome | Interpretation |
|---------|----------------|
| Natural code > 99th percentile | Strong evidence for optimality |
| Natural code > 95th percentile | Moderate evidence for optimality |
| Natural code > 50th percentile | Weak/no evidence for optimality |
| Natural code < 50th percentile | Evidence against optimality hypothesis |

---

## Connections to Literature

**freeland_hurst_2004** - "The Genetic Code is One in a Million" - showed code minimizes impact of errors
**itzkovitz_2007** - Overlapping codes and information density
**novozhilov_2007** - Circular code hypothesis
**our_contribution** - First test using p-adic/hyperbolic geometry framework

---

## Open Questions

- Does the encoder's training on biological data bias results?
- Should we use untrained (random init) encoder as additional null?
- How to handle STOP codon (not an amino acid)?
- What is the correct volume measure in Poincare ball?
