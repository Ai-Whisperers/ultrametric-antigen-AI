# Genetic Code Discovery

**Doc-Type:** Research Index · Version 1.0 · Updated 2025-12-16

---

## Overview

The foundational discovery: the Ternary VAE's 64 natural embedding positions map perfectly to the 64 codons of the genetic code, clustering into 21 amino acid groups with 100% accuracy.

---

## The Discovery

```
VAE Embedding Space                     Genetic Code
      (learned)                          (biology)
         │                                  │
    64 positions ────────────────────► 64 codons
         │                                  │
    21 clusters  ────────────────────► 21 amino acids
         │                                  │
   Wobble pattern ───────────────────► Position 3 degeneracy
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **Cluster Accuracy** | 100% |
| **Synonymous Accuracy** | 100% |
| **Separation Ratio** | 193.5x |
| **Wobble Variance** | Highest at position 3 |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `01_bioinformatics_analysis.py` | Initial biological analysis |
| `02_genetic_code_padic.py` | P-adic genetic code structure |
| `03_reverse_padic_search.py` | Search for natural positions |
| `04_fast_reverse_search.py` | Optimized position search |
| `05_analyze_natural_positions.py` | Analyze position properties |
| `06_learn_codon_mapping.py` | Train codon→cluster encoder |

---

## Key Artifacts

| File | Description |
|------|-------------|
| `data/codon_encoder.pt` | Trained neural network (12→16→21) |
| `data/learned_codon_mapping.json` | Codon→position→cluster mapping |

---

## The Wobble Pattern

Position 3 (wobble position) shows highest variance:
- Positions 1-2: Determine amino acid identity
- Position 3: Allows synonymous variation within cluster

This matches known biology: wobble base pairing allows flexibility at the third codon position.

---

## Running

```bash
cd scripts
python 02_genetic_code_padic.py      # Analyze structure
python 06_learn_codon_mapping.py     # Train encoder
```

---

## Connection to Bioinformatics

The codon encoder trained here is used by:
- `../bioinformatics/rheumatoid_arthritis/` - HLA and citrullination analysis
- `../bioinformatics/hiv/` - Escape and resistance analysis

---

**Status:** Discovery validated, encoder production-ready
