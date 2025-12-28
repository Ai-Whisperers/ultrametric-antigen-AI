# Archived Euclidean Codon Encoder (Deprecated)

**Archived:** 2025-12-27

## Why Archived

These files implement a **Euclidean distance-based** codon encoder that was superseded by the **p-adic/hyperbolic** approach. The Euclidean version does not preserve the ultrametric (hierarchical) structure of the genetic code.

## Key Difference

| Approach | Distance Metric | Structure Preserved |
|----------|-----------------|---------------------|
| Euclidean (archived) | `F.pairwise_distance` | Flat, no hierarchy |
| P-adic/Hyperbolic (current) | `poincare_distance` | 3-adic ultrametric hierarchy |

## Archived Contents

- `06_learn_codon_mapping.py` - Euclidean-based codon learner script
- `learned_codon_mapping.json` - Mapping output from Euclidean training
- `euclidean_and_clustering_optimized/` - Euclidean optimization results
- `legacy/` - Earlier legacy implementations

## Current Implementation

Use these files instead:
- **Trainer:** `scripts/09_train_codon_encoder_3adic.py` (Poincare geodesic distances)
- **Checkpoint:** `data/codon_encoder_3adic.pt`
- **Mapping:** `data/codon_mapping_3adic.json`
- **Core Encoder:** `src/encoders/codon_encoder.py` (p-adic initialization + metrics)

## Technical Notes

The Euclidean approach achieved high cluster accuracy (100%) but did not respect the biological constraint that mutations are hierarchical (first base > second base > third base importance). The p-adic metric naturally encodes this via:

```
d(codon1, codon2) = 1.0      if first bases differ
                 = 1/3      if second bases differ (first same)
                 = 1/9      if third bases differ (first two same)
```

This matches the wobble position biology where third-base mutations are often synonymous.
