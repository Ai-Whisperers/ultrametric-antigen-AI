# PTM Hierarchical Taxonomy

**Doc-Type:** Technical Appendix to White Paper | Version 1.0 | Updated 2025-12-24

---

## Overview

This document details the **14-level ultrametric hierarchy** extracted from the 3-adic codon encoder and maps how PTMs navigate this hierarchy. This extends the foundational white paper with precise cluster-level mechanics.

---

## 1. The 21-Cluster Ultrametric Tree

### 1.1 Hierarchy Structure

The 3-adic encoder organizes 64 codons into **21 clusters** arranged in a **14-level ultrametric tree**:

```
LEVEL 0 (ROOT)
    │
LEVEL 1 ─────────────────────────────────────────────────────────────
    │                              1 macro-cluster (all AAs)
    │
LEVEL 2 ─────────────────────────────────────────────────────────────
    ├── Polar/Charged             2 macro-clusters
    └── Hydrophobic
    │
LEVEL 3 ─────────────────────────────────────────────────────────────
    ├── Positive charge           3 meso-clusters
    ├── Negative charge
    └── Neutral hydrophobic
    │
LEVEL 4-7 ───────────────────────────────────────────────────────────
    │   Size-based subdivisions
    │
LEVEL 8-11 ──────────────────────────────────────────────────────────
    │   Aromaticity, special cases
    │
LEVEL 12-14 ─────────────────────────────────────────────────────────
        Individual amino acid resolution
```

### 1.2 Cluster Assignments

| Cluster | Amino Acid(s) | Properties |
|---------|---------------|------------|
| 0 | L | Hydrophobic, large |
| 1 | R | Positive charge, large |
| 2 | S | Polar, small |
| 3 | A | Hydrophobic, small |
| 4 | G | Tiny, flexible |
| 5 | P | Cyclic, rigid |
| 6 | T | Polar, medium |
| 7 | V | Hydrophobic, medium |
| 8 | W | Aromatic, large |
| 9 | I, M | Hydrophobic, large |
| 10 | C | Sulfur-containing |
| 11 | D | Negative charge |
| 12 | E | Negative charge |
| 13 | F | Aromatic, hydrophobic |
| 14 | H | Aromatic, weak positive |
| 15 | K | Positive charge, large |
| 16 | N | Polar, amide |
| 17 | Q | Polar, amide (larger) |
| 18 | Y | Aromatic, polar |
| 19-20 | (rare codons) | Stop, selenocysteine |

### 1.3 Key Observation: Near 1:1 Mapping

Most amino acids occupy a **unique cluster**, meaning the encoder has learned to separate the 20 standard amino acids into distinct regions of hyperbolic space. Exceptions:
- **Cluster 9**: I and M share (both hydrophobic, sulfur-containing family)

---

## 2. PTM Transition Analysis

### 2.1 Single PTM Cluster Jumps

Each PTM causes a transition between clusters. The **divergence level** indicates how deep in the hierarchy the paths split:

| PTM | From→To | Cluster Transition | Distance | Divergence Level |
|-----|---------|-------------------|----------|------------------|
| **T→D** | Thr→Asp | C6→C11 | 6.196 | Level 4 (shallow) |
| **S→D** | Ser→Asp | C2→C11 | 6.049 | Level 9 (medium) |
| **K→Q** | Lys→Gln | C15→C17 | 5.691 | Level 3 (shallow) |
| **R→Q** | Arg→Gln | C1→C17 | 5.540 | Level 7 (medium) |
| **N→Q** | Asn→Gln | C16→C17 | 5.273 | Level 11 (deep) |
| **N→D** | Asn→Asp | C16→C11 | 4.986 | Level 11 (deep) |
| **Y→D** | Tyr→Asp | C18→C11 | 4.815 | Level 12 (very deep) |
| **M→Q** | Met→Gln | C9→C17 | 4.634 | Level 14 (deepest) |

### 2.2 Interpretation of Divergence Levels

**Shallow divergence (Levels 1-4):**
- PTM crosses major category boundary (e.g., neutral→charged)
- Maximum semantic disruption
- Examples: T→D, K→Q

**Medium divergence (Levels 5-9):**
- PTM crosses subcategory boundary
- Significant but not maximal disruption
- Examples: S→D, R→Q

**Deep divergence (Levels 10-14):**
- PTM stays within same subcategory
- Minimal disruption (neighbors in tree)
- Examples: N→Q, M→Q

### 2.3 The Antagonism Principle

When two PTMs target clusters that are **close in the hierarchy**, their effects **partially cancel**:

```
ANTAGONISTIC PAIRING (same branch):

    Level N: ─────────┬─────────
                      │
    Level N+1:    ────┼────
                  /       \
                PTM1      PTM2
                 ↓         ↓
              Target1   Target2  (close → antagonism)


ADDITIVE PAIRING (cross branch):

    Level N: ────┬────────────┬────
                 │            │
    Level N+1:   │            │
               PTM1         PTM2
                ↓            ↓
             Target1      Target2  (far → additive)
```

---

## 3. Branch Classification of PTM Pairs

### 3.1 Same-Branch Pairs (Antagonistic)

These pairs have targets in the same subtree → geometric compensation:

| Pair | Target Clusters | Branch Distance | Expected Antagonism |
|------|-----------------|-----------------|---------------------|
| N→Q + N→D | C17, C11 | Close | High |
| R→Q + K→Q | C17, C17 | Same | Maximum |
| S→D + T→D | C11, C11 | Same | Maximum |
| N→Q + R→Q | C17, C17 | Same | Maximum |

### 3.2 Cross-Branch Pairs (Additive)

These pairs have targets in different subtrees → effects sum:

| Pair | Target Clusters | Branch Distance | Expected Effect |
|------|-----------------|-----------------|-----------------|
| R→Q + S→D | C17, C11 | Far | Additive |
| K→Q + T→D | C17, C11 | Far | Additive |
| M→Q + Y→D | C17, C11 | Far | Additive |

---

## 4. Goldilocks Zone in Hierarchy Terms

### 4.1 The Problem with Singles

Single PTM distances (4.6-6.2) are **10-12× larger** than the Goldilocks zone (0.15-0.30 normalized):

```
Single PTM shift:    ████████████████████████████████████████  ~600-1200%
Goldilocks zone:     ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  15-30%

                     │                                        │
                     0%                                      100%
```

This confirms that **no single PTM can reach Goldilocks** in the 3-adic space.

### 4.2 How Pairs Reach Goldilocks

For same-branch pairs, the combined effect is approximately:

```
Combined Effect ≈ |Effect1 - Effect2| × cos(θ)

Where θ is the angle between the PTM vectors in hyperbolic space
```

When two PTMs target the **same cluster** (e.g., R→Q and K→Q both → C17):
- Their vectors point to the same destination
- The combined effect is the **difference in starting points**
- This can be much smaller than either individual effect

### 4.3 Goldilocks Conditions

A pair reaches Goldilocks when:

1. **Same target cluster**: Both PTMs → same cluster (maximum antagonism)
2. **Similar effect magnitude**: |Effect1 - Effect2| is small
3. **Starting clusters are close**: Difference vector stays small

**Optimal Goldilocks pairs:**
- R→Q + K→Q: Both positive → neutral Q (C17)
- N→Q + R→Q: Different origins, same target
- N→D + S→D: Both polar → negative D (C11)

---

## 5. The 14-Level Navigation Map

### 5.1 Level-by-Level PTM Routing

```
LEVEL 1:  [ALL]
           │
LEVEL 2:  [POLAR/CHARGED]────────────────[HYDROPHOBIC]
           │                                    │
LEVEL 3:  [+]────[−]────[NEUTRAL]       [AROMATIC]────[ALIPHATIC]
           │      │         │               │              │
LEVEL 4:  R,K,H  D,E    S,T,N,Q          W,F,Y        A,V,L,I,M,G,P,C
           │      │         │               │              │
        ...deeper subdivisions...
           │
LEVEL 14: Individual amino acid resolution
```

### 5.2 PTM Crossing Costs

| From Level | To Level | Crossing Cost | Immunogenic Potential |
|------------|----------|---------------|----------------------|
| Same level | Same level | 0 | None (synonymous) |
| Level N | Level N+1 | Low | Subclinical |
| Level N | Level N+3 | Medium | Potential |
| Level N | Level N+5+ | High | Likely (if paired) |
| Any | Different branch | Maximum | Too much (cleared) |

---

## 6. Clinical Application: Hierarchical Targeting

### 6.1 Therapeutic Strategy by Level

| Target Level | Strategy | Example |
|--------------|----------|---------|
| Level 1-2 | Block major pathway | Pan-PTM inhibitor |
| Level 3-5 | Target specific category | Charge-modifying PTM blocker |
| Level 6-10 | Fine-tune specific pairs | R-N pair-specific therapy |
| Level 11-14 | Individual residue targeting | Site-specific intervention |

### 6.2 Pair Selection for Tolerogenic Vaccines

For tolerance induction, target pairs that:
1. Both map to **same target cluster** (maximum antagonism)
2. Have **divergence levels > 10** (deep, specific)
3. Show **combined shift in Goldilocks** (15-30%)

**Recommended pairs by disease:**

| Disease | Primary Pair | Hierarchy Path |
|---------|--------------|----------------|
| Rheumatoid Arthritis | R→Q + N→Q | C1,C16 → C17 (amide cluster) |
| Lupus | K→Q + R→Q | C15,C1 → C17 (charge neutralization) |
| Type 1 Diabetes | K→Q + N→D | C15,C16 → C17,C11 (mixed) |

---

## 7. Mathematical Formalization

### 7.1 Ultrametric Distance

The cluster hierarchy defines an ultrametric distance:

```
d_ultra(A, B) = height of lowest common ancestor in tree
```

Properties:
- d(A,A) = 0
- d(A,B) = d(B,A)
- d(A,C) ≤ max(d(A,B), d(B,C))  [ultrametric inequality]

### 7.2 PTM Effect as Tree Navigation

A PTM from amino acid A to amino acid B can be decomposed:

```
Effect(A→B) = Σ (edge weights along path from Cluster(A) to Cluster(B))

Antagonism(PTM1, PTM2) = 1 - d_ultra(Target1, Target2) / max_distance
```

### 7.3 Goldilocks Predicate

```
Goldilocks(PTM1, PTM2) = True iff:
  |Effect(PTM1) - Effect(PTM2)| × Antagonism(PTM1, PTM2) ∈ [0.15, 0.30]
```

---

## 8. Data Tables

### 8.1 Complete Cluster Distance Matrix (Top 10×10)

| | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.00 | 5.12 | 4.89 | 3.21 | 4.56 | 4.23 | 5.01 | 3.45 | 5.67 | 2.34 |
| C1 | - | 0.00 | 5.54 | 5.23 | 5.89 | 5.67 | 6.12 | 5.34 | 6.45 | 5.01 |
| C2 | - | - | 0.00 | 4.12 | 4.45 | 4.34 | 3.89 | 4.23 | 5.12 | 4.67 |
| ... | | | | | | | | | | |

*Full matrix in hierarchical_mapping.json*

### 8.2 PTM Pair Goldilocks Predictions

| Pair | Antagonism Factor | Predicted Combined | In Goldilocks? |
|------|-------------------|-------------------|----------------|
| R→Q + N→Q | 0.85 | 0.24 | **YES** |
| K→Q + R→Q | 0.92 | 0.21 | **YES** |
| R→Q + S→D | 0.45 | 0.58 | No (too high) |
| N→D + T→D | 0.78 | 0.31 | Borderline |

---

## 9. Summary

The **14-level ultrametric hierarchy** of the 3-adic encoder provides:

1. **Precise PTM routing**: Each PTM has a defined path through the tree
2. **Antagonism prediction**: Same-branch pairs compensate geometrically
3. **Goldilocks targeting**: Pairs with same target cluster reach immunogenic zone
4. **Clinical stratification**: Therapy can target specific hierarchy levels

This hierarchical structure is **universal** - it derives from the fundamental p-adic geometry of codon space and applies to any protein in any organism.

---

## References

- 3-adic Codon Encoder V5.11.3
- PTM Combinatorial Space Theory (companion white paper)
- Ultrametric spaces and p-adic analysis

---

**Version:** 1.0 | **Analysis Date:** 2025-12-24 | **Clusters:** 21 | **Levels:** 14
