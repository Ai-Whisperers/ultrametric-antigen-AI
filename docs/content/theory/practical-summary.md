# Just Enough Theory

> **One-page summary for bioinformatics practitioners** - The intuition without the math.
>
> For full mathematical details, see [../../mathematical-foundations/](../../mathematical-foundations/).

---

## The Core Idea

**Traditional ML** treats sequences as flat data - every position is equally important.

**This project** recognizes that sequences have **hierarchy**:
- Some positions matter more than others
- Some mutations are worse than others
- Some patterns are more fundamental than others

We capture this hierarchy using geometric embeddings.

---

## What "P-adic" Means for Codons

Think of it as "position weighting":

| Codon Position | Impact of Mutation | P-adic Term |
|----------------|-------------------|-------------|
| Position 1 | Usually changes amino acid | "High valuation" |
| Position 2 | Usually changes amino acid | "High valuation" |
| Position 3 (wobble) | Often synonymous | "Low valuation" |

**The key insight:** Codons that differ only at position 3 should be "close" in our embedding because they often encode the same amino acid.

**Practical implication:** Our distance metric captures biological similarity, not just sequence similarity.

---

## What "Hyperbolic" Means

Imagine embedding data on the inside of a ball:

- **Center**: Fundamental, conserved patterns
- **Edge**: Diverse, specialized patterns
- **Distance grows exponentially** as you move toward the edge

**Why this helps:**
- Tree structures (like evolution) fit naturally
- Hierarchies are preserved, not distorted
- Common ancestors are near the center

**Practical implication:** Radial position in our embedding tells you how "fundamental" a sequence pattern is.

---

## The Hierarchy Metric

We measure how well the embedding captures hierarchy using **Spearman correlation**:

| Value | Meaning |
|-------|---------|
| -1.0 | Perfect: rare patterns at center, common at edge |
| -0.83 | Excellent (our ceiling due to data distribution) |
| 0.0 | No relationship |
| +0.70 | "Frequency-optimal" (valid alternative) |

**Why negative is usually better:** We want important/rare patterns near the center where they have more "room" (hyperbolic space grows outward).

---

## What This Means for Your Application

### Protein Stability (ΔΔG)

- Mutations between "close" codons tend to be tolerated
- Mutations between "far" codons tend to be destabilizing
- Our distance predicts stability change

### Contact Prediction

- Residues with "close" embeddings tend to be spatially close
- Works because structural constraints drive codon selection
- No structure needed - just sequence

### Primer Design

- Conserved regions have low "hyperbolic variance" across sequences
- These are good primer targets
- Orthogonal to traditional Shannon entropy analysis

---

## What You DON'T Need to Know

To use this project, you don't need to understand:

- P-adic number theory
- Poincaré ball geodesics
- Möbius transformations
- Exponential/logarithmic maps

The checkpoints are pre-trained. Just load them and use the distance functions.

---

## Quick Code Example

```python
from src.geometry import poincare_distance
import torch

# Load embeddings (pre-computed)
emb_A = model.get_codon_embedding("GCU")  # Alanine
emb_V = model.get_codon_embedding("GUU")  # Valine
emb_D = model.get_codon_embedding("GAU")  # Aspartate

# Compute distances
dist_AV = poincare_distance(emb_A, emb_V, c=1.0)  # Small (both hydrophobic)
dist_AD = poincare_distance(emb_A, emb_D, c=1.0)  # Large (different properties)

# Use distance to predict mutation effect
# (actual prediction involves more features, but distance is key)
```

---

## Summary

| Concept | Intuition | What It Does |
|---------|-----------|--------------|
| P-adic | "Position importance" | Makes codon distances biologically meaningful |
| Hyperbolic | "Tree-friendly space" | Preserves evolutionary/structural hierarchy |
| Hierarchy | "Fundamental vs specialized" | Identifies conserved patterns |

---

*For the full mathematical treatment, see [../../mathematical-foundations/](../../mathematical-foundations/).*
