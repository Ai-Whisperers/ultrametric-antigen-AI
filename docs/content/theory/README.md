# Theory & Foundations

> **Mathematical and biological foundations of the Ternary VAE framework.**

---

## Overview

Ternary VAE combines three theoretical pillars:

1. **Hyperbolic Geometry** - Tree-like latent space for evolutionary hierarchies
2. **P-adic Numbers** - Codon relationships based on 3-adic valuation
3. **Biological Context** - Drug resistance, immune evasion, evolution

---

## Core Concepts

### Why Hyperbolic Geometry?

Evolution produces **tree-like structures**. Hyperbolic space naturally represents trees because:
- Distance grows exponentially toward boundary
- More "room" at edges for diverse leaves
- Geodesics encode hierarchical relationships

```
Euclidean (flat):     Hyperbolic (curved):
    o                      o
   / \                   / | \
  o   o                 o  o  o
 /|\  |\               /|\ |\ /|\
o o o o o             o o o o o o o o
```

### Why P-adic Numbers?

The genetic code has a **ternary structure**:
- 3 nucleotide positions per codon
- 64 codons = 4³ combinations
- Hierarchical groupings by position

3-adic valuation captures "closeness" by shared structure:
- `AAA` and `AAG` differ in position 3 (low valuation)
- `AAA` and `GAA` differ in position 1 (high valuation)

---

## Documents

| Topic | Document | Description |
|:------|:---------|:------------|
| Hyperbolic Geometry | [hyperbolic.md](hyperbolic.md) | Poincare ball, geodesics, curvature |
| P-adic Numbers | [p-adic.md](p-adic.md) | 3-adic valuation, codon relationships |
| Biological Context | [biology.md](biology.md) | Drug resistance, immune evasion |

---

## Key Equations

### Poincare Ball Distance

$$d_{\mathcal{B}}(x, y) = \text{arccosh}\left(1 + 2\frac{\|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)}\right)$$

### 3-adic Valuation

$$v_3(n) = \max\{k : 3^k | n\}$$

### KL Divergence (Wrapped Normal)

$$D_{KL}(q \| p) = \frac{\mu^2 + \sigma^2 - 1 - \log\sigma^2}{2 \cdot c}$$

Where $c$ is the curvature.

---

## Related Documentation

- [Architecture Overview](../architecture/README.md)
- [Research Findings](../research/README.md)
- [Validation Benchmarks](../validation/README.md)

---

_Last updated: 2025-12-28_
