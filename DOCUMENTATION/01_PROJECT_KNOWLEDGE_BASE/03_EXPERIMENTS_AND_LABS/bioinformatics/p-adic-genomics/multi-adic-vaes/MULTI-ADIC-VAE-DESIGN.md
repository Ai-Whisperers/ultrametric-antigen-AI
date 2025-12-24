# Multi-Adic VAE Design: 2-Adic and 5-Adic Variants for Genetic Code Modeling

**Doc-Type:** Technical Design Document - Version 1.0 - Updated 2025-12-18 - Author AI Whisperers

---

## Motivation

The genetic code exhibits ultrametric structure that different p-adic bases capture differently:

**2-adic** - Captures codon degeneracy patterns. The 64 codons form multiplets where amino acid identity depends heavily on first two positions. Binary branching (2-adic trees) naturally models the complementarity rules (A-T, C-G) and the "wobble" degeneracy at position 3.

**3-adic** - Currently implemented. Captures the 3-position hierarchy of codons (1st, 2nd, 3rd base). Intermediate branching factor without binary bias. Maps well to post-translational modification cascades.

**5-adic** - Models DNA sequence structure with complementarity constraint (C+G = A+T = 5 in classical Dragovich encoding). The prime 5 relates to the "magic number" in codon complementarity. Captures evolutionary constraints at sequence level.

---

## Mathematical Foundations

### p-Adic Valuation

For integer n and prime p, the p-adic valuation v_p(n) is the largest power of p dividing n:

```
v_p(n) = max{k : p^k | n}
v_p(0) = infinity (by convention)
```

### p-Adic Distance

```
d_p(x, y) = p^(-v_p(x - y))
```

This generates an ultrametric: d(x,z) <= max(d(x,y), d(y,z))

### Encoding Schemes

**2-adic for Codons (Degeneracy Focus)**
- Encode 64 codons as integers 0-63
- 2-adic distance clusters codons by wobble degeneracy
- v_2(n) captures how "degenerate" a codon family is

**5-adic for DNA Sequences (Complementarity Focus)**
- Dragovich encoding: A=0, C=1, G=2, T=3, (gap=4)
- Sequence as base-5 integer: n = sum(d_i * 5^i)
- Captures evolutionary substitution patterns

---

## Architecture Design

### Base Class: PAdicSpace

```python
class PAdicSpace:
    """Generic p-adic space with configurable prime base."""

    def __init__(self, prime: int, dimension: int):
        self.p = prime
        self.dim = dimension
        self._precompute_tables()

    def valuation(self, n: int) -> int:
        """Compute v_p(n)."""
        if n == 0:
            return float('inf')
        v = 0
        while n % self.p == 0:
            n //= self.p
            v += 1
        return v

    def distance(self, x: int, y: int) -> float:
        """Compute d_p(x, y) = p^(-v_p(x-y))."""
        return self.p ** (-self.valuation(x - y))
```

### 2-Adic VAE for Codon Degeneracy

**Input Space**: 64 codons encoded as 6-bit integers
**Latent Space**: 8D Poincare ball (2^3 = 8 natural branching)
**Loss Function**: 2-adic geodesic loss aligning hyperbolic distance with 2-adic distance

**Key Insight**: Codons differing only at wobble position (3rd) should cluster tightly in hyperbolic space.

### 5-Adic VAE for DNA Sequences

**Input Space**: DNA k-mers encoded as base-5 integers (with gap symbol)
**Latent Space**: 16D or 25D Poincare ball (5^k natural)
**Loss Function**: 5-adic geodesic loss capturing evolutionary substitution distances

**Key Insight**: Synonymous mutations (same amino acid) should have smaller 5-adic distance than non-synonymous.

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Priority)

1. **PAdicSpace base class** - Generic p-adic operations for any prime
2. **TwoAdicSpace** - Specialized 2-adic with optimized bit operations
3. **FiveAdicSpace** - 5-adic with Dragovich encoding utilities
4. **Unit tests** - Full coverage of valuation, distance, conversion

### Phase 2: Model Architecture

1. **PAdicEncoder** - Generic encoder accepting PAdicSpace config
2. **PAdicProjection** - Hyperbolic projection with p-adic loss alignment
3. **CodonDegeneracyVAE** - 2-adic specialized for codon clustering
4. **DNASequenceVAE** - 5-adic specialized for k-mer embeddings

### Phase 3: Loss Functions

1. **PAdicGeodesicLoss** - Generalized for any prime
2. **DegeneracyPreservingLoss** - 2-adic specific for amino acid clusters
3. **EvolutionaryDistanceLoss** - 5-adic specific for substitution matrices

### Phase 4: Training Pipeline

1. **Codon dataset** - 64 codons with amino acid labels
2. **K-mer dataset** - DNA sequences with evolutionary annotations
3. **Multi-adic trainer** - Unified training loop for any p-adic VAE
4. **Evaluation metrics** - Coverage, hierarchy preservation, biological validity

### Phase 5: Validation

1. **Degeneracy clustering** - Verify 2-adic VAE groups synonymous codons
2. **Evolutionary distance** - Verify 5-adic captures substitution rates
3. **Cross-validation** - Compare 2-adic, 3-adic, 5-adic on same tasks
4. **Biological benchmarks** - Predict mutation impact, PTM effects

---

## Directory Structure

```
multi-adic-vaes/
├── MULTI-ADIC-VAE-DESIGN.md    # This document
├── docs/
│   ├── 2-ADIC-THEORY.md        # Mathematical details for 2-adic
│   └── 5-ADIC-THEORY.md        # Mathematical details for 5-adic
├── src/
│   ├── spaces/
│   │   ├── __init__.py
│   │   ├── base.py             # PAdicSpace base class
│   │   ├── two_adic.py         # TwoAdicSpace (bit-optimized)
│   │   └── five_adic.py        # FiveAdicSpace (Dragovich encoding)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── padic_vae.py        # Generic PAdicVAE
│   │   ├── codon_vae.py        # 2-adic CodonDegeneracyVAE
│   │   └── sequence_vae.py     # 5-adic DNASequenceVAE
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── padic_geodesic.py   # Generic p-adic geodesic loss
│   │   ├── degeneracy.py       # 2-adic degeneracy preservation
│   │   └── evolutionary.py     # 5-adic evolutionary distance
│   └── data/
│       ├── __init__.py
│       ├── codons.py           # Codon dataset generation
│       └── kmers.py            # K-mer dataset generation
├── configs/
│   ├── 2_adic_codon.yaml       # 2-adic VAE configuration
│   └── 5_adic_sequence.yaml    # 5-adic VAE configuration
├── scripts/
│   ├── train_2adic.py          # Training script for 2-adic
│   ├── train_5adic.py          # Training script for 5-adic
│   └── evaluate.py             # Cross-adic evaluation
└── tests/
    ├── test_spaces.py          # Unit tests for p-adic spaces
    ├── test_models.py          # Unit tests for VAE models
    └── test_losses.py          # Unit tests for loss functions
```

---

## Integration with Existing Codebase

**Reusable Components**:
- `src/models/hyperbolic_projection.py` - Poincare ball projection
- `src/losses/padic_geodesic.py` - Template for geodesic loss
- `src/training/trainer.py` - Base trainer class
- `src/training/monitor.py` - TensorBoard logging

**New Components**:
- Generalized PAdicSpace replacing TernarySpace (3-adic specific)
- Configurable prime parameter in all p-adic operations
- Biological domain data loaders (codons, k-mers)

---

## Success Criteria

1. **Full Coverage**: 100% of codon space (64) and target k-mer space covered
2. **Hierarchy Preservation**: Spearman correlation > 0.9 between p-adic and hyperbolic distances
3. **Biological Validity**: Synonymous codon clusters in 2-adic latent space
4. **Evolutionary Fidelity**: 5-adic distances correlate with PAM/BLOSUM scores
5. **Cross-adic Comparison**: Quantitative comparison of 2, 3, 5-adic on shared tasks

---

## References

- Dragovich, B., Khrennikov, A., Kozyrev, S. (2009). p-Adic mathematical physics
- Khrennikov, A. (2016). Ultrametric models of genetic code
- Dragovich, B. (2012). p-Adic approach to the genetic code and genome
- Current 3-adic implementation: `/src/core/ternary.py`

---

## Status

**Phase**: Design Complete, Implementation Pending
**Priority**: High - Foundational for genetic code applications
**Dependencies**: Existing V5.11 hyperbolic VAE architecture
**Estimated Effort**: 5 implementation phases (see roadmap above)
