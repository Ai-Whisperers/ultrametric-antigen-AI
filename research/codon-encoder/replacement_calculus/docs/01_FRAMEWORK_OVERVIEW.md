# Replacement Calculus Framework Overview

**Doc-Type:** Technical Documentation · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Purpose

The Replacement Calculus framework provides a rigorous mathematical foundation for analyzing local minima in optimization landscapes. Rather than treating local minima as "traps" to escape, this framework models them as **algebraic groups** with structure that can be preserved through **morphisms**.

---

## Core Thesis

> "We should focus on a **replacement calculus**, not an optimization trick. The insight is: local minima are not single states—they are **groups**."

### Key Principles

1. **Local Minima as Groups**: Each stable configuration has generators (primitives), relations (constraints), and symmetries
2. **Transformations as Morphisms**: Valid transformations preserve structure, not just improve metrics
3. **Global Optima as Groupoids**: Multiple coordinate systems with partial symmetries
4. **Minimal Invariant Basis**: (valuation, redundancy, symmetry_rank) characterizes any configuration

---

## Mathematical Foundation

### Invariant Tuple

Every configuration has an invariant signature:

```
I(x) = (ν(x), ρ(x), σ(x))
```

Where:
- **ν(x)**: p-adic valuation (hierarchical depth)
- **ρ(x)**: Redundancy (coset index / degeneracy)
- **σ(x)**: Symmetry rank (automorphism dimension)

### Partial Ordering

Configurations form a partial order:

```
x ≤ y  iff  ν(x) ≤ ν(y) AND ρ(x) ≤ ρ(y) AND σ(x) ≤ σ(y)
```

This is NOT a total order - some configurations are incomparable.

### Morphism Validity

A morphism φ: G → H is valid iff:
1. **Valuation Preservation**: ∀x: ν(φ(x)) ≥ ν(x)
2. **Invariant Preservation**: I(H) ≥ I(G)
3. **Entropy Displacement**: Entropy change must be absorbed elsewhere

---

## Module Architecture

```
replacement_calculus/
├── __init__.py              # Package exports
├── invariants.py            # InvariantTuple, valuation, redundancy, symmetry_rank
├── groups.py                # LocalMinimum, Constraint classes
├── morphisms.py             # Morphism, validity checking, composition
├── groupoids.py             # Groupoid, escape paths, analysis
├── integration/             # VAE integration scripts
│   ├── vae_groupoid_builder.py
│   └── embedding_groupoid.py
├── tests/                   # Unit tests
│   └── test_replacement_calculus.py
└── docs/                    # Documentation
```

---

## Usage Example

```python
from replacement_calculus import (
    InvariantTuple, LocalMinimum, Morphism, Groupoid,
    valuation, is_valid_morphism, find_escape_path
)

# Create local minima
source = LocalMinimum(
    name="config_A",
    generators=[3, 6, 9],  # Elements divisible by 3
    members=[np.array([0.1, 0.1]) for _ in range(3)],
)

target = LocalMinimum(
    name="config_B",
    generators=[9, 18, 27],  # Higher valuations
    members=[np.array([0.2, 0.2]) for _ in range(3)],
)

# Create morphism
morphism = Morphism(
    source=source,
    target=target,
    map_function=lambda x: x * 3,
)

# Validate
is_valid, reason = is_valid_morphism(morphism, p=3)
print(f"Valid: {is_valid} - {reason}")

# Build groupoid and find escape paths
groupoid = Groupoid(name="example")
idx_a = groupoid.add_object(source)
idx_b = groupoid.add_object(target)
groupoid.add_morphism(idx_a, idx_b, morphism)

path = find_escape_path(groupoid, idx_a, idx_b)
```

---

## Biological Application

The framework was validated against codon embeddings from TernaryVAE:

1. **Amino acids as LocalMinima**: Each amino acid's codons form a group
2. **Substitutions as Morphisms**: Valid substitutions preserve structure
3. **Genetic code as Groupoid**: Multiple coordinate systems (degenerate codons)

See `02_VALIDATION_RESULTS.md` for detailed findings.

---

## References

- Original proposal: `proposals/REPLACEMENT_CALCULUS.md`
- Test suite: `tests/test_replacement_calculus.py`
- Integration scripts: `integration/`
