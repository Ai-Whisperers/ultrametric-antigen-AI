# Discovery: Architecture-Encoded Prime Capacity

**Date:** 2025-12-16
**Model:** V5.11.3 Structural (v1.1.0)

---

## The Observation

The radial exponent **c = 1/6** emerges from:

```
c = 1/(latent_dim - n_trits - 1) = 1/(16 - 9 - 1) = 1/6
```

And **6 = 2 × 3** - the product of the first two primes.

---

## The Hypothesis

### Current Architecture

```
latent_dim = 16
n_trits = 9 (for 3⁹ = 19683 operations)
effective_slack = 16 - 9 - 1 = 6 = 2 × 3
```

The "slack" of 6 dimensions determines the radial hierarchy's capacity.

### Proposed Design Principle

**To encode p-adic structure for primes {p₁, p₂, ...}, choose:**

```
latent_dim = n_trits + 1 + Π pᵢ
```

This gives exponent `c = 1/Π pᵢ`, encoding all specified primes.

### Predictions

| Primes | Product | latent_dim | Predicted c |
|:-------|:--------|:-----------|:------------|
| {3} | 3 | 9+1+3=13 | 1/3 = 0.333 |
| {2,3} | 6 | 9+1+6=16 | **1/6 = 0.167** ✓ |
| {2,3,5} | 30 | 9+1+30=40 | 1/30 = 0.033 |
| {2,3,5,7} | 210 | 9+1+210=220 | 1/210 = 0.005 |

---

## Interpretation: Binary Approximates Ternary

The user's insight: "2×3 is essentially using ternary but constraining it dynamically for capturing binary bifurcations."

### In Floating Point

Binary FP approximates continuous values through:
- Mantissa (precision)
- Exponent (range/scale)

The approximation works because binary can represent any rational to arbitrary precision.

### In Hyperbolic Embeddings

The 6-dimensional "slack" works similarly:
- **2 dimensions**: Could encode binary bifurcations (even/odd structure)
- **3 dimensions**: Encode ternary refinement (mod 3 residue classes)

Even though we only trained on 3-adic structure, the architecture **has capacity** for 2-adic structure too.

---

## Experimental Evidence

### What We Found

```
3-adic exponent: 0.183 ≈ 1/6 (trained)
2-adic exponent: 0.0001 ≈ 0 (not trained, capacity unused)
```

The 2-adic capacity EXISTS but is UNUSED because:
1. Training loss only penalized 3-adic violations
2. No gradient signal to organize by mod 2

### Prediction for Multi-Prime Training

If we add a 2-adic loss term:
```python
loss_2adic = radial_hierarchy_loss(z, valuations_mod_2)
```

Then the 2 dimensions of "binary slack" would become active, and we'd see:
- 2-adic exponent b → 1/6 (or some fraction)
- Better multi-prime structure
- Possibly GUE statistics emerging

---

## The Deeper Pattern

### Architecture as Constraint

The formula `c = 1/(latent_dim - n_trits - 1)` shows that:

1. **Architecture constrains learning**: The network CAN'T learn an arbitrary exponent
2. **Capacity is pre-allocated**: 6 dimensions = 2×3 is "waiting" for multi-prime structure
3. **Training fills capacity**: Loss function determines WHICH prime structure is learned

### Analogy to Number Theory

This mirrors how the integers decompose:

```
ℤ ≅ ℤ₂ × ℤ₃ × ℤ₅ × ... (p-adic decomposition)
```

Our architecture implicitly reserves:
```
R^16 ≅ R^9 (trits) × R^1 (identity) × R^6 (prime capacity)
```

And 6 = 2×3 means capacity for exactly the first two primes.

---

## Design Recommendations

### For Single-Prime (Current)

```python
latent_dim = n_trits + 1 + p  # For prime p
# p=3: latent_dim = 9+1+3 = 13 → c = 1/3
```

### For Multi-Prime (Adelic)

```python
latent_dim = n_trits + 1 + lcm(p1, p2, ...)  # Or product
# p={2,3,5}: latent_dim = 9+1+30 = 40 → c = 1/30
```

### For Approximating All Primes ≤ N

```python
latent_dim = n_trits + 1 + primorial(N)
# primorial(5) = 2×3×5 = 30
# primorial(7) = 2×3×5×7 = 210
```

---

## Next Steps

1. **Verify prediction**: Train with latent_dim=13, check if c=1/3
2. **Multi-prime training**: Add 2-adic loss, see if binary structure emerges
3. **Adelic architecture**: Design latent_dim for specific prime sets
4. **Test GUE**: Does multi-prime training produce GUE statistics?

---

## Connection to Riemann Hypothesis

The zeta function's Euler product:
```
ζ(s) = Π_p (1 - p^{-s})^{-1}
```

Our architecture might encode a "truncated" Euler product:
```
capacity = Π_{p ≤ P} p = primorial(P)
```

If GUE statistics require ALL primes, we'd need:
```
latent_dim → ∞ (impractical)
```

But if GUE emerges from the STRUCTURE of the product (not its size), then:
```
latent_dim = n_trits + 1 + 6 might already capture essential adelic structure
```

This is the optimistic interpretation of why 6 = 2×3 matters.

---

**Status:** Hypothesis - requires experimental verification
**Key Test:** Train with latent_dim=13, verify c=1/3
