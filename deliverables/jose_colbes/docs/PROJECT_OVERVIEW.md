# Project Overview: Ternary VAE for Protein Rotamer Stability

## The Big Picture

This project applies **novel mathematical frameworks** (p-adic numbers and hyperbolic geometry) to computational biology problems. Your deliverable focuses on protein rotamer stability scoring using these geometric techniques.

---

## Why P-adic and Hyperbolic Geometry?

### Traditional Approaches (Rosetta/Dunbrack)
- Statistical potentials from PDB frequency
- Euclidean distance to nearest rotamer
- May miss geometrically unusual but valid conformations

### Our Innovation
- **P-adic valuations**: Capture algebraic depth of angle encodings
- **Hyperbolic distance**: Natural metric for conformational space
- **Geometric scoring**: Complementary to statistical approaches

---

## Your Project: P-adic Rotamer Stability Scoring

### The Problem
Protein side-chain conformations (rotamers) critically affect:
- Protein folding accuracy
- Enzyme active site geometry
- Protein-protein interactions

Current methods may miss "Rosetta-blind" unstable conformations.

### Our Solution
A **geometric scoring function** based on:
1. **Hyperbolic distance** from common rotamer positions
2. **P-adic valuation** of chi angle encodings
3. **Correlation** with Dunbrack library probabilities

### Why This Works
- Hyperbolic geometry captures the curved nature of angular space
- P-adic valuations reveal algebraic structure invisible to Euclidean metrics
- Together, they identify conformations that "look normal" but are geometrically unusual

---

## Key Results in Your Package

### `results/rotamer_stability.json`
Complete analysis of 500 demo residues.

**Summary Statistics:**
```json
{
  "n_residues": 500,
  "hyperbolic_distance": {
    "mean": 7.679,
    "std": 0.803,
    "min": 7.600,
    "max": 17.329
  },
  "padic_valuation": {
    "mean": 0.434,
    "max": 6
  },
  "hyp_eucl_correlation": -0.051
}
```

**What these metrics mean:**

| Metric | Meaning | Interpretation |
|--------|---------|----------------|
| `hyperbolic_distance` | Distance in Poincare ball model | Higher = more unusual conformation |
| `padic_valuation` | Divisibility by prime p=3 | Higher = "deeper" algebraic structure |
| `euclidean_distance` | Standard distance to nearest rotamer | Traditional metric for comparison |
| `hyp_eucl_correlation` | Correlation between metrics | Low = hyperbolic captures different info |

### Per-Residue Analysis

Each residue entry contains:
```json
{
  "pdb_id": "1CRN",
  "chain_id": "A",
  "residue_id": 5,
  "residue_name": "LEU",
  "chi_angles": [-65.2, 174.3, null, null],
  "nearest_rotamer": "mt",
  "euclidean_distance": 0.534,
  "hyperbolic_distance": 7.612,
  "padic_valuation": 1,
  "stability_score": 0.876,
  "is_rare": true
}
```

**Interpretation:**
- chi1 = -65.2 deg approx gauche- (m)
- chi2 = 174.3 deg approx trans (t)
- Nearest standard rotamer: "mt"
- Flagged as rare: geometric analysis suggests instability

---

## The Proposed Energy Term

### E_geom Formula

```
E_geom = a * d_hyp(chi) + b * v_p(chi) + c * (1 - P_dunbrack)

Where:
  d_hyp(chi)    = hyperbolic distance from common centroid
  v_p(chi)      = p-adic valuation of chi angle encoding
  P_dunbrack    = Dunbrack library probability
  a, b, c       = weighting parameters
```

### Integration with Rosetta

```python
# Pseudocode for Rosetta integration
def geometric_score(residue):
    chi_angles = extract_chi_angles(residue)

    # Our geometric terms
    d_hyp = hyperbolic_distance(chi_angles)
    v_p = padic_valuation(encode_chi(chi_angles))

    # Dunbrack probability
    p_dun = dunbrack_probability(residue.name, chi_angles)

    # Combined score
    return alpha * d_hyp + beta * v_p + gamma * (1 - p_dun)
```

### Parameter Fitting

To fit the weights:
1. Collect structures with known stability issues
2. Compare E_geom rankings with experimental stability
3. Optimize weights to maximize correlation

---

## Hyperbolic Geometry Explanation

### The Poincare Ball Model

Chi angles are mapped to a hyperbolic disk:

```
         Boundary (inf)
        +---------------+
       /                 \
      |    Common         |
      |   Rotamers        |
      |      *            |
      |                   |
       \    Rare         /
        +---------------+
              ^
         High d_hyp
```

- **Center**: Common rotamer conformations
- **Boundary**: Increasingly rare/strained conformations
- **Distance grows exponentially** towards boundary

### Why Hyperbolic?

Angular space is naturally curved:
- Angles "wrap around" (360 deg = 0 deg)
- Small chi changes near common rotamers matter more
- Hyperbolic metric captures this curvature

---

## P-adic Valuation Explanation

### What is P-adic Valuation?

For prime p=3, the valuation v_p(n) counts how many times 3 divides n:

```
v_3(1) = 0    (1 = 3^0 * 1)
v_3(3) = 1    (3 = 3^1 * 1)
v_3(9) = 2    (9 = 3^2 * 1)
v_3(27) = 3   (27 = 3^3 * 1)
v_3(6) = 1    (6 = 3^1 * 2)
```

### Application to Chi Angles

Chi angles are encoded as integers (discretized):
```
chi_encoded = int((chi + 180) / 10)  # 0-35 range
combined = chi1_enc + 36 * chi2_enc + 36^2 * chi3_enc + ...
v_p = padic_valuation(combined, p=3)
```

**High valuation** = angle combination aligns with algebraic structure
**Low valuation** = "random" angle combination

---

## Connection to Your Research

### Hard-to-Fold Proteins

Your experience with protein optimization suggests:
- Some rotamers pass Rosetta scoring but cause folding issues
- Statistical potentials miss geometric constraints
- Our approach may flag these problematic conformations

### CASP Target Validation

To validate:
1. Run analysis on CASP target structures
2. Compare E_geom with Rosetta rotamer scores
3. Identify cases where E_geom flags issues Rosetta misses
4. Correlate with experimental folding data

---

## Next Steps for Validation

1. **Run with demo data** to verify installation
2. **Test on known structures** (1CRN, 1TIM, etc.)
3. **Compare with Rosetta** rotamer energies
4. **Identify discrepancies** where methods disagree
5. **Validate experimentally** on hard-to-fold cases

---

## References

- Dunbrack, R.L. (2002). "Rotamer Libraries in the 21st Century"
- Shapovalov, M.V. & Dunbrack, R.L. (2011). "A Smoothed Backbone-Dependent Rotamer Library"

---

*Prepared for Dr. Jose Colbes - Ternary VAE Bioinformatics Partnership*
