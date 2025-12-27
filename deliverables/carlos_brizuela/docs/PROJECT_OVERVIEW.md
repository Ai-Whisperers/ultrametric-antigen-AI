# Project Overview: Ternary VAE for Antimicrobial Peptide Optimization

## The Big Picture

This project applies **novel mathematical frameworks** (p-adic numbers and hyperbolic geometry) to computational biology problems. Your deliverable focuses on multi-objective optimization of antimicrobial peptides using these techniques.

---

## Why P-adic and Hyperbolic Geometry?

### Traditional Approaches
- Discrete sequence optimization (combinatorial explosion)
- Euclidean distance metrics (poor for hierarchical data)
- Statistical potentials (miss geometric structure)

### Our Innovation
- **P-adic numbers**: Capture hierarchical/tree-like structure in biological sequences
- **Hyperbolic geometry**: Natural space for evolutionary trees and sequence similarity
- **VAE latent space**: Continuous representation enabling smooth optimization

---

## Your Project: NSGA-II Latent Space Optimization

### The Problem
Designing antimicrobial peptides requires balancing:
- High antimicrobial activity
- Low toxicity to host cells
- Structural stability

Traditional methods mutate sequences directly - a combinatorial explosion (20^L possibilities).

### Our Solution
Optimize in the **continuous latent space** of a trained VAE:
1. VAE learns compressed representation of peptide sequences
2. NSGA-II explores this smooth 16D space
3. Decode optimal latent vectors back to sequences

### Why This Works
- Latent space is continuous - gradient-free optimization is efficient
- Similar sequences cluster together - local search finds related variants
- Multiple objectives handled naturally by Pareto optimization

---

## Key Results in Your Package

### `results/pareto_peptides.csv`
Contains 100 Pareto-optimal solutions from demo run.

**What the columns mean:**

| Column | Meaning | Ideal Value |
|--------|---------|-------------|
| `objective_0` | Reconstruction loss | Lower = valid sequence |
| `objective_1` | Toxicity score | Lower = safer |
| `objective_2` | Negative activity | More negative = more active |
| `z_0` ... `z_15` | Latent coordinates | Between -3 and 3 |
| `crowding_distance` | Diversity measure | Higher = more unique |

**How to interpret the Pareto front:**
- All 100 solutions are "optimal" - no solution dominates another
- Solutions with `inf` crowding distance are at the boundaries
- Choose based on your priority (safety vs. efficacy)

### Sample Trade-offs

```
Solution 2: obj_0=0.740, obj_1=0.00002, obj_2=-0.851
           - Very low toxicity, moderate activity

Solution 3: obj_0=0.306, obj_1=0.132, obj_2=-0.534
           - Balanced across all objectives

Solution 0: obj_0=7.054, obj_1=1.007, obj_2=-2.615
           - Highest activity, but also higher toxicity
```

---

## Connection to Your Research

### StarPepDB Integration
The framework connects to StarPepDB for real AMP data:
- Hemolytic activity scores
- MIC (Minimum Inhibitory Concentration)
- Sequence-activity relationships

### With Your Trained VAE
Replace the demo objectives with:
```python
def real_toxicity(z):
    sequence = vae.decode(z)
    return toxicity_model.predict(sequence)

def real_activity(z):
    sequence = vae.decode(z)
    return -activity_model.predict(sequence)  # Negative for minimization
```

---

## Technical Foundation

### NSGA-II Algorithm
1. **Initialize** random population in latent space
2. **Evaluate** all objectives for each individual
3. **Non-dominated sort** to identify Pareto fronts
4. **Crowding distance** to maintain diversity
5. **Selection** via binary tournament
6. **Crossover** (SBX) and **Mutation** (Polynomial)
7. **Repeat** for N generations

### Mathematical Details

**SBX Crossover** (Simulated Binary Crossover):
```
b = (2u)^(1/(n+1))           if u <= 0.5
b = (1/(2(1-u)))^(1/(n+1))   if u > 0.5

child1 = 0.5 * ((p1 + p2) - b * |p2 - p1|)
child2 = 0.5 * ((p1 + p2) + b * |p2 - p1|)
```

**Polynomial Mutation**:
```
d = (2u)^(1/(n+1)) - 1       if u < 0.5
d = 1 - (2(1-u))^(1/(n+1))   if u >= 0.5

mutant = parent + d * (upper - lower)
```

---

## Next Steps for Validation

1. **Run with demo data** to verify installation
2. **Connect your VAE** for real sequence decoding
3. **Add real objective functions** (toxicity, activity models)
4. **Analyze Pareto front** for peptide candidates
5. **Decode and synthesize** top candidates
6. **Experimental validation** in the lab

---

## References

- Deb, K., et al. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
- Deb, K. & Agrawal, R.B. (1995). "Simulated Binary Crossover for Continuous Search Space"
- StarPepDB: https://starpepdh.sytes.net/

---

*Prepared for Carlos Brizuela - Ternary VAE Bioinformatics Partnership*
