# Deliverable Package: Carlos Brizuela
## Antimicrobial Peptide Multi-Objective Optimization

**Prepared for:** Carlos Brizuela
**Project:** Ternary VAE Bioinformatics - Partnership Phase 1
**Date:** December 26, 2024

---

## Overview

This package contains all materials for validating the **NSGA-II Latent Space Optimizer** for antimicrobial peptide design. The system enables multi-objective optimization directly in the continuous latent space of a Variational Autoencoder, avoiding the discrete combinatorial explosion of traditional sequence-based methods.

---

## What's Included

### 1. Core Scripts

| File | Description |
|------|-------------|
| `scripts/latent_nsga2.py` | Complete NSGA-II implementation (490 lines) |
| `scripts/__init__.py` | Module initialization |

### 2. Interactive Notebook

| File | Description |
|------|-------------|
| `notebooks/brizuela_amp_navigator.ipynb` | Jupyter notebook for visualization and exploration |

### 3. Results

| File | Description |
|------|-------------|
| `results/pareto_peptides.csv` | 100 Pareto-optimal solutions from demo run |

### 4. Reference Data

| File | Description |
|------|-------------|
| `data/demo_amp_embeddings.csv` | 20 reference AMP sequences with properties |

### 5. Documentation

| File | Description |
|------|-------------|
| `docs/TECHNICAL_BRIEF.md` | Original technical proposal |
| `docs/IMPLEMENTATION_GUIDE.md` | Implementation specifications |

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install numpy torch pandas matplotlib seaborn
```

### Step 2: Run Demo Optimization

```bash
cd scripts
python latent_nsga2.py \
    --generations 50 \
    --population 100 \
    --output ../results/pareto_demo.csv
```

**Expected Output:**
```
Using mock objectives (no VAE checkpoint provided)

Starting NSGA-II optimization:
  Population: 100
  Generations: 50
  Latent dim: 16
  Objectives: 3

Gen    0: Pareto front size = 74, Best objectives = [ 1.69  0.01 -2.16]
Gen   10: Pareto front size = 100, Best objectives = [ 1.23  0.00 -2.43]
...
Gen   40: Pareto front size = 100, Best objectives = [ 0.40  0.00 -2.59]
Exported 100 Pareto-optimal solutions to results/pareto_demo.csv

Optimization complete!
  Pareto front size: 100
```

### Step 3: Explore Results in Notebook

```bash
jupyter notebook notebooks/brizuela_amp_navigator.ipynb
```

---

## Technical Details

### The Three Objectives (All Minimized)

| Objective | Description | Interpretation |
|-----------|-------------|----------------|
| `objective_0` | Reconstruction Loss | How well the latent vector reconstructs to a valid peptide |
| `objective_1` | Toxicity Score | Predicted toxicity to host cells (lower = safer) |
| `objective_2` | Negative Activity | Antimicrobial activity (more negative = more active) |

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 16 | Dimensionality of VAE latent space |
| `population_size` | 200 | Number of individuals per generation |
| `generations` | 100 | Number of evolutionary generations |
| `crossover_prob` | 0.9 | Probability of SBX crossover |
| `mutation_prob` | 0.1 | Per-gene mutation probability |
| `latent_bounds` | (-3, 3) | Valid latent coordinate range |

### Output CSV Format

```csv
id,rank,crowding_distance,objective_0,objective_1,objective_2,z_0,z_1,...,z_15
0,0,inf,0.305,0.132,-0.534,-0.234,1.456,...,0.678
```

- **id**: Solution identifier
- **rank**: Pareto rank (0 = optimal front)
- **crowding_distance**: Diversity measure (inf = boundary solution)
- **objective_X**: Objective values
- **z_X**: 16D latent coordinates

---

## Integration with Your VAE

To use with your trained Ternary VAE:

```python
from latent_nsga2 import LatentNSGA2, OptimizationConfig

# Define objective functions using your models
def reconstruction_objective(z):
    """Use your VAE decoder + reconstruction loss."""
    decoded = vae.decode(torch.tensor(z))
    recon_loss = vae.reconstruction_loss(decoded)
    return float(recon_loss)

def toxicity_objective(z):
    """Use your toxicity regressor."""
    return float(toxicity_model(torch.tensor(z)))

def activity_objective(z):
    """Negative activity (to minimize)."""
    return -float(activity_model(torch.tensor(z)))

# Configure and run
config = OptimizationConfig(
    latent_dim=16,  # Match your VAE
    population_size=200,
    generations=100,
)

optimizer = LatentNSGA2(
    config=config,
    objective_functions=[reconstruction_objective, toxicity_objective, activity_objective],
    decoder=lambda z: vae.decode_to_sequence(z),  # Optional: for sequence output
)

pareto_front = optimizer.run(verbose=True)
```

---

## Key Innovation

**Traditional Approach:**
```
Sequence Space (20^L combinations) → Discrete mutations → Evaluation
```

**Our Approach:**
```
Latent Space (continuous 16D) → Smooth optimization → Decode to sequences
```

This enables:
- Gradient-free multi-objective optimization
- Smooth exploration of peptide property landscape
- Direct Pareto front discovery
- Interpretable trade-offs between objectives

---

## Validation Checklist

- [ ] Scripts run without errors
- [ ] Demo optimization completes 50 generations
- [ ] Pareto front contains diverse solutions
- [ ] CSV export is readable
- [ ] Notebook visualizations render correctly
- [ ] Integration with your VAE works

---

## Questions?

- See docstrings in `latent_nsga2.py` for implementation details
- Algorithm based on Deb et al. (2002) NSGA-II paper
- SBX crossover based on Deb & Agrawal (1995)

---

*Prepared as part of the Ternary VAE Bioinformatics Partnership*
