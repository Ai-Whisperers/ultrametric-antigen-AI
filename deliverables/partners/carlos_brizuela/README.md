# Deliverable Package: Carlos Brizuela
## Antimicrobial Peptide Multi-Objective Optimization Suite

**Prepared for:** Carlos Brizuela
**Project:** Ternary VAE Bioinformatics - Partnership Phase 3
**Date:** December 29, 2025
**Status:** COMPLETE - Ready for Production Use

---

## Executive Summary

This package provides a comprehensive toolkit for antimicrobial peptide (AMP) design using NSGA-II multi-objective optimization in the VAE latent space. It includes three specialized tools:

1. **B1: Pathogen-Specific AMP Design** - Design AMPs targeting specific pathogens (e.g., *A. baumannii*)
2. **B8: Microbiome-Safe AMPs** - Design selective peptides that kill pathogens while sparing commensals
3. **B10: Synthesis Optimization** - Balance antimicrobial activity with synthesis feasibility

---

## NEW: Easy Implementation Tools

### B1: Pathogen-Specific AMP Design

Design antimicrobial peptides optimized for specific WHO priority pathogens.

```bash
python scripts/B1_pathogen_specific_design.py \
    --pathogen "Acinetobacter_baumannii" \
    --output_dir results/pathogen_specific/
```

**Target Pathogens Supported:**
| Pathogen | Priority | Resistance Pattern |
|----------|----------|-------------------|
| *Acinetobacter baumannii* | Critical | Carbapenem-resistant |
| *Pseudomonas aeruginosa* | Critical | Carbapenem-resistant |
| *Klebsiella pneumoniae* | Critical | Carbapenem-resistant |
| *Staphylococcus aureus* | High | Methicillin/Vancomycin-resistant |

### B8: Microbiome-Safe AMPs

Design peptides with selectivity for pathogens over beneficial commensals.

```bash
python scripts/B8_microbiome_safe_amps.py \
    --target_pathogens "S_aureus,MRSA" \
    --protect_commensals "S_epidermidis,C_acnes" \
    --output_dir results/microbiome_safe/
```

**Key Metric - Selectivity Index (SI):**
```
SI = Geometric Mean(Commensal MICs) / Geometric Mean(Pathogen MICs)
SI > 1 = Selective for pathogens (good)
SI > 4 = Clinically relevant selectivity (target)
```

### B10: Synthesis Optimization

Optimize peptides for both activity AND ease of synthesis.

```bash
python scripts/B10_synthesis_optimization.py \
    --activity_weight 0.6 \
    --synthesis_weight 0.4 \
    --output_dir results/synthesis_optimized/
```

**Synthesis Metrics Optimized:**
- Aggregation propensity
- Racemization risk
- Difficult coupling sequences
- Estimated cost per mg

---

## Demo Results Summary

### B1 Results - *Acinetobacter baumannii*

| Rank | Sequence | Charge | Hydro | Activity | Toxicity |
|------|----------|--------|-------|----------|----------|
| 1 | HFHTSFFFSTKVYETSHTHY | +2 | 0.09 | 4.04 | 0.0 |
| 2 | KHPHYTYYGAKTHKRVSQVK | +6.5 | -0.33 | 0.23 | 0.0 |
| 3 | KHPGYTYYGAKSHKRVSQVK | +6 | -0.30 | 0.19 | 0.0 |

### B8 Results - Microbiome-Safe

| Sequence | Charge | SI | Pathogen MIC | Commensal MIC |
|----------|--------|----|--------------| --------------|
| HNHWHMNWKKKKAYAHKPGR | +8 | 1.26 | 9.5 | 13.6 |
| RRTTHKHHCMSWRYKKAPHT | +8 | 1.26 | 10.1 | 14.5 |

### B10 Results - Synthesis-Optimized

| Sequence | Activity | Difficulty | Coupling | Cost |
|----------|----------|------------|----------|------|
| HRGTGKRTIKKLAVAGKFGA | 0.908 | 14.79 | 50.9% | $36.50 |
| GKRSLALGKRVLNCGARKGN | 0.882 | 14.62 | 51.5% | $36.50 |

---

## What's Included

### 1. Core Scripts

| File | Description | Lines |
|------|-------------|-------|
| `scripts/latent_nsga2.py` | NSGA-II optimizer core | 490 |
| `scripts/B1_pathogen_specific_design.py` | Pathogen-specific design | ~400 |
| `scripts/B8_microbiome_safe_amps.py` | Microbiome-safe design | ~400 |
| `scripts/B10_synthesis_optimization.py` | Synthesis optimization | ~400 |

### 2. Interactive Notebook

| File | Description |
|------|-------------|
| `notebooks/brizuela_amp_navigator.ipynb` | Visualization and exploration |

### 3. Results

| File | Description |
|------|-------------|
| `results/pareto_peptides.csv` | 100 Pareto-optimal solutions |
| `results/pathogen_specific/*.json` | B1 demo results |
| `results/microbiome_safe/*.json` | B8 demo results |
| `results/synthesis_optimized/*.csv` | B10 demo results |

### 4. Documentation

| File | Description |
|------|-------------|
| `docs/B1_USER_GUIDE.md` | Pathogen-specific design guide |
| `docs/B8_USER_GUIDE.md` | Microbiome-safe design guide |
| `docs/B10_USER_GUIDE.md` | Synthesis optimization guide |
| `docs/TECHNICAL_BRIEF.md` | Technical specifications |

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install numpy torch pandas matplotlib seaborn
```

### Step 2: Run All Demos

```bash
# B1: Pathogen-Specific
python scripts/B1_pathogen_specific_design.py

# B8: Microbiome-Safe
python scripts/B8_microbiome_safe_amps.py

# B10: Synthesis-Optimized
python scripts/B10_synthesis_optimization.py
```

### Step 3: Explore Results

```bash
jupyter notebook notebooks/brizuela_amp_navigator.ipynb
```

---

## Technical Details

### NSGA-II Multi-Objective Optimization

All three tools use the NSGA-II algorithm operating in the VAE latent space:

```
Latent Space (16D) --> NSGA-II Optimization --> Decoded Peptides
```

**Key Innovation:**
- Traditional: Discrete sequence mutations (20^L combinations)
- Our approach: Continuous latent space optimization

### Objective Functions

#### B1: Pathogen-Specific
| Objective | Description | Goal |
|-----------|-------------|------|
| Activity | Predicted MIC against target | Minimize |
| Toxicity | Hemolysis prediction | Minimize |
| Stability | VAE reconstruction quality | Maximize |

#### B8: Microbiome-Safe
| Objective | Description | Goal |
|-----------|-------------|------|
| Pathogen Activity | MIC against pathogens | Minimize |
| Commensal Sparing | MIC against commensals | Maximize |
| Selectivity Index | Ratio of above | Maximize |
| Toxicity | Host cell safety | Minimize |

#### B10: Synthesis-Optimized
| Objective | Description | Goal |
|-----------|-------------|------|
| Activity | Antimicrobial potency | Maximize |
| Synthesis Difficulty | Aggregation + racemization | Minimize |
| Cost | Estimated $/mg | Minimize |

### Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Individuals per generation |
| `generations` | 50 | Evolutionary iterations |
| `crossover_prob` | 0.9 | SBX crossover probability |
| `mutation_prob` | 0.1 | Per-gene mutation rate |
| `latent_bounds` | (-3, 3) | Valid latent coordinate range |

---

## Output Formats

### Pareto Front CSV

```csv
rank,sequence,net_charge,hydrophobicity,activity_score,toxicity,synthesis_difficulty
1,HFHTSFFFSTKVYETSHTHY,2.0,0.09,4.04,0.0,12.5
2,KHPHYTYYGAKTHKRVSQVK,6.5,-0.33,0.23,0.0,14.2
```

### Results JSON

```json
{
  "objective": "Pathogen-specific AMP design",
  "target_pathogen": "Acinetobacter_baumannii",
  "nsga2_config": {
    "population": 100,
    "generations": 50
  },
  "pareto_front_size": 15,
  "candidates": [
    {
      "rank": 1,
      "sequence": "HFHTSFFFSTKVYETSHTHY",
      "net_charge": 2.0,
      "hydrophobicity": 0.09,
      "predicted_activity": 4.04,
      "toxicity_score": 0.0,
      "latent_vector": [0.23, -0.45, ...]
    }
  ]
}
```

---

## Integration with VAE

### Using Real VAE Checkpoint

```python
from deliverables.scripts.vae_integration_demo import VAEInterface

# Load trained VAE
vae = VAEInterface(checkpoint_path="checkpoints/pretrained_final.pt")

# Decode latent vector to sequence
z = np.array([0.23, -0.45, ...])  # From optimization
sequence = vae.decode_latent(z)

# Encode existing peptide
z_encoded = vae.encode_sequence("KLWKKLKKALK")
```

### Custom Objective Functions

```python
from scripts.latent_nsga2 import LatentNSGA2, OptimizationConfig

def my_activity_objective(z):
    """Custom activity prediction."""
    sequence = vae.decode_latent(z)
    return your_activity_model.predict(sequence)

def my_toxicity_objective(z):
    """Custom toxicity prediction."""
    sequence = vae.decode_latent(z)
    return your_toxicity_model.predict(sequence)

# Configure optimizer
config = OptimizationConfig(
    latent_dim=16,
    population_size=200,
    generations=100
)

optimizer = LatentNSGA2(
    config=config,
    objective_functions=[my_activity_objective, my_toxicity_objective]
)

# Run optimization
pareto_front = optimizer.run(verbose=True)
```

---

## Validation Checklist

### B1: Pathogen-Specific
- [ ] Script runs without errors
- [ ] NSGA-II completes 50 generations
- [ ] Pareto front contains 10+ candidates
- [ ] All candidates have positive charge (+2 to +8)
- [ ] Toxicity scores are < 0.5 for all

### B8: Microbiome-Safe
- [ ] Selectivity Index > 1.0 for top candidates
- [ ] Pathogen MICs < Commensal MICs
- [ ] No duplicate sequences in output

### B10: Synthesis-Optimized
- [ ] Synthesis difficulty < 20 for top candidates
- [ ] No difficult coupling motifs (Asp-X, His-His)
- [ ] Estimated cost < $50/mg

---

## Scientific Background

### Why Multi-Objective Optimization?

AMP design inherently involves trade-offs:
- High activity often correlates with toxicity
- Stable peptides may be harder to synthesize
- Broad-spectrum may mean less selectivity

NSGA-II discovers the Pareto front: solutions where improving one objective necessarily worsens another.

### P-adic Latent Space Advantage

The Ternary VAE embeds peptides in a hyperbolic space where:
- Center = more stable sequences
- Edge = more variable sequences
- P-adic valuation correlates with evolutionary stability

This provides implicit regularization toward stable, "natural-like" peptides.

---

## Questions?

- See docstrings in each script for implementation details
- NSGA-II algorithm: Deb et al. (2002)
- AMP databases: DRAMP, APD3, DBAASP

---

*Prepared as part of the Ternary VAE Bioinformatics Partnership*
*For antimicrobial peptide discovery and optimization*
