# Implementation Plan: Antimicrobial Peptide Optimization

> **Technical Roadmap for Multi-Objective AMP Design**

This document details the technical implementation steps for your antimicrobial peptide optimization project using NSGA-II in VAE latent space.

---

## Project Focus

| Aspect | Details |
|--------|---------|
| **Domain** | Antimicrobial Peptides (AMPs) |
| **Key Deliverable** | `AMP_Hyperbolic_Navigator.ipynb` |
| **Core Script** | `scripts/latent_nsga2.py` |

---

## Data Acquisition (StarPepDB)

- **Source:** StarPepDB (Graph Database of AMPs)
- **Existing Tool:** `scripts/ingest/ingest_starpep.py`
- **Critical Fields:**
  - Hemolytic Activity (toxicity metric)
  - MIC (Minimum Inhibitory Concentration)
  - Sequence and length
  - Activity against specific pathogens

### Data Format

```json
{
  "id": "starPep_00001",
  "sequence": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
  "length": 37,
  "hemolytic_activity": 0.23,
  "mic_ecoli": 4.0,
  "mic_saureus": 8.0,
  "net_charge": 6,
  "hydrophobicity": 0.45
}
```

---

## Implementation Components

### 1. Hyperbolic AMP Navigator

**Script:** `scripts/analysis/amp_hyperbolic_navigator.py`

**Logic:**
1. Load AMP data from `starpep_data.json`
2. Encode sequences into 5D Hyperbolic Space using the Ternary VAE
3. Compute "Geodesic Paths" from toxic AMPs to non-toxic clusters
4. Visualize the peptide landscape

### 2. NSGA-II Latent Optimizer

**Script:** `scripts/optimization/latent_nsga2.py`

**Logic:**
- Decision variables: Latent coordinates (z_1, ..., z_16)
- NOT discrete amino acids (avoids combinatorial explosion)
- Multi-objective optimization in continuous space

**Objectives (all minimized):**
1. Reconstruction loss (sequence validity)
2. Toxicity score (hemolytic activity)
3. Negative activity (more negative = better)

---

## Algorithm Configuration

### NSGA-II Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `population_size` | 200 | Individuals per generation |
| `generations` | 100 | Evolutionary cycles |
| `latent_dim` | 16 | VAE latent dimensionality |
| `crossover_prob` | 0.9 | SBX crossover probability |
| `mutation_prob` | 0.1 | Per-gene mutation rate |
| `latent_bounds` | (-3, 3) | Valid latent range |

### Genetic Operators

**SBX Crossover:**
- Simulates single-point crossover behavior in continuous space
- Distribution index controls offspring spread

**Polynomial Mutation:**
- Bounded perturbation of latent coordinates
- Respects latent space boundaries

---

## Output Specification

### Pareto Front CSV

```csv
id,rank,crowding_distance,objective_0,objective_1,objective_2,z_0,z_1,...,z_15
0,0,inf,0.305,0.132,-0.534,-0.234,1.456,...,0.678
1,0,2.341,0.412,0.089,-0.623,0.123,-0.987,...,0.234
```

### Interpretation

- **rank=0**: All solutions on the Pareto-optimal front
- **crowding_distance=inf**: Boundary solutions (extreme objectives)
- **objective_2 more negative**: Higher antimicrobial activity

---

## Integration Steps

### Step 1: Prepare Your VAE

```python
from models.ternary_vae import TernaryVAE

vae = TernaryVAE.load("path/to/checkpoint.pt")
```

### Step 2: Define Objective Functions

```python
def reconstruction_objective(z):
    decoded = vae.decode(torch.tensor(z))
    return float(vae.reconstruction_loss(decoded))

def toxicity_objective(z):
    sequence = vae.decode_to_sequence(z)
    return float(toxicity_model.predict(sequence))

def activity_objective(z):
    sequence = vae.decode_to_sequence(z)
    return -float(activity_model.predict(sequence))
```

### Step 3: Run Optimization

```python
from latent_nsga2 import LatentNSGA2, OptimizationConfig

config = OptimizationConfig(
    latent_dim=16,
    population_size=200,
    generations=100
)

optimizer = LatentNSGA2(
    config=config,
    objective_functions=[reconstruction_objective, toxicity_objective, activity_objective],
    decoder=lambda z: vae.decode_to_sequence(z)
)

pareto_front = optimizer.run(verbose=True)
optimizer.export_pareto_front("results/optimized_amps.csv")
```

---

## Validation Workflow

1. **Demo Run**: Execute with mock objectives (no VAE required)
2. **VAE Integration**: Connect your trained model
3. **Objective Calibration**: Verify objective function outputs
4. **Pareto Analysis**: Examine trade-offs in results
5. **Sequence Decoding**: Convert best latent vectors to sequences
6. **Lab Validation**: Synthesize and test top candidates

---

## Expected Results

With real StarPepDB data and trained models:

- **100+ Pareto-optimal solutions** covering the activity-toxicity trade-off
- **Clear clustering** of low-toxicity candidates
- **Novel sequences** not in training data (generative capability)
- **Interpretable trade-offs** for rational selection

---

## Dependencies

```bash
pip install numpy torch pandas matplotlib seaborn
```

---

*Implementation Plan for Carlos Brizuela - Antimicrobial Peptide Optimization*
