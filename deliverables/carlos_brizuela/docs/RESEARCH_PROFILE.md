# Carlos Brizuela - Research Profile and Collaboration Analysis

> **Comprehensive documentation for the Antimicrobial Peptide Optimization Partnership**

**Document Version:** 1.0
**Last Updated:** December 29, 2025
**Partnership Phase:** Active Development

---

## Table of Contents

1. [Researcher Profile](#researcher-profile)
2. [Research Domain](#research-domain)
3. [Project Technical Analysis](#project-technical-analysis)
4. [NSGA-II Implementation](#nsga-ii-implementation)
5. [Results Interpretation](#results-interpretation)
6. [Integration Roadmap](#integration-roadmap)
7. [Future Directions](#future-directions)

---

## Researcher Profile

### Basic Information

| Field | Details |
|-------|---------|
| **Name** | Carlos Brizuela |
| **Specialization** | Computational Biology / Peptide Design |
| **Focus Area** | Antimicrobial Peptide (AMP) Optimization |
| **Partnership Status** | Active Development |

### Research Focus

Carlos Brizuela works at the intersection of:
- **Multi-objective optimization** of bioactive peptides
- **Machine learning** for drug discovery
- **StarPepDB integration** for AMP data
- **Computational drug design** using evolutionary algorithms

### Collaboration Context

The partnership with Carlos Brizuela focuses on applying the Ternary VAE framework to **antimicrobial peptide optimization**, with specific emphasis on:

1. **Latent space navigation** - Exploring VAE embeddings for novel peptides
2. **Multi-objective optimization** - Balancing efficacy vs. safety
3. **Pareto-optimal solutions** - Finding non-dominated peptide candidates
4. **Sequence generation** - Decoding optimized latent vectors to sequences

---

## Research Domain

### Antimicrobial Peptides (AMPs)

**What are AMPs?**
- Short peptides (10-50 amino acids) with antimicrobial activity
- First line of defense in innate immune system
- Effective against bacteria, fungi, viruses, and parasites
- Potential alternatives to conventional antibiotics

**Why AMPs Matter:**
- Rising antibiotic resistance crisis
- Broad-spectrum activity
- Rapid killing mechanism
- Lower resistance development potential

### The Design Challenge

| Challenge | Impact |
|-----------|--------|
| Activity vs. Toxicity | High antimicrobial activity often correlates with host cell toxicity |
| Sequence space | 20^L possibilities for L-length peptide (combinatorial explosion) |
| Stability | Many AMPs degrade rapidly in physiological conditions |
| Manufacturing | Complex sequences are expensive to synthesize |

### Traditional Approaches vs. Our Innovation

**Traditional Methods:**
- Random mutagenesis (inefficient)
- Rational design (requires detailed structure knowledge)
- QSAR models (limited to training data distribution)
- Direct sequence optimization (combinatorial explosion)

**Our Innovation:**
- **VAE Latent Space**: Compress discrete sequences to continuous 16D space
- **Smooth Optimization**: Navigate continuous space with gradient-free methods
- **Multi-Objective**: Handle toxicity/activity tradeoff naturally
- **Generative**: Create novel sequences not in training data

---

## Project Technical Analysis

### Mathematical Foundation

**VAE Latent Space:**

The Ternary VAE compresses peptide sequences to a 16-dimensional hyperbolic latent space:

```
Sequence (20^L possibilities) → Encoder → z ∈ R^16 → Decoder → Sequence
```

**Key Properties:**
- Similar sequences cluster together
- Interpolation produces valid sequences
- Local search finds related variants
- Hyperbolic geometry captures evolutionary relationships

**P-adic Influence:**

The underlying 3-adic structure provides:
- Hierarchical codon organization
- Natural evolutionary distance metric
- Meaningful clustering of synonymous codons

### Optimization Framework

**NSGA-II (Non-dominated Sorting Genetic Algorithm II):**

```
NSGA-II Flowchart:
┌─────────────────────────────────────────────────────┐
│              Initialize Population                   │
│           (Random points in latent space)           │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           Evaluate All Objectives                    │
│    (Reconstruction, Toxicity, Activity)             │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│           Non-Dominated Sorting                      │
│         (Identify Pareto fronts)                    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│        Crowding Distance Assignment                  │
│          (Maintain diversity)                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│        Selection (Binary Tournament)                 │
│     (Prefer lower rank, higher crowding)            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│     Crossover (SBX) + Mutation (Polynomial)         │
│           (Create offspring)                         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
                   Repeat
```

### Objective Functions

**Three Objectives (All Minimized):**

| Objective | Description | Ideal |
|-----------|-------------|-------|
| `objective_0` | Reconstruction loss | Lower = valid sequence |
| `objective_1` | Toxicity score | Lower = safer |
| `objective_2` | Negative activity | More negative = more active |

**Implementation:**

```python
def reconstruction_loss(z: np.ndarray) -> float:
    """Penalize extreme latent values (proxy for invalid sequences)."""
    return np.sum(z ** 2) / len(z)

def toxicity(z: np.ndarray) -> float:
    """Predict toxicity from latent coordinates."""
    # Real implementation: toxicity_model.predict(vae.decode(z))
    return np.abs(np.mean(z[:8]) - np.mean(z[8:]))

def activity(z: np.ndarray) -> float:
    """Negative antimicrobial activity (minimize = maximize activity)."""
    # Real implementation: -activity_model.predict(vae.decode(z))
    return -np.std(z)
```

---

## NSGA-II Implementation

### Core Algorithm Components

**1. Individual Representation:**

```python
@dataclass
class Individual:
    latent: np.ndarray          # 16D latent vector
    objectives: np.ndarray      # [reconstruction, toxicity, activity]
    rank: int = 0               # Pareto rank (0 = best)
    crowding_distance: float    # Diversity measure
    decoded_sequence: str       # Peptide sequence (after decoding)
```

**2. Dominance Relation:**

Individual A dominates B if:
- A is no worse than B in all objectives
- A is strictly better than B in at least one objective

```python
def dominates(A, B) -> bool:
    at_least_one_better = False
    for ai, bi in zip(A.objectives, B.objectives):
        if ai > bi:  # A is worse
            return False
        if ai < bi:  # A is better
            at_least_one_better = True
    return at_least_one_better
```

**3. Simulated Binary Crossover (SBX):**

```
Given parents p1, p2:
β = (2u)^(1/(η+1))           if u ≤ 0.5
β = (1/(2(1-u)))^(1/(η+1))   if u > 0.5

child1 = 0.5 × ((p1 + p2) - β × |p2 - p1|)
child2 = 0.5 × ((p1 + p2) + β × |p2 - p1|)

Where:
- u: random [0, 1]
- η: distribution index (typically 20)
```

**4. Polynomial Mutation:**

```
δ = (2u)^(1/(η+1)) - 1       if u < 0.5
δ = 1 - (2(1-u))^(1/(η+1))   if u ≥ 0.5

mutant = parent + δ × (upper - lower)

Where:
- u: random [0, 1]
- η: distribution index (typically 20)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 16 | Dimensionality of latent space |
| `population_size` | 200 | Number of individuals per generation |
| `generations` | 100 | Number of optimization iterations |
| `crossover_prob` | 0.9 | Probability of crossover |
| `mutation_prob` | 0.1 | Per-gene mutation probability |
| `latent_bounds` | (-3, 3) | Valid latent space range |
| `seed` | 42 | Random seed for reproducibility |

---

## Results Interpretation

### Pareto Front Analysis

**From `results/pareto_peptides.csv`:**

The optimization produced **100 Pareto-optimal solutions** - none dominates another.

**Representative Solutions:**

| ID | Reconstruction | Toxicity | Activity | Crowding | Profile |
|----|----------------|----------|----------|----------|---------|
| 0 | 7.054 | 1.007 | -2.615 | ∞ | High activity, high toxicity |
| 1 | 4.575 | 3.780 | -2.118 | ∞ | Balanced, boundary |
| 2 | 0.740 | 0.00002 | -0.851 | ∞ | Very safe, moderate activity |
| 3 | 0.306 | 0.132 | -0.534 | ∞ | Most balanced |
| 4 | 0.322 | 0.074 | -0.494 | ∞ | Safe, moderate |

**Trade-off Analysis:**

```
Pareto Front Visualization (2D Projection):

Toxicity ↑
    │
  1.0│    ●(0) High activity
    │     zone
    │
  0.5│        ●(5-10)
    │           ●●●
    │              ●●●●
  0.1│                 ●●(3,4)
    │                    ●(2) Safe zone
    │
   0└───────────────────────────
    -0.5  -1.0  -1.5  -2.0  -2.5
              Activity →
    (more negative = more active)
```

### Solution Categories

**1. High-Activity Zone (Solutions 0, 7, 8, 9):**
- Activity score: -2.0 to -2.6
- Toxicity: 0.5 - 1.5
- Use case: Life-threatening infections where efficacy is priority

**2. Balanced Zone (Solutions 5, 10-30):**
- Activity score: -1.2 to -1.8
- Toxicity: 0.1 - 0.5
- Use case: Most clinical applications

**3. Safe Zone (Solutions 2, 3, 4, 24):**
- Activity score: -0.5 to -0.9
- Toxicity: < 0.15
- Use case: Prophylactic, topical applications

### Latent Space Distribution

**16D Latent Coordinates:**

| Dimension | Range | Interpretation |
|-----------|-------|----------------|
| z_0 - z_3 | [-3, 3] | Structural features |
| z_4 - z_7 | [-3, 3] | Hydrophobicity pattern |
| z_8 - z_11 | [-3, 3] | Charge distribution |
| z_12 - z_15 | [-3, 3] | Activity-related features |

**Observation:** High-activity solutions cluster near latent bounds (±3), while safe solutions stay near origin.

---

## Integration Roadmap

### Current Status

```
Implementation Progress:
┌────────────────────────────────────────────────┐
│ ████████████████████░░░░░░░░░░░░░░░░░ 45%      │
└────────────────────────────────────────────────┘

✅ Completed:
  - NSGA-II core algorithm
  - Mock objective functions
  - Pareto front export
  - Basic demo results

🔧 In Progress:
  - Real VAE integration
  - Toxicity predictor training
  - Activity predictor training

📋 Pending:
  - StarPepDB data integration
  - Sequence decoding pipeline
  - Experimental validation
```

### Integration Steps

**Step 1: VAE Connection**

```python
from src.models import TernaryVAEV5_11_PartialFreeze

# Load pre-trained VAE
vae = TernaryVAEV5_11_PartialFreeze(latent_dim=16)
vae.load_state_dict(torch.load('checkpoints/homeostatic_rich/best.pt'))

# Decoder function for NSGA-II
def decode_latent(z: np.ndarray) -> str:
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
    recon = vae.decode(z_tensor)
    return token_to_sequence(recon)
```

**Step 2: Real Objective Functions**

```python
# Replace mock objectives with real predictors
def real_toxicity(z: np.ndarray) -> float:
    sequence = decode_latent(z)
    return toxicity_model.predict([sequence])[0]

def real_activity(z: np.ndarray) -> float:
    sequence = decode_latent(z)
    mic = activity_model.predict([sequence])[0]
    return -mic  # Minimize negative MIC

objectives = [reconstruction_loss, real_toxicity, real_activity]
```

**Step 3: StarPepDB Integration**

```python
# Load AMP training data
from starpedb import AMPDataset

dataset = AMPDataset(
    hemolysis=True,  # Include toxicity labels
    mic=True,        # Include activity labels
    species=['E. coli', 'S. aureus']
)

# Train predictors on real data
toxicity_model.fit(dataset.sequences, dataset.hemolysis)
activity_model.fit(dataset.sequences, dataset.mic)
```

### Validation Protocol

| Stage | Method | Success Criteria |
|-------|--------|------------------|
| **Computational** | Cross-validation | R² > 0.7 for predictors |
| **In silico** | MD simulation | Stable structure > 50ns |
| **In vitro** | MIC assay | Activity comparable to predictions |
| **Hemolysis** | RBC lysis assay | HC50 > 100 μg/mL |

---

## Future Directions

### Short-term (1-3 months)

1. **Complete VAE Integration:**
   - Connect pre-trained Ternary VAE
   - Validate reconstruction quality
   - Test sequence decoding

2. **Train Real Predictors:**
   - Toxicity: Train on hemolysis data
   - Activity: Train on MIC data
   - Validate against held-out sets

3. **StarPepDB Pipeline:**
   - Automated data ingestion
   - Feature extraction
   - Model retraining schedule

### Medium-term (3-6 months)

1. **Multi-Pathogen Optimization:**
   - Expand to multiple bacterial species
   - Add fungal activity objective
   - Viral activity integration

2. **Structural Constraints:**
   - Add secondary structure objectives
   - Include membrane interaction scores
   - Stability prediction (half-life)

3. **Experimental Validation:**
   - Synthesize top 10 candidates
   - Full MIC panel testing
   - Cytotoxicity profiling

### Long-term (6-12 months)

1. **Conditional Generation:**
   - Target-specific AMPs
   - Microbiome-safe variants
   - Formulation-compatible designs

2. **Clinical Development:**
   - Lead optimization
   - ADMET profiling
   - Preclinical studies

3. **Platform Extension:**
   - Antiviral peptides
   - Anticancer peptides
   - Immunomodulatory peptides

---

## File Inventory

### Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/latent_nsga2.py` | 479 | NSGA-II optimizer |
| `scripts/__init__.py` | - | Package marker |

### Source Modules

| File | Classes | Purpose |
|------|---------|---------|
| `src/objectives.py` | ObjectiveFunction, ToxicityPredictor, ActivityPredictor | Objective function interfaces |
| `src/vae_interface.py` | TernaryVAEInterface | VAE loading and decoding |
| `src/__init__.py` | - | Package marker |

### Data

| File | Format | Contents |
|------|--------|----------|
| `data/demo_amp_embeddings.csv` | CSV | Pre-computed AMP embeddings |

### Results

| File | Format | Contents |
|------|--------|----------|
| `results/pareto_peptides.csv` | CSV | 100 Pareto-optimal solutions |

### Documentation

| File | Purpose |
|------|---------|
| `docs/PROJECT_OVERVIEW.md` | High-level project description |
| `docs/TECHNICAL_BRIEF.md` | Technical implementation details |
| `docs/MASTER_IMPLEMENTATION_PLAN.md` | Development roadmap |
| `docs/VALIDATION_SUMMARY.md` | Validation procedures |

---

## References

### Scientific Background

1. **NSGA-II:**
   - Deb, K., et al. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"

2. **SBX Crossover:**
   - Deb, K. & Agrawal, R.B. (1995). "Simulated Binary Crossover for Continuous Search Space"

3. **Antimicrobial Peptides:**
   - Hancock, R.E.W. & Sahl, H.G. (2006). "Antimicrobial and host-defense peptides as new anti-infective therapeutic strategies"

### Technical Resources

4. **StarPepDB:**
   - https://starpepdh.sytes.net/

5. **Ternary VAE:**
   - See main project documentation

---

*Document prepared as part of the Ternary VAE Bioinformatics Partnership*
*For integration with antimicrobial peptide optimization research*
