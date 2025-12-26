# Architecture Overview

This document provides a high-level overview of the Ternary VAE codebase architecture.

**Last Updated**: 2025-12-26

---

## Directory Structure

```
ternary-vaes-bioinformatics/
├── src/                          # Core library
│   ├── models/                   # VAE architectures
│   │   ├── ternary_vae.py       # Current production model (V5.11)
│   │   ├── swarm_vae.py         # Swarm intelligence VAE
│   │   ├── homeostasis.py       # Homeostatic regulation
│   │   └── curriculum.py        # Curriculum learning
│   ├── losses/                   # Loss functions
│   │   ├── dual_vae_loss.py     # Complete loss system
│   │   ├── padic_losses.py      # 3-adic geometry losses
│   │   ├── hyperbolic_prior.py  # Hyperbolic prior losses
│   │   ├── hyperbolic_recon.py  # Hyperbolic reconstruction losses
│   │   └── ...
│   ├── training/                 # Training infrastructure
│   │   ├── trainer.py           # TernaryVAETrainer (main loop)
│   │   ├── hyperbolic_trainer.py # HyperbolicVAETrainer (geometry-aware)
│   │   ├── base.py              # BaseTrainer with defensive patterns
│   │   ├── data.py              # TernaryDataset, StratifiedBatchSampler
│   │   ├── schedulers.py        # Parameter scheduling
│   │   ├── monitor.py           # Logging and metrics
│   │   └── environment.py       # Pre-training environment validation
│   ├── biology/                  # Centralized biology constants (NEW)
│   │   ├── amino_acids.py       # Amino acid properties and mappings
│   │   └── codons.py            # Genetic code, codon indices
│   ├── analysis/                 # Analysis modules
│   │   ├── immunology/          # Shared immunology utilities (NEW)
│   │   │   ├── epitope_encoding.py  # Epitope sequence encoding
│   │   │   ├── genetic_risk.py      # HLA risk computation
│   │   │   ├── padic_utils.py       # P-adic valuation utilities
│   │   │   └── types.py             # EpitopeAnalysisResult, HLAAlleleRisk
│   │   ├── crispr/              # CRISPR off-target analysis
│   │   └── ...
│   ├── optimizers/               # Custom optimizers
│   │   ├── riemannian.py        # Riemannian optimizers (Poincaré, Lorentz)
│   │   └── multi_objective.py   # Multi-objective optimization
│   ├── encoders/                 # Specialized encoders
│   │   ├── circadian_encoder.py # Circadian rhythm encoding
│   │   ├── surface_encoder.py   # Protein surface encoding
│   │   └── ...
│   ├── data/                     # Data handling
│   │   ├── generation.py        # Ternary operation generation
│   │   └── dataset.py           # PyTorch datasets
│   ├── artifacts/                # Checkpoint management
│   └── utils/                    # Metrics and utilities
├── scripts/                      # Entry points
│   ├── train.py                 # Main training entry point
│   ├── train_codon_vae_hiv.py   # HIV-specific codon VAE training
│   ├── analyze_all_datasets.py  # Comprehensive dataset analysis
│   ├── clinical_applications.py # Clinical decision support
│   ├── research_discoveries.py  # Research findings pipeline
│   ├── benchmark/                # Benchmarking
│   └── visualization/            # Visualization tools
├── configs/                      # YAML configurations
├── tests/                        # Test suite
│   └── unit/
│       ├── training/            # Training tests (hyperbolic, monitor)
│       └── visualization/       # Visualization tests
├── research/                     # Research experiments
│   ├── alphafold3/              # AlphaFold3 integration
│   └── bioinformatics/          # Bioinformatics research
│       └── codon_encoder_research/
│           └── hiv/             # HIV analysis (200K+ sequences)
├── results/                      # Analysis outputs
│   ├── clinical_applications/   # Clinical decision support reports
│   └── research_discoveries/    # Research findings reports
└── DOCUMENTATION/                # Comprehensive docs
```

---

## Core Components

### 1. Dual-VAE Architecture

The model consists of two VAEs that work together:

```
Input → [VAE-A Encoder] → z_A (exploration)
      → [VAE-B Encoder] → z_B (refinement)
      → [StateNet] → control signals (ρ, λ adjustments)
      → [Decoder] → reconstruction
```

- **VAE-A**: Explores the ternary operation space (chaotic)
- **VAE-B**: Refines and stabilizes (anchor)
- **StateNet**: Meta-controller that balances both VAEs

### 2. Hyperbolic Geometry

The latent space uses Poincare ball geometry:
- Points near origin = high-valuation (simple operations)
- Points near boundary = low-valuation (complex operations)
- Geodesic distance encodes 3-adic relationships

### 3. Loss System

```python
total_loss = (
    reconstruction_loss +        # Cross-entropy
    beta_A * kl_divergence_A +   # VAE-A regularization
    beta_B * kl_divergence_B +   # VAE-B regularization
    lambda_3 * padic_loss +      # 3-adic structure
    entropy_regularization       # Output diversity
)
```

---

## Training Flow

```
1. Load config (YAML)
2. Initialize model, optimizer, schedulers
3. For each epoch:
   a. Phase-based scheduling (beta, temperature)
   b. Train step with gradient balancing
   c. StateNet corrections
   d. Logging to TensorBoard
   e. Checkpoint if best
4. Final evaluation and reporting
```

---

## Naming Conventions

This project follows PEP 8 naming conventions with additional domain-specific guidelines.

### Files and Modules

```python
# Files: lowercase_snake_case
ternary_vae.py           # Good
ternary_vae_partial_freeze.py  # Good (descriptive variant name)

# Avoid cryptic abbreviations in filenames
ternary_vae_optionc.py   # Deprecated (cryptic)
```

### Classes

```python
# PascalCase with clear version suffix pattern: V{major}_{minor}
TernaryVAEV5_11           # Good
TernaryVAEV5_11_PartialFreeze  # Good (descriptive variant)

# Avoid cryptic option names
TernaryVAEV5_11_OptionC   # Deprecated (use PartialFreeze)

# Domain abbreviations should be clear
PAdicRankingLoss          # Good (P-Adic is domain term)
```

### Functions

```python
# Use action verb prefixes consistently:
#   compute_* - Expensive O(n) operations
#   get_*     - O(1) lookups/retrievals
#   create_*  - Factory functions
#   load_*    - File/resource loading
#   validate_* - Validation checks

compute_padic_distance()       # Expensive calculation
get_amino_acid_property()      # Dictionary lookup
create_model_from_config()     # Factory
load_checkpoint()              # File loading
validate_training_config()     # Validation
```

### Codon/Biology Functions

The canonical source of truth is `src/biology/codons.py`:

```python
from src.biology.codons import (
    codon_to_index,        # Convert codon string to index (0-63)
    index_to_codon,        # Convert index (0-63) to codon string
    triplet_to_codon_index,  # Alias for codon_to_index
    codon_index_to_triplet,  # Alias for index_to_codon
)
```

Do NOT define duplicate codon conversion functions in other modules.

### Variables

```python
# Descriptive names for configuration values
curvature = 2.0           # Good (not 'c')
prime_base = 3            # Good (not 'p')
learning_rate = 1e-3      # Good (not 'lr')

# Short names acceptable in mathematical contexts
z_poincare, z_euclidean   # Good (tensor variables)
mu_A, logvar_A            # Good (VAE standard notation)

# Tensor suffixes indicate space
z_hyp → z_poincare        # Preferred (more specific)
z_euc → z_euclidean       # Preferred (clearer)
```

### Constants

```python
# UPPER_SNAKE_CASE with clear prefixes
N_TERNARY_OPERATIONS = 19683
DEFAULT_LEARNING_RATE = 1e-3
EPSILON_LOG = 1e-10
POINCARE_MAX_NORM = 0.95
```

### Deprecation Pattern

When renaming, preserve old names as aliases:

```python
# New name (preferred)
class TernaryVAEV5_11_PartialFreeze(TernaryVAEV5_11):
    ...

# Deprecated alias for backward compatibility
TernaryVAEV5_11_OptionC = TernaryVAEV5_11_PartialFreeze
```

---

## Key Design Decisions

### Single Responsibility Principle (SRP)
Each module has one job:
- `trainer.py` → training loop only
- `schedulers.py` → parameter scheduling only
- `monitor.py` → logging only
- `dual_vae_loss.py` → loss computation only

### Dependency Injection
Components are injected, not hard-coded:
```python
trainer = TernaryVAETrainer(
    model,
    config,
    device,
    monitor=custom_monitor  # Optional injection
)
```

### Phase-Scheduled Training
Training proceeds in phases:
1. **Phase 1 (0-40)**: VAE-A exploration, β-warmup
2. **Phase 2 (40-49)**: Consolidation
3. **Phase 3 (50)**: Disruption (β-B warmup)
4. **Phase 4 (50+)**: Convergence

---

## New Modules (2025-12-26)

### Biology Module (`src/biology/`)
Centralized biology constants - Single Source of Truth:
```python
from src.biology import (
    GENETIC_CODE,           # Codon → Amino acid mapping
    AMINO_ACID_PROPERTIES,  # Hydrophobicity, charge, volume
    codon_to_amino_acid,    # Conversion utilities
    CODON_TO_INDEX,         # 64 codons indexed
)
```

### Immunology Analysis (`src/analysis/immunology/`)
Shared immunology utilities for disease modules:
```python
from src.analysis.immunology import (
    encode_amino_acid_sequence,  # Epitope encoding
    compute_hla_genetic_risk,    # HLA risk scoring
    compute_padic_valuation,     # P-adic geometry
    compute_goldilocks_score,    # Optimal stability zone
    EpitopeAnalysisResult,       # Result dataclass
)
```

### Training Data Infrastructure (`src/training/data.py`)
Production-ready data handling:
- `TernaryDataset`: PyTorch dataset with stratified sampling
- `StratifiedBatchSampler`: Ensures balanced batches
- `create_stratified_batches`: Batch generation utility

### Hyperbolic VAE Trainer (`src/training/hyperbolic_trainer.py`)
Geometry-aware training loop:
- Riemannian gradient updates on Poincaré ball
- Hyperbolic distance metrics
- Curvature-adaptive learning rates

---

## Bioinformatics Applications

### Codon Encoder Research
Located in `research/bioinformatics/codon_encoder_research/`:
- **HIV**: 200,000+ sequences analyzed
  - Glycan shield analysis
  - Drug resistance correlation (r = 0.41)
  - 328 vaccine targets identified
  - 85% tropism prediction accuracy
- **SARS-CoV-2**: Spike protein analysis
- **Rheumatoid Arthritis**: HLA-autoantigen relationships
- **Neurodegeneration**: Tau phosphorylation

### Clinical Applications (`results/clinical_applications/`)
Generated 2025-12-26:
- **Top Vaccine Candidate**: TPQDLNTML (Gag, priority score: 0.970)
- **MDR Screening**: 2489 high-risk sequences (34.8%)
- **Drug Targets**: 247 Tat-interacting druggable proteins
- Clinical decision support JSON for integration

### Research Discoveries (`results/research_discoveries/`)
Key findings from comprehensive analysis:
- 387 vaccine targets ranked by evolutionary stability
- P-adic vs Hamming correlation: r = 0.8339
- 1032 MDR-enriched mutations identified
- 19 HIV proteins targeting 3+ druggable human proteins
- Top host-directed target: Tat (449 druggable targets)

### AlphaFold3 Integration
Located in `research/alphafold3/`:
- 6300x storage reduction via hybrid approach
- Structural validation pipeline

---

## Configuration

### Directory Organization

The project separates configuration into two locations:

| Location | Purpose | Format |
|----------|---------|--------|
| `configs/` | Runtime parameters (learning rate, epochs, etc.) | YAML/JSON |
| `src/config/` | Configuration classes, schemas, and validation | Python |

- **`configs/`**: User-editable YAML files for training runs and experiments
- **`src/config/`**: Python dataclasses defining the structure and defaults

### Example Config (`configs/ternary.yaml`)

```yaml
model:
  input_dim: 9
  latent_dim: 16
  rho_min: 0.1
  rho_max: 0.7

vae_a:
  beta_start: 0.3
  beta_end: 0.8
  beta_warmup_epochs: 50

vae_b:
  beta_start: 0.0
  beta_end: 0.5
  beta_warmup_epochs: 50

training:
  batch_size: 256
  total_epochs: 300

torch_compile:
  enabled: true
  backend: inductor
```

---

## Performance

| Metric | Value |
|:-------|:------|
| Model Parameters | 168,770 |
| Coverage (hash-validated) | 86-87% |
| Inference Speed (VAE-A) | 4.4M samples/sec |
| Inference Speed (VAE-B) | 6.1M samples/sec |
| Training Speedup (torch.compile) | 1.4-2x |

---

*For detailed API documentation, see `DOCUMENTATION/03_PRESENTATION_TIERS/03_TIER_3_TECHNICAL/`*
