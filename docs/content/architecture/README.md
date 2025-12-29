# Architecture Overview

> **Single source of truth for Ternary VAE architecture.**

**Last Updated**: 2025-12-28

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Directory Structure](#directory-structure)
4. [Architecture Improvements (2025)](#architecture-improvements-2025)
5. [Configuration](#configuration)
6. [Performance](#performance)

---

## System Overview

### Dual-VAE Architecture

The model consists of two VAEs that work together:

```
Input → [VAE-A Encoder] → z_A (exploration)
      → [VAE-B Encoder] → z_B (refinement)
      → [StateNet] → control signals (rho, lambda adjustments)
      → [Decoder] → reconstruction
```

- **VAE-A**: Explores the ternary operation space (chaotic)
- **VAE-B**: Refines and stabilizes (anchor)
- **StateNet**: Meta-controller that balances both VAEs

### Hyperbolic Geometry

The latent space uses Poincare ball geometry:
- Points near origin = high-valuation (simple operations)
- Points near boundary = low-valuation (complex operations)
- Geodesic distance encodes 3-adic relationships

### Loss System

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

## Core Components

### Source Code Structure

```
src/
├── models/           # VAE architectures
│   ├── ternary_vae.py       # Production model (V5.11)
│   ├── base_vae.py          # Unified base class
│   ├── structure_aware_vae.py # AlphaFold2 integration
│   ├── epistasis_module.py  # Mutation interactions
│   ├── swarm_vae.py         # Swarm intelligence
│   └── maml_vae.py          # Meta-learning
├── losses/           # Loss functions
│   ├── dual_vae_loss.py     # Complete loss system
│   ├── padic_losses.py      # 3-adic geometry losses
│   ├── epistasis_loss.py    # Epistasis loss
│   └── hyperbolic_*.py      # Hyperbolic losses
├── training/         # Training infrastructure
│   ├── trainer.py           # TernaryVAETrainer
│   ├── transfer_pipeline.py # Multi-disease transfer
│   ├── data.py              # Datasets, samplers
│   └── schedulers.py        # Parameter scheduling
├── diseases/         # Multi-disease framework (11 diseases)
│   ├── base.py              # DiseaseAnalyzer base
│   ├── uncertainty_aware_analyzer.py # Uncertainty
│   ├── hiv_analyzer.py      # HIV (23 ARVs)
│   ├── tuberculosis_analyzer.py # TB (MDR/XDR)
│   └── ...                  # 9 more analyzers
├── encoders/         # Specialized encoders
│   ├── alphafold_encoder.py # SE(3)-equivariant
│   └── codon_encoder.py     # Codon encoding
├── biology/          # Biological constants
│   ├── amino_acids.py
│   └── codons.py            # Single source of truth
└── geometry/         # Hyperbolic geometry
    └── poincare.py
```

### Key Modules

| Module | Purpose | Tests |
|--------|---------|-------|
| `models/base_vae.py` | Unified base class for 19+ VAE variants | 33 tests |
| `models/structure_aware_vae.py` | AlphaFold2 + SE(3) integration | 35 tests |
| `models/epistasis_module.py` | Mutation interaction modeling | 32 tests |
| `losses/epistasis_loss.py` | Unified epistasis loss | 29 tests |
| `diseases/uncertainty_aware_analyzer.py` | Uncertainty quantification | 21 tests |
| `training/transfer_pipeline.py` | Cross-disease transfer | 30 tests |
| `encoders/alphafold_encoder.py` | Structure encoding | 18 tests |

---

## Directory Structure

```
ternary-vaes-bioinformatics/
├── src/                    # Core Python library
├── scripts/                # Entry points
│   ├── train/              # Training scripts
│   ├── analysis/           # Analysis tools
│   └── visualization/      # Visualization
├── tests/                  # Test suite (231 tests)
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── configs/                # YAML configurations
├── data/                   # Research datasets
├── research/               # Research experiments
├── results/                # Generated outputs
├── docs/                   # Documentation (YOU ARE HERE)
│   ├── content/            # Consolidated docs
│   └── source/             # Sphinx API docs
└── DOCUMENTATION/          # Legacy docs (being consolidated)
```

---

## Architecture Improvements (2025)

### Phase 1: BaseVAE Abstraction

Unified base class reducing code duplication across 19+ VAE variants:

```python
from src.models.base_vae import BaseVAE, VAEConfig, VAEOutput

class MyCustomVAE(BaseVAE):
    def encode(self, x) -> Tuple[Tensor, Tensor]:
        return mu, logvar

    def decode(self, z) -> Tensor:
        return reconstruction
```

**Details**: [base-vae.md](base-vae.md)

### Phase 2: Uncertainty Integration

Integrated uncertainty methods for clinical decision support:

| Method | Forward Passes | Decomposition |
|--------|----------------|---------------|
| MC Dropout | N (e.g., 50) | Epistemic only |
| Evidential | 1 | Full (epistemic + aleatoric) |
| Ensemble | M (e.g., 5) | Epistemic only |

```python
from src.diseases.uncertainty_aware_analyzer import (
    UncertaintyAwareAnalyzer, UncertaintyConfig, UncertaintyMethod
)

config = UncertaintyConfig(method=UncertaintyMethod.EVIDENTIAL)
analyzer = UncertaintyAwareAnalyzer(base_analyzer, config=config, model=model)
results = analyzer.analyze_with_uncertainty(sequences, encodings=x)
```

**Details**: [uncertainty.md](uncertainty.md)

### Phase 3: Epistasis Module

Mutation interaction modeling for drug resistance:

```python
from src.models.epistasis_module import EpistasisModule

epistasis = EpistasisModule(n_positions=300, embed_dim=64)
result = epistasis(positions=torch.tensor([[65, 184, 215]]))
# result.interaction_score, result.synergistic, result.antagonistic
```

**Details**: [epistasis.md](epistasis.md)

### Phase 4: Transfer Learning Pipeline

Multi-disease transfer learning strategies:

| Strategy | Description | Best For |
|----------|-------------|----------|
| FROZEN_ENCODER | Freeze encoder, train head | <1000 samples |
| FULL_FINETUNE | Train all parameters | >5000 samples |
| ADAPTER | Add adapter modules | Moderate data |
| LORA | Low-rank adaptation | Large models |
| MAML | Meta-learning | Few-shot (5-50) |

```python
from src.training.transfer_pipeline import TransferLearningPipeline, TransferConfig

pipeline = TransferLearningPipeline(config)
pretrained = pipeline.pretrain(all_disease_data)
finetuned = pipeline.finetune("hiv", hiv_data)
```

**Details**: [transfer.md](transfer.md)

### Phase 5: Structure-Aware VAE

AlphaFold2 integration with SE(3)-equivariant encoding:

```python
from src.models.structure_aware_vae import StructureAwareVAE, StructureConfig

config = StructureConfig(use_structure=True, fusion_type="cross_attention")
model = StructureAwareVAE(input_dim=128, latent_dim=32, structure_config=config)
outputs = model(x=seq_embed, structure=coords, plddt=confidence)
```

**Details**: [structure.md](structure.md)

### Phase 6: Testing

Comprehensive test suite with 231 tests (97.4% pass rate):

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Configuration

### Runtime Configuration

| Location | Purpose | Format |
|----------|---------|--------|
| `configs/` | Runtime parameters | YAML/JSON |
| `src/config/` | Configuration classes | Python |

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

## Related Documentation

| Topic | Link |
|:------|:-----|
| BaseVAE Deep Dive | [base-vae.md](base-vae.md) |
| Uncertainty Quantification | [uncertainty.md](uncertainty.md) |
| Epistasis Modeling | [epistasis.md](epistasis.md) |
| Transfer Learning | [transfer.md](transfer.md) |
| Structure-Aware Modeling | [structure.md](structure.md) |
| Theory Foundations | [../theory/README.md](../theory/README.md) |

---

_Last updated: 2025-12-28_
