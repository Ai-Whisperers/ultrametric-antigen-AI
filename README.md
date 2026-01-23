# Ternary VAE

[![Version](https://img.shields.io/badge/version-5.12.5-blue.svg)](.claude/CLAUDE.md)
[![License: PolyForm Non-Commercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20NC-lightgrey.svg)](LEGAL_AND_IP/LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LEGAL_AND_IP/RESULTS_LICENSE.md)

---

## Overview

**Ternary VAE** is a variational autoencoder framework that learns hierarchical structure in hyperbolic space using p-adic number theory. The project implements a dual-framework architecture spanning mathematical foundations and practical applications.

### The Core Problem

Standard VAEs using Euclidean geometry fail to capture hierarchical structure because flat space distorts tree-like relationships exponentially. We solve this by matching the geometry of the model to the geometry of the data:

- **P-adic valuation** provides the algebraic hierarchy (3-adic numbers for ternary operations)
- **Hyperbolic geometry** (Poincaré ball) provides the continuous differentiable space
- **The isomorphism**: Low p-adic distance ↔ Close in hyperbolic space; High valuation ↔ Close to origin

### Validated Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Coverage** | 100% | 100% | Achieved |
| **Hierarchy** | -0.8321 | -0.83 | Mathematical ceiling |
| **Richness** | 0.00787 | >0.005 | 5.8x baseline |

The model embeds 19,683 ternary operations ({-1, 0, +1}^9 = 3^9) into a 16-dimensional hyperbolic space—a 1,230x compression while preserving the underlying 3-adic valuation hierarchy.

---

## Dual-Framework Architecture

The project is organized into two complementary tiers:

### TIER 1: Models and Mathematical Foundations

Core AI/ML training infrastructure and mathematical primitives. Highly validated and generalizable.

```
src/
├── core/           # P-adic mathematics, ternary operations, metrics
├── geometry/       # Poincaré ball, hyperbolic distances (geoopt-backed)
├── models/         # VAE architectures (dual-encoder, homeostatic control)
├── training/       # Training loops, optimizations, grokking detection
└── losses/         # Manifold organization, p-adic ranking losses
```

**Key Components:**
- `TernaryVAEV5_11_PartialFreeze`: Dual-encoder architecture with homeostatic controller
- `poincare_distance()`: Correct hyperbolic distance computation
- `padic_valuation()`: 3-adic valuation for hierarchy encoding
- Mixed precision training with torch.compile (3-4x speedup)

**Validated Checkpoints:**

| Checkpoint | Coverage | Hierarchy | Use Case |
|------------|----------|-----------|----------|
| `homeostatic_rich` | 100% | -0.8321 | Semantic reasoning, DDG prediction |
| `v5_12_4` | 100% | -0.82 | General purpose |
| `v5_11_structural` | 100% | -0.74 | Contact prediction (AUC=0.67) |
| `v5_11_progressive` | 100% | +0.78 | Compression, retrieval (frequency-optimal) |

### TIER 2: Applications

Domain-specific applications built on TIER 1 foundations.

#### Bioinformatics (Primary, Most Validated)

The codon-level application of p-adic geometry to biological sequences.

| Application | Metric | Value | Status |
|-------------|--------|-------|--------|
| **DDG Prediction** | LOO Spearman | 0.585 | Validated (S669) |
| **Contact Prediction** | AUC-ROC | 0.67 | Validated |
| **Force Constants** | Correlation | 0.86 | Validated |

**Partner Packages** (`deliverables/partners/`):
- **Jose Colbes**: Protein stability prediction (LOO ρ=0.585)
- **Carlos Brizuela**: AMP optimization (PeptideVAE r=0.63)
- **Alejandra Rojas**: Arbovirus primer design

#### Other Application Domains (Research Stage)

The mathematical framework generalizes beyond bioinformatics:

| Domain | Application | Status |
|--------|-------------|--------|
| **Number Theory** | Financial time series with p-adic structure | Theoretical |
| **Thermodynamics** | Constrained hardware computation | Theoretical |
| **HPC/SIMD** | Emulation and testing pipelines | Theoretical |
| **Materials Science** | Hierarchical material properties | Theoretical |
| **Fluid Dynamics** | Aerodynamics/hydrodynamics modeling | Theoretical |

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train with validated config
python scripts/training/train_v5_12.py --config configs/v5_12_4_fixed_checkpoint.yaml --epochs 100

# Quick validation (5 epochs)
python scripts/training/train_v5_12.py --config configs/v5_12_4_fixed_checkpoint.yaml --epochs 5
```

### Using the Trained Model

```python
from src.models import TernaryVAEV5_11_PartialFreeze
from src.geometry import poincare_distance
import torch

# Load model
model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16, hidden_dim=64, max_radius=0.99,
    curvature=1.0, use_controller=True
)
ckpt = torch.load('sandbox-training/checkpoints/homeostatic_rich/best.pt')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Get embeddings
out = model(operations)
z_hyp = out['z_B_hyp']  # Use VAE-B for p-adic hierarchy

# Compute hyperbolic radii (NOT Euclidean norm)
origin = torch.zeros_like(z_hyp)
radii = poincare_distance(z_hyp, origin, c=1.0)
```

---

## Project Structure

```
ternary-vaes-bioinformatics/
├── src/                    # TIER 1: Core library
│   ├── core/              # P-adic math, ternary operations
│   ├── geometry/          # Hyperbolic geometry (Poincaré ball)
│   ├── models/            # VAE architectures
│   ├── training/          # Training infrastructure
│   └── losses/            # Loss functions
│
├── configs/               # Training configurations
├── sandbox-training/      # Checkpoints and training artifacts
│
├── deliverables/          # TIER 2: Bioinformatics applications
│   └── partners/          # Partner-specific packages
│
├── research/              # Research experiments
│   ├── codon-encoder/     # TrainableCodonEncoder
│   └── contact-prediction/# Contact prediction from embeddings
│
├── docs/                  # Documentation
│   ├── content/          # User guides
│   └── audits/           # Code audits
│
└── LEGAL_AND_IP/          # Licensing
```

---

## Theoretical Foundation

### The P-adic Hierarchy

For prime p=3, the 3-adic valuation v₃(n) counts the multiplicity of 3 in n:
- v₃(9) = 2 (9 = 3²)
- v₃(6) = 1 (6 = 2×3)
- v₃(5) = 0 (5 not divisible by 3)

This creates a natural tree structure where operations with higher valuation (divisible by more powers of 3) are "closer to the root."

### The Hyperbolic Realization

The Poincaré ball provides:
- **Exponential volume growth**: Room for exponentially many leaves
- **Geodesic distances**: Proper metric for tree structures
- **Differentiability**: Enables gradient-based optimization

### The Ultrametric Property

In p-adic space, all triangles are isosceles:
```
d(x, z) ≤ max(d(x, y), d(y, z))
```

This creates perfect hierarchical clustering—clusters within clusters—matching biological taxonomy and phylogenetic trees.

---

## Dual Manifold Organization

The framework supports two valid manifold types:

| Type | Hierarchy | Optimizes For | Best Applications |
|------|-----------|---------------|-------------------|
| **Valuation-optimal** | Negative (-0.8 to -1.0) | P-adic semantic structure | Genetic code, DDG prediction |
| **Frequency-optimal** | Positive (+0.6 to +0.8) | Shannon information efficiency | Compression, fast retrieval |

Both are mathematically valid—choose based on your application requirements.

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| PyTorch | 2.0 | 2.1+ |
| CUDA | Optional | 11.8+ |
| RAM | 8GB | 16GB |
| VRAM | 4GB | 6GB+ |

**Core dependencies**: torch, numpy, scipy, geoopt, scikit-learn

```bash
pip install -e ".[all]"  # Full installation with all extras
```

---

## Documentation

| Topic | Location |
|-------|----------|
| Getting Started | `docs/content/getting-started/` |
| Architecture | `docs/content/architecture/` |
| Theory | `docs/content/theory/` |
| API Reference | `docs/source/api/` |
| Partner Packages | `deliverables/partners/DELIVERABLES_INDEX.md` |

---

## License

### Software (Code)
**PolyForm Non-Commercial 1.0.0**
- Permitted: Academic, educational, non-profit use
- Commercial use requires separate license

### Research Outputs (Data, Figures, Models)
**CC-BY-4.0**
- Free for any reuse with attribution

All legal documents: [`LEGAL_AND_IP/`](LEGAL_AND_IP/)

---

## Citation

```bibtex
@software{ternary_vae,
  author = {{AI Whisperers}},
  title = {Ternary VAE: P-adic Hyperbolic Variational Autoencoders},
  year = {2026},
  url = {https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics}
}
```

---

## Contributing

1. Read [`LEGAL_AND_IP/CLA.md`](LEGAL_AND_IP/CLA.md)
2. Review [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
3. Follow [`CONTRIBUTING.md`](CONTRIBUTING.md)
4. Open a Pull Request

---

## Contact

- Issues: GitHub Issues
- Commercial licensing: support@aiwhisperers.com

---

*Version 5.12.5 · Updated 2026-01-23*
