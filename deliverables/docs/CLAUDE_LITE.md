# Ultrametric Antigen AI - Quick Reference

**Doc-Type:** AI Context (Lite) · Version 1.0 · 2026-02-03

---

## Project Overview

Variational Autoencoder framework learning hierarchical structure in hyperbolic space using p-adic number theory. Applies [3-adic-ml foundation](https://github.com/Ai-Whisperers/3-adic-ml) to bioinformatics.

**Current Version:** 5.12.5
**Status:** Production-ready

---

## Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Coverage | 100% | 100% |
| Hierarchy | -0.83 | -0.8321 (ceiling) |
| Richness | >0.005 | 0.00787 |

---

## Validated Applications

| Application | Metric | Value | Status |
|-------------|--------|-------|--------|
| DDG Prediction | LOO Spearman | 0.58 | Production |
| Contact Prediction | AUC-ROC | 0.67 | Research |
| AMP Fitness | Pearson r | 0.61 | Production |
| Force Constants | Correlation | 0.86 | Validated |

---

## Checkpoint Quick Reference

| Checkpoint | Use Case | Coverage | Hierarchy |
|------------|----------|:--------:|:---------:|
| `homeostatic_rich` | DDG, semantic reasoning | 100% | -0.8321 |
| `v5_12_4/best_Q.pt` | General purpose | 100% | -0.82 |
| `v5_11_structural` | Contact prediction | 100% | -0.74 |
| `v5_11_progressive` | Compression, retrieval | 100% | +0.78 |

---

## Partner Packages

| Package | Focus | Status | Key Metric |
|---------|-------|:------:|------------|
| `protein_stability_ddg` | DDG prediction | 95% | LOO rho=0.52 |
| `antimicrobial_peptides` | AMP optimization | 90% | 5/5 models significant |
| `arbovirus_surveillance` | Primer design | 90% | 7 viruses covered |
| `hiv_research_package` | Drug resistance | Complete | Stanford HIVdb |

**Location:** `deliverables/partners/`

---

## Quick Evaluation

```python
from src.models import TernaryVAEV5_11_PartialFreeze
from src.geometry import poincare_distance
import torch

model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16, hidden_dim=64, max_radius=0.99,
    curvature=1.0, use_controller=True
)
ckpt = torch.load('checkpoints/homeostatic_rich/best.pt')
model.load_state_dict(ckpt['model_state_dict'])

# Get embeddings - use hyperbolic distance for radii
out = model(ops)
origin = torch.zeros_like(out['z_B_hyp'])
radii = poincare_distance(out['z_B_hyp'], origin, c=1.0)
```

---

## Key Files

| Purpose | Location |
|---------|----------|
| Main model | `src/models/ternary_vae.py` |
| Geometry | `src/geometry/` |
| Training | `src/scripts/training/train_v5_12.py` |
| Configs | `src/configs/v5_12_4_fixed_checkpoint.yaml` |

---

## Documentation Links

| Document | Purpose |
|----------|---------|
| [CLAUDE_BIO.md](CLAUDE_BIO.md) | Bioinformatics applications |
| [CLAUDE_DEV.md](CLAUDE_DEV.md) | Developer reference |
| [Bioinformatics Guide](docs/BIOINFORMATICS_GUIDE.md) | No-math user guide |
| [Mathematical Foundations](docs/mathematical-foundations/) | Deep theory |

---

## Critical Patterns

```python
# CORRECT - Hyperbolic distance
from src.geometry import poincare_distance
radius = poincare_distance(z_hyp, origin, c=curvature)

# WRONG - Euclidean norm on hyperbolic embeddings
radius = torch.norm(z_hyp, dim=-1)  # DO NOT USE
```

---

## Training Command

```bash
python src/scripts/training/train_v5_12.py \
    --config src/configs/v5_12_4_fixed_checkpoint.yaml \
    --epochs 100
```

---

*For bioinformatics applications: [CLAUDE_BIO.md](CLAUDE_BIO.md)*
*For developer details: [CLAUDE_DEV.md](CLAUDE_DEV.md)*
