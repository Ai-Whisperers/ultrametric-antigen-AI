# Ultrametric Antigen AI

[![Version](https://img.shields.io/badge/version-5.12.5-blue.svg)](docs/mathematical-foundations/README.md)
[![Foundation](https://img.shields.io/badge/foundation-3--adic--ml-green.svg)](https://github.com/Ai-Whisperers/3-adic-ml)
[![License: PolyForm Non-Commercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20NC-lightgrey.svg)](LEGAL_AND_IP/LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LEGAL_AND_IP/RESULTS_LICENSE.md)

---

## For Bioinformatics Specialists

Start with the **[Bioinformatics Guide](docs/BIOINFORMATICS_GUIDE.md)** - no mathematical background required.

**AI/Developer Context:** [CLAUDE_LITE.md](CLAUDE_LITE.md) | [CLAUDE_BIO.md](CLAUDE_BIO.md) | [CLAUDE_DEV.md](CLAUDE_DEV.md)

---

## Overview

**Ultrametric Antigen AI** is a variational autoencoder framework that learns hierarchical structure in hyperbolic space using p-adic number theory. This project applies the [3-adic-ml foundation](https://github.com/Ai-Whisperers/3-adic-ml) to bioinformatics problems.

> Open-source hyperbolic & 3-adic VAE for bioinformatics applications

**This framework complements classical AI, not competes with it.** We augment standard deep learning with geometry-aware priors that match hierarchical data structure. The VAE models in `src/` evolved through extensive research (v5.5→v5.11→v5.12) to validate these mathematical foundations—they are research artifacts that proved the concepts now formalized in [3-adic-ml](https://github.com/Ai-Whisperers/3-adic-ml).

### The Core Problem

We retain the efficiency of Euclidean deep learning while augmenting it with geometry-aware representations that unlock superior performance on hierarchical and relational data. Standard deep-learning infrastructure is optimized for Euclidean geometry, which is highly efficient but not naturally aligned with strongly hierarchical data. Our approach addresses this by aligning the model’s geometric assumptions with the intrinsic structure of the data, improving representation efficiency and downstream performance on tree-like and relational problems:

- **P-adic valuation** provides the algebraic hierarchy (3-adic numbers for ternary operations)
- **Hyperbolic geometry** (Poincaré ball) provides the continuous differentiable space
- **The isomorphism**: Low p-adic distance ↔ Close in hyperbolic space; High valuation ↔ Close to origin

**DISCLAIMER:** While current deep-learning infrastructure operates on Euclidean latent spaces, our architecture incorporates curvature-aware techniques using mature libraries such as geoopt to approximate hyperbolic structure where it provides measurable advantage. Fully native non-Euclidean embedding frameworks are not yet an industry standard; however, our approach delivers many of the practical benefits of hyperbolic representations today. This has already produced strong empirical results with clear commercial applicability and near-term research value, while remaining compatible with existing production ecosystems.



---

## Framework

The project is organized into two complementary project tiers:

### TIER 1: Applications

Domain-specific applications built on TIER 2 foundations.

#### Bioinformatics

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

---

### TIER 2: Models and Mathematical Foundations

Core AI/ML training infrastructure and mathematical primitives. Highly validated and generalizable. Review the improved versions with more details on https://github.com/Ai-Whisperers/3-adic-ml

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

### Validated Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Coverage** | 100% | 100% | Achieved |
| **Hierarchy** | -0.8321 | -0.83 | Mathematical ceiling |
| **Richness** | 0.00787 | >0.005 | 5.8x baseline |

The model embeds 19,683 ternary operations ({-1, 0, +1}^9 = 3^9) into a 16-dimensional hyperbolic space—a 1,230x compression while preserving the underlying 3-adic valuation hierarchy.

## Quick Start


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

## Project Structure


---

<details>
<summary><strong>Theoretical Foundations</strong> (click to expand)</summary>

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

**Deep dive:** [Mathematical Foundations](docs/mathematical-foundations/)

</details>

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

## Related Work

### P-adic and Ultrametric Neural Networks

- **v-PuNNs: van der Put Neural Networks for Transparent Ultrametric Representation Learning**
  N'guessan, G. L. R. (2025). [arXiv:2508.01010](https://arxiv.org/abs/2508.01010)
  *Ultrametric neural network architectures using van der Put basis for hierarchical learning.*

- **Geometry of Information Integration**
  Amari, S. (2017). [arXiv:1709.02050](https://arxiv.org/abs/1709.02050)
  *Information geometry foundations for neural information integration.*

### Bioinformatics Applications

- **A Simple yet Effective DDG Predictor is An Unsupervised Antibody Optimizer and Explainer**
  (2025). [arXiv:2502.06913v1](https://arxiv.org/abs/2502.06913v1)
  *DDG prediction methods for antibody optimization, relevant to our protein stability work.*

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
@software{ultrametric_antigen_ai,
  author = {{AI Whisperers}},
  title = {Ultrametric Antigen AI: P-adic Hyperbolic Variational Autoencoders for Bioinformatics},
  year = {2026},
  url = {https://github.com/Ai-Whisperers/ultrametric-antigen-AI}
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
- Commercial licensing: ai.whisperer.wvdp@gmail.com

---

*Version 5.12.5 · Updated 2026-02-01*
