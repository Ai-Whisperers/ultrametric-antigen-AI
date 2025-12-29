# For Scientists & Researchers

> **Technical deep dive for academic and research collaboration.**

---

## Quick Start

1. [Theory Foundations](../theory/README.md) - Mathematical background
2. [Architecture](../architecture/README.md) - System design
3. [Validation](../validation/README.md) - Benchmarks and proofs

---

## Key Innovations

### 1. Hyperbolic Latent Space

We use **Poincare ball geometry** for the VAE latent space because:
- Evolution produces tree-like structures
- Hyperbolic space naturally embeds trees
- Distance encodes hierarchical relationships

**Read more**: [Hyperbolic Geometry](../theory/hyperbolic.md)

### 2. P-adic Codon Encoding

We use **3-adic valuation** for codon relationships:
- Captures hierarchical structure of genetic code
- Position-dependent mutation effects
- Validates with r=0.8339 correlation to Hamming

**Read more**: [P-adic Theory](../theory/p-adic.md)

### 3. Uncertainty Quantification

Clinical-grade predictions require uncertainty:
- MC Dropout, Evidential, Ensemble methods
- Epistemic/aleatoric decomposition
- Temperature calibration

**Read more**: [Uncertainty](../architecture/uncertainty.md)

---

## Validation Results

| Benchmark | Result | Method |
|:----------|:-------|:-------|
| HIV drug resistance | r=0.890 (23 drugs) | Spearman correlation |
| Known mutations | 65-70% F1 | Attention analysis |
| P-adic vs Hamming | r=0.8339 | Geometry validation |
| Cross-resistance | Captured | TAMs, M184V patterns |

---

## Research Domains

| Domain | Status | Documentation |
|:-------|:-------|:--------------|
| HIV | Active | [HIV Research](../research/hiv/README.md) |
| SARS-CoV-2 | Active | See multi-disease |
| Tuberculosis | Active | See multi-disease |
| Rheumatoid Arthritis | Exploratory | Legacy DOCUMENTATION |
| Neurodegeneration | Exploratory | Legacy DOCUMENTATION |

---

## Collaboration

### Using Our Methods

All research outputs (data, figures, models) are **CC-BY-4.0**:
- Free for any reuse with attribution
- Pre-trained models available
- Training data from public sources

### Code Access

Source code is **PolyForm Non-Commercial 1.0.0**:
- Academic/research use permitted
- Commercial use requires license
- Contact: support@aiwhisperers.com

### Citing

```bibtex
@software{ternary_vae,
  author = {{AI Whisperers}},
  title = {Ternary VAE Bioinformatics},
  year = {2025},
  url = {https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics}
}
```

---

## Replicating Results

```bash
# Clone repository
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics

# Install dependencies
pip install -r requirements.txt

# Run HIV validation
python scripts/experiments/run_full_validation.py

# Generate figures
python scripts/visualization/generate_figures.py
```

---

## Contact

- GitHub Issues: Bug reports, questions
- Email: support@aiwhisperers.com (collaborations)

---

_Last updated: 2025-12-28_
