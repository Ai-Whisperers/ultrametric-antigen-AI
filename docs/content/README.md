# Ternary VAE Documentation

> **Single source of truth for all project documentation.**

---

## Quick Navigation

| I want to... | Go to |
|:-------------|:------|
| Get started quickly | [Getting Started](getting-started/README.md) |
| Understand the architecture | [Architecture](architecture/README.md) |
| Learn the theory | [Theory](theory/README.md) |
| See research findings | [Research](research/README.md) |
| Contribute to development | [Development](development/README.md) |
| Find role-specific guides | [Stakeholders](stakeholders/README.md) |
| Review validation results | [Validation](validation/README.md) |

---

## Documentation Structure

```
docs/content/
├── getting-started/     # Installation, quick start, tutorials
│   ├── installation.md
│   ├── quick-start.md
│   └── tutorials/
├── architecture/        # System architecture (single source of truth)
│   ├── README.md        # Architecture overview
│   ├── base-vae.md      # BaseVAE abstraction
│   ├── uncertainty.md   # Uncertainty quantification
│   ├── transfer.md      # Transfer learning
│   ├── epistasis.md     # Epistasis modeling
│   └── structure.md     # Structure-aware modeling
├── theory/              # Mathematical and biological foundations
│   ├── hyperbolic.md    # Hyperbolic geometry
│   ├── p-adic.md        # P-adic number theory
│   └── biology.md       # Biological context
├── research/            # Research domains
│   ├── hiv/             # HIV analysis (200K+ sequences)
│   └── multi-disease/   # 11-disease platform
├── development/         # Contributing and roadmap
│   ├── contributing.md
│   ├── testing.md
│   └── roadmap.md
├── stakeholders/        # Role-specific guides
│   ├── scientists.md
│   ├── investors.md
│   ├── developers.md
│   └── clinicians.md
└── validation/          # Testing and benchmarks
    ├── benchmarks.md
    └── test-strategy.md
```

---

## API Reference

For auto-generated API documentation, see [docs/source/api/](../source/api/).

---

## Key Features

- **Dual VAE Architecture**: VAE-A explores, VAE-B refines
- **Hyperbolic Geometry**: Poincare ball latent space captures hierarchical structure
- **P-adic Structure**: 3-adic valuation for codon relationships
- **Multi-Disease Platform**: 11 diseases with unified framework
- **Uncertainty Quantification**: MC Dropout, Evidential, Ensemble methods
- **Transfer Learning**: Cross-disease knowledge transfer

---

## External Links

- [GitHub Repository](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics)
- [API Documentation](../source/api/index.rst)

---

_Last updated: 2025-12-28_
