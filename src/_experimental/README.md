# Experimental Modules

> **These modules are experimental and not yet production-ready.**

This directory contains research and experimental modules that are being developed but are not part of the core library.

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| `categorical/` | Category theory abstractions | Research |
| `category/` | Alternative category implementation | Research |
| `contrastive/` | Contrastive learning methods | Research |
| `diffusion/` | Diffusion models for sequences | Research |
| `equivariant/` | SE(3)-equivariant networks | Experimental |
| `graphs/` | Graph neural networks | Research |
| `implementations/` | Alternative implementations | Archive |
| `information/` | Information-theoretic methods | Research |
| `linguistics/` | Linguistic analysis of codons | Research |
| `meta/` | Meta-learning approaches | Research |
| `physics/` | Physics-inspired methods | Research |
| `quantum/` | Quantum computing integration | Research |
| `topology/` | Topological data analysis | Research |
| `tropical/` | Tropical geometry methods | Research |
| `core/` | Experimental core utilities | Research |

## Usage

These modules are not imported by default. To use them:

```python
from src._experimental.quantum import QuantumCodonEncoder
from src._experimental.tropical import TropicalVAE
```

## Contributing

If you want to promote a module to production:

1. Add comprehensive tests
2. Document the API
3. Validate performance
4. Submit a PR to move it to `src/`

---

_Last updated: 2025-12-28_
