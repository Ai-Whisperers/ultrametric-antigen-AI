# Future/Aspirational Modules

This directory contains placeholder modules for **planned future functionality**. These modules are **not implemented** and should not be imported in production code.

## Status

| Module | Intended Purpose | Priority |
|--------|------------------|----------|
| `topology/` | Persistent homology and TDA for structure analysis | Medium |
| `categorical/` | Category theory abstractions for compositional networks | Low |
| `tropical/` | Tropical semiring operations for optimization | Low |
| `equivariant/` | SE(3) equivariant networks for structure | High |
| `information/` | Information-theoretic measures (Fisher, etc.) | Medium |
| `graphs/` | Protein graph neural networks | High |
| `meta/` | Meta-learning (MAML, task sampling) | Low |
| `contrastive/` | Contrastive learning losses | Medium |
| `diffusion/` | Diffusion models for generation | Medium |
| `physics/` | Statistical physics (spin glass models) | Low |

## Implementation Notes

Each module contains an `__init__.py` with:
- Docstring describing intended functionality
- `__all__` exports listing planned classes/functions
- Some have lazy import patterns (will fail until implemented)

## Contributing

To implement a module:
1. Create the implementation files (e.g., `topology/persistent_homology.py`)
2. Update the `__init__.py` to import from actual implementations
3. Add tests in `tests/unit/<module_name>/`
4. Move the module from `_future/` to `src/`
5. Update `src/README.md` module tier classification

## Priority Rationale

**High Priority**:
- `equivariant/`: SE(3) equivariance critical for protein structure
- `graphs/`: GNNs for protein-protein interaction modeling

**Medium Priority**:
- `topology/`: TDA provides complementary structure features
- `information/`: Fisher information for natural gradients
- `contrastive/`: Self-supervised pretraining
- `diffusion/`: Generative modeling for sequences

**Low Priority**:
- `categorical/`: Academic interest, not critical path
- `tropical/`: Specialized optimization
- `meta/`: Requires extensive dataset collection
- `physics/`: Interesting theory, unclear practical benefit
