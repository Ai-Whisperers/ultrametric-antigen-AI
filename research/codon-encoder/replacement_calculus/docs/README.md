# Replacement Calculus Documentation

**Doc-Type:** Documentation Index · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Documents

| Document | Description |
|----------|-------------|
| [01_FRAMEWORK_OVERVIEW.md](01_FRAMEWORK_OVERVIEW.md) | Mathematical foundation and module architecture |
| [02_VALIDATION_RESULTS.md](02_VALIDATION_RESULTS.md) | Completed validation experiments and findings |
| [03_PENDING_VALIDATIONS.md](03_PENDING_VALIDATIONS.md) | Roadmap for remaining experiments |

---

## Quick Links

**Implementation**: `../` (parent directory)
- `invariants.py` - Minimal invariant basis
- `groups.py` - LocalMinimum as algebraic groups
- `morphisms.py` - Structure-preserving transformations
- `groupoids.py` - Groupoid construction and escape paths

**Tests**: `../tests/test_replacement_calculus.py`

**Integration**: `../integration/`
- `vae_groupoid_builder.py` - P-adic based groupoid
- `embedding_groupoid.py` - Embedding distance based groupoid

---

## Key Findings Summary

1. **P-adic structure ≠ biological function** (21.4% accuracy)
2. **Embeddings capture partial structure** (97.8% recall, 26.3% precision)
3. **Framework validated** - machinery works, needs hybrid validity criteria

---

## Related Documents

- `../proposals/REPLACEMENT_CALCULUS.md` - Original proposal
- `../../falsification/COMBINED_FALSIFICATION_RESULTS.md` - Falsification studies
