# P2: Biological Data Interoperability (ScanPy)

**Status:** Open
**Source:** IMPROVEMENT_PLAN.md (Sec 2)
**Area:** Bioinformatics

## Objective

Make the tool a "Plugin" for standard biotech pipelines like ScanPy.

## Tasks

- [ ] **Implement `AnnData` Support**: Create `src/data/bio_loader.py` using `scanpy`.
  - Feature: Save embeddings to `adata.obsm['X_ternary_vae']`.
- [ ] **Evolutionary Velocity**: Prototype fitness gradient metric based on `scvelo`.
