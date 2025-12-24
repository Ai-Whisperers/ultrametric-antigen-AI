# AlphaFold 2 OSS Integration Plan

**Doc-Type:** Implementation Plan · Version 1.0 · Updated 2025-12-18

---

## Executive Summary

This document outlines the strategy for integrating AlphaFold 2 open-source into the p-adic genomics research pipeline. The goal is to enable local structure prediction for modified sequences (citrullinated peptides, PTM variants) that do not exist in pre-computed databases, and to cross-validate our p-adic geometric predictions against structural perturbations.

---

## Motivation

### Current Limitations

| Approach | Limitation |
|----------|------------|
| AlphaFold EBI Database | Only pre-computed structures for canonical UniProt sequences |
| AlphaFold 3 Server | Rate-limited, closed weights, no batch automation |
| PDB Experimental | Sparse coverage of PTM variants |

### What AF2 OSS Enables

- **Custom sequence prediction** - Citrullinated, phosphorylated, or engineered sequences
- **Batch automation** - Hundreds of predictions without rate limits
- **Weight access** - Potential for fine-tuning and embedding extraction
- **Reproducibility** - Full control over inference parameters

---

## Infrastructure Requirements

### Storage Tiers

| Tier | Database | Size | Use Case |
|------|----------|------|----------|
| Minimal | ColabFold/MMseqs2 | ~50GB | Quick experimentation, proof of concept |
| Reduced | Reduced BFD + UniRef30 | ~200GB | Local development, moderate accuracy |
| Full | Complete AF2 databases | ~2.5TB | Production, maximum accuracy |

### Current Status

```
Available storage: Limited (< 500GB free)
Target tier: Minimal → Reduced (current sprint)
Future tier: Full (when lab hardware available)
```

### Compute Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA T4 (16GB) | A100 (40GB) or better |
| RAM | 32GB | 64GB+ |
| CPU | 8 cores | 16+ cores |
| Storage | SSD recommended | NVMe for database |

---

## Phased Implementation

### Phase 1: Quick Experimentation (Current)

**Timeline:** Immediate
**Infrastructure:** Local GPU or Google Colab

**Approach:**
1. Use ColabFold with MMseqs2 server for MSA generation
2. No local database required
3. Suitable for 10-50 predictions

**Tasks:**
- [ ] Set up ColabFold notebook for citrullination predictions
- [ ] Predict native vs. citrullinated structures for key epitopes
- [ ] Extract pLDDT and coordinate data for p-adic correlation

**Target Sequences:**
```
Priority 1 (Sentinel epitopes):
- FGA_R38: GPRVVERHQS (native) → GPXVVERHQS (citrullinated)
- FLG_R30: SHQESTRGRS (native) → SHQESTXGRS (citrullinated)

Priority 2 (Goldilocks validation):
- VIM_R71: RLRSSVPGVR
- ENO1_CEP1: KIREEIFDSRGNP
- FGB_R406: SARGHRPLDKK
```

**Deliverable:** Structural perturbation vs. p-adic entropy change correlation

---

### Phase 2: Local Reduced Database (Near-term)

**Timeline:** When 500GB+ storage available
**Infrastructure:** Local workstation with GPU

**Approach:**
1. Install AlphaFold 2 via Docker
2. Download reduced databases (~200GB)
3. Enable batch predictions without external API dependency

**Tasks:**
- [ ] Provision 500GB+ storage
- [ ] Install Docker + NVIDIA Container Toolkit
- [ ] Clone AlphaFold repository
- [ ] Download reduced databases:
  - [ ] UniRef30 (~50GB)
  - [ ] Reduced BFD (~60GB)
  - [ ] PDB70 (~20GB)
  - [ ] PDB mmCIF (~50GB)
- [ ] Validate installation with test prediction
- [ ] Create batch prediction pipeline for epitope library

**Database Download Commands:**
```bash
# From alphafold repository
scripts/download_alphafold_params.sh /data/alphafold/params
scripts/download_uniref30.sh /data/alphafold/databases
scripts/download_pdb70.sh /data/alphafold/databases
scripts/download_pdb_mmcif.sh /data/alphafold/databases
```

**Deliverable:** Local prediction pipeline for all RA autoantigens

---

### Phase 3: Full Production Database (Future)

**Timeline:** When lab hardware available
**Infrastructure:** Dedicated server or HPC cluster

**Approach:**
1. Full 2.5TB database installation
2. Maximum MSA depth for highest accuracy
3. Integration with research compute cluster

**Tasks:**
- [ ] Provision 3TB+ storage (with headroom)
- [ ] Download full databases:
  - [ ] Full BFD (~1.7TB)
  - [ ] MGnify (~120GB)
  - [ ] UniRef90 (~100GB)
  - [ ] UniClust30 (~200GB)
- [ ] Configure for cluster job submission (SLURM/PBS)
- [ ] Benchmark prediction quality vs. reduced database

**Full Database Download:**
```bash
# Complete installation
scripts/download_all_data.sh /data/alphafold
```

**Deliverable:** Production-grade structure prediction for entire proteome-wide citrullination analysis

---

## Integration with P-Adic Pipeline

### Cross-Validation Strategy

```
P-Adic Pipeline                    AlphaFold 2 Pipeline
      │                                    │
      ▼                                    ▼
Codon Encoder                        MSA + Templates
      │                                    │
      ▼                                    ▼
16-dim embedding                     Structure prediction
      │                                    │
      ▼                                    ▼
Entropy change (ΔH)                  Structural metrics
      │                                    │
      └──────────────┬─────────────────────┘
                     ▼
              Correlation Analysis
              - ΔH vs RMSD
              - ΔH vs ΔpLDDT
              - ΔH vs accessibility change
```

### Structural Metrics to Extract

| Metric | Description | Hypothesis |
|--------|-------------|------------|
| RMSD | Backbone deviation native→modified | Correlates with p-adic shift |
| ΔpLDDT | Confidence change at modification site | Low pLDDT = disordered = accessible |
| SASA | Solvent accessible surface area | Exposed sites more immunogenic |
| Secondary structure | Helix/sheet/coil classification | Coil regions more tolerant to PTM |

### Expected Outcomes

1. **Concordance:** Structural perturbation correlates with p-adic entropy change → orthogonal validation
2. **Divergence:** Metrics disagree → structure and sequence geometry capture different aspects
3. **Synergy:** Combined model outperforms either alone → publishable methodology

---

## Fine-Tuning Opportunities (Phase 3+)

### Tier 1: Feature Extraction Only

Use AF2 as frozen feature extractor:
- Extract Evoformer representations
- Feed into downstream p-adic models
- No modification of AF2 weights

### Tier 2: Transfer Learning

Fine-tune final layers on PTM dataset:
- Curate dataset of known PTM structures
- Use p-adic shift as soft supervision
- Requires OpenFold (PyTorch) for cleaner gradient flow

### Tier 3: Hybrid Architecture

Inject p-adic embeddings into AF2:
- Add 16-dim codon embeddings to MSA representation
- Joint training on structure + p-adic objectives
- Research-grade contribution

**Recommended codebase for fine-tuning:** OpenFold (`github.com/aqlaboratory/openfold`)
- PyTorch implementation (vs JAX original)
- Cleaner model surgery
- Active community

---

## Alternative: ColabFold for Rapid Prototyping

### Advantages

- No local database required
- Free GPU via Google Colab
- MMseqs2 server handles MSA generation
- Suitable for < 100 predictions

### Setup

```python
# ColabFold installation (in Colab notebook)
!pip install colabfold[alphafold]
from colabfold.batch import get_queries, run

# Predict structure
run(
    queries=[("epitope_native", "GPRVVERHQS")],
    result_dir="predictions/",
    use_templates=True,
    num_models=5
)
```

### Limitations

- Rate limited by MMseqs2 server
- No citrulline (non-standard AA) in standard pipeline
- Requires workaround for PTM residues

### Citrulline Workaround

```
Option A: Model as glutamine (Q) - similar size, neutral charge
Option B: Model as arginine (R) then analyze local perturbation
Option C: Use AF3 server for true citrulline (CIR code)
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Storage unavailable | Medium | High | Start with ColabFold, defer local install |
| GPU memory insufficient | Low | Medium | Use model_preset=reduced, batch size=1 |
| Citrulline modeling inaccurate | Medium | Medium | Validate against known structures, use AF3 for key predictions |
| Correlation with p-adic weak | Unknown | Low | Negative result still publishable as methodology comparison |

---

## Success Criteria

### Phase 1 (Quick Experimentation)
- [ ] Successfully predict 5+ native/modified structure pairs
- [ ] Extract pLDDT and RMSD metrics
- [ ] Compute correlation with p-adic entropy change
- [ ] Document findings in results/

### Phase 2 (Local Reduced)
- [ ] Local AF2 installation functional
- [ ] Batch pipeline for 50+ predictions
- [ ] Automated metric extraction integrated with existing scripts

### Phase 3 (Full Production)
- [ ] Full database installed and validated
- [ ] Proteome-wide citrullination structural analysis
- [ ] Fine-tuning experiments initiated

---

## References

| Resource | URL |
|----------|-----|
| AlphaFold 2 Repository | github.com/google-deepmind/alphafold |
| OpenFold (PyTorch) | github.com/aqlaboratory/openfold |
| ColabFold | github.com/sokrypton/ColabFold |
| AlphaFold EBI Database | alphafold.ebi.ac.uk |
| AF2 Paper | doi.org/10.1038/s41586-021-03819-2 |

---

## Changelog

| Date | Version | Description |
|------|---------|-------------|
| 2025-12-18 | 1.0 | Initial planning document |

---

**Status:** Planning complete, ready for Phase 1 implementation
