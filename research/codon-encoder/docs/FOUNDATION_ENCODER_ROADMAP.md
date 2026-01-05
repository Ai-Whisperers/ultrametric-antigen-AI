# Foundation Encoder Research Roadmap

**Doc-Type:** Research Plan · Version 1.0 · Updated 2026-01-05 · AI Whisperers

---

## Executive Summary

**Vision:** Unified encoder with multi-task heads for protein stability (DDG), antimicrobial peptide fitness (AMP), viral clade classification, and drug resistance prediction.

**Approach:** Data-first, sequential training with falsification before production.

**Hardware Constraints:** 3-4GB VRAM, 8-10GB RAM (all training must fit within these limits)

---

## Partner Package Assessment Summary

### jose_colbes (DDG Prediction) - 7/10 Ready

| Category | Status | Details |
|----------|--------|---------|
| **READY NOW** | DDG prediction | Spearman ρ=0.60 (LOO CV, n=52), beats ESM-1v |
| **READY NOW** | AA embeddings | 16D hyperbolic Poincaré ball, 20 AAs |
| **READY NOW** | S669 benchmark | Validated, p<0.001, reproducible |
| **READY NOW** | Speed advantage | <0.1s per mutation |
| **NEEDS WORK** | Large-scale validation | Need N>2,000 (S2648, ProThermDB) |
| **NEEDS WORK** | Rosetta-blind | Only synthetic demo data |
| **NEEDS WORK** | Structural context | Missing RSA, secondary structure |

**Foundation Encoder Use:** DDG as secondary signal in ensemble, embeddings for feature extraction

---

### carlos_brizuela (AMP Optimization) - 88% Ready

| Category | Status | Details |
|----------|--------|---------|
| **READY NOW** | NSGA-II framework | 100 Pareto candidates proven working |
| **READY NOW** | Activity prediction | 5 models validated (Pearson r=0.56, p<0.05) |
| **READY NOW** | MIC data | 272 samples, 178 unique peptides |
| **READY NOW** | Validation suite | 7 scripts, falsification tests implemented |
| **BLOCKING** | PeptideVAE checkpoint | Training script ready, NOT YET RUN (~1 GPU hr) |

**Foundation Encoder Use:** READY after 1hr training run. Best partner for immediate integration.

**Memory Estimate:** PeptideVAE training: ~2GB VRAM, ~4GB RAM (within constraints)

---

### alejandra_rojas (Primer Design) - 6.5/10 Ready

| Category | Status | Details |
|----------|--------|---------|
| **READY NOW** | Phylogenetic structure | 270 DENV-4 genomes, 5 clades validated |
| **READY NOW** | Entropy measurements | 36 windows, NS5 regions mapped |
| **READY NOW** | P-adic integration | Codon encoder correlation data |
| **READY NOW** | Hypothesis falsification | Root cause = ancient cryptic diversity |
| **NEEDS WORK** | Working primer pairs | 0 validated (library is scaffolding only) |
| **NEEDS WORK** | In silico PCR | Not implemented |
| **NEEDS WORK** | Cross-reactivity | No 7×7 matrix |

**Foundation Encoder Use:** Excellent for clade classification + conservation features. NOT ready for primer prediction.

---

### hiv_research_package (Drug Resistance) - 75% Architecture / 30% Validated

| Category | Status | Details |
|----------|--------|---------|
| **READY NOW** | WHO SDRM database | 50+ mutations mapped to drugs |
| **READY NOW** | Drug classification | NRTI/NNRTI/INSTI/PI properly categorized |
| **READY NOW** | Result schemas | TDRResult, LASelectionResult dataclasses |
| **READY NOW** | Treatment logic | WHO first-line regimens implemented |
| **NEEDS WORK** | Real patient sequences | 0 collected (demo only) |
| **NEEDS WORK** | Stanford validation | API integration exists, NOT TESTED |
| **NEEDS WORK** | Clinical outcomes | No LA success probability validation |

**Foundation Encoder Use:** Reference data ready. Resistance prediction needs external validation.

---

## Data Inventory

### Data We HAVE (Validated)

| Dataset | Source | N | Validation | Confidence | FE Ready? |
|---------|--------|---|------------|------------|:---------:|
| S669 DDG | jose_colbes | 52 (LOO) | ρ=0.60, p<0.001 | 95% | YES |
| AA Embeddings | jose_colbes | 20 AAs | 16D hyperbolic | 90% | YES |
| AMP Activity | carlos_brizuela | 272 pairs | 5 models, r=0.56 | 90% | YES |
| NSGA-II Framework | carlos_brizuela | 100 Pareto | Working demo | 95% | YES |
| DENV-4 Phylogeny | alejandra_rojas | 270 genomes | 5 clades | 90% | YES (features) |
| Entropy Maps | alejandra_rojas | 36 windows | NS5 validated | 85% | YES |
| WHO SDRM | hiv_research | 50+ mutations | Drug mappings | 95% | YES (reference) |
| V5 Zone Classification | V5 validation | 190 pairs | CV 66.8% | 95% | YES |
| Position Thresholds | V5 validation | buried/surface | p<0.0001 | 85% | YES |

### Data We NEED (Collect First) - Blocking Items

| Dataset | Source | Current | Target | Blocking? | Effort |
|---------|--------|---------|--------|:---------:|--------|
| PeptideVAE Checkpoint | carlos_brizuela | 0 | 1 model | **YES** | 1hr GPU |
| Large-scale DDG | jose_colbes | 52 | 2,000+ | Partial | 2-4 weeks |
| Rosetta-blind (real) | jose_colbes | 0 (synthetic) | 50+ PDBs | YES | 4-6 weeks |
| Stanford Validation | hiv_research | 0 | 50+ sequences | YES | 1 week |
| Real HIV Sequences | hiv_research | 0 | 500+ | YES | 4-8 weeks |
| Validated Primers | alejandra_rojas | 0 pairs | 5+ per virus | YES | 2-4 weeks |
| In silico PCR | alejandra_rojas | 0 | Genome-scale | YES | 3-5 days |
| AlphaFold RSA | EBI API | 0 | All S669 | Partial | 1 week |

### Data to FALSIFY (Before Production)

| Hypothesis | Test | Falsification Criterion | Status |
|------------|------|------------------------|--------|
| Hybrid > Simple generalizes | S2648 test | If ρ < 0.5 on held-out | PENDING |
| Position thresholds stable | AlphaFold RSA | If shift >1.0 | PENDING |
| PeptideVAE beats sklearn | Fold validation | If r < 0.56 | PENDING |
| Stanford concordance | API comparison | If <95% agreement | PENDING |
| DENV-4 primers viable | In silico PCR | If <80% coverage | PENDING |

---

## Partner Integration Phases

### Immediate: carlos_brizuela (1hr blocker)

**Action:** Run PeptideVAE training
```bash
cd deliverables/partners/carlos_brizuela
python training/train_peptide_encoder.py --epochs 50
```

**Memory:** ~2GB VRAM, ~4GB RAM (within 3-4GB VRAM / 8-10GB RAM constraints)

**Output:** PeptideVAE checkpoint

**Validation:** Must beat sklearn r=0.56

**Impact:** Unlocks 88% → 100% ready

---

### Phase 1: jose_colbes (P1 - Expand Scale)

- Current: 52 mutations LOO validated
- Target: 500+ with S2648/ProThermDB
- Also: Rosetta-blind on 50+ real PDBs
- Success: ρ≥0.65 on expanded set

---

### Phase 2: hiv_research_package (P2 - External Validation)

- Action: Obtain Stanford API credentials
- Validate: 50+ sequences against HIVdb
- Target: >95% concordance
- Then: Collect 500+ real patient sequences

---

### Phase 3: alejandra_rojas (P3 - Primer Pipeline)

- Implement: In silico PCR simulation
- Test: CDC primer recovery benchmark
- Generate: Cross-reactivity 7×7 matrix
- Defer: Wet-lab validation (future version)

---

## Model Architecture Plan

### Backbone

- TrainableCodonEncoder (12→64→64→16) - existing, validated
- Frozen after pre-training on all READY data
- Memory: ~50MB model, fits easily in constraints

### Task Heads (Prioritized by Readiness)

| Head | Input | Output | Training Data | Ready? | Memory |
|------|-------|--------|---------------|:------:|--------|
| DDG | embedding + position | kcal/mol | jose_colbes S669 | YES | ~10MB |
| AMP Activity | embedding | MIC score | carlos_brizuela 272 pairs | YES* | ~10MB |
| Clade | embedding | 5-class | alejandra_rojas 270 genomes | YES | ~5MB |
| Resistance | embedding + drug | class | hiv_research WHO SDRM | PARTIAL | ~10MB |

*Requires PeptideVAE checkpoint (1hr training)

**Total Memory:** All heads combined ~100MB, well within 3-4GB VRAM constraint

### Regime Router

- Input: (wt, mut, position_context)
- Output: which head/predictor to trust
- Trained on: V5 zone classifications (190 pairs)
- Ready: YES (data exists from V5 validation)

---

## Training Sequence

```
IMMEDIATE (This Session):
├── Train PeptideVAE checkpoint (~1hr, <2GB VRAM)
└── Validate beats sklearn r=0.56

PHASE 1 (Ready Data):
├── Train DDG head on S669 (jose_colbes)
├── Train Clade head on 270 genomes (alejandra_rojas)
└── Train Regime Router on V5 zones

PHASE 2 (After Blocking Items):
├── Expand DDG to S2648 (jose_colbes)
├── Train AMP head with real PeptideVAE (carlos_brizuela)
└── Validate Stanford concordance (hiv_research)

PHASE 3 (Future - More Data Needed):
├── Resistance head (needs real sequences)
├── Primer prediction (needs PCR validation)
└── Multi-task joint training
```

---

## Success Criteria

| Metric | Current | Minimum | Target | Stretch |
|--------|---------|---------|--------|---------|
| DDG Spearman | 0.60 | 0.60 | 0.70 | 0.80 |
| AMP Pearson | 0.56 (sklearn) | 0.56 | 0.65 | 0.75 |
| Clade accuracy | N/A | 90% | 95% | 99% |
| Resistance AUC | N/A | 0.70 | 0.80 | 0.90 |
| Regime accuracy | 66.8% | 70% | 80% | 90% |

---

## What to DEFER to Future Versions

| Component | Reason | Prerequisites |
|-----------|--------|---------------|
| Primer prediction head | 0 validated primers | In silico PCR, wet-lab |
| Clinical resistance | 0 real patient sequences | 500+ HIV sequences |
| LA selection probability | No outcome validation | Prospective cohort data |
| Multi-mutation epistasis | Single-mutation only | 500+ double mutants |
| Rosetta-blind validation | Synthetic demo only | 50+ real PDB structures |

---

## Hardware Constraints Summary

All training must fit within:
- **VRAM:** 3-4GB maximum
- **RAM:** 8-10GB maximum

### Memory Estimates by Component

| Component | VRAM | RAM | Fits? |
|-----------|------|-----|:-----:|
| PeptideVAE training | ~2GB | ~4GB | YES |
| DDG head training | ~1GB | ~2GB | YES |
| Clade head training | ~1GB | ~2GB | YES |
| Full Foundation Encoder | ~3GB | ~6GB | YES |
| Batch inference (1000 seq) | ~2GB | ~4GB | YES |

---

## Key Insights from Partner Exploration

1. **carlos_brizuela is lowest-hanging fruit** - 88% ready with only 1hr GPU training blocking

2. **jose_colbes and alejandra_rojas** provide validated data for immediate Foundation Encoder training (DDG + Clade heads)

3. **hiv_research_package** needs external validation (Stanford API) before any integration beyond reference data

4. **All training fits within hardware constraints** - No component exceeds 3-4GB VRAM or 8-10GB RAM

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-05 | 1.0 | Initial roadmap from partner package exploration |

---

**End of Document**
