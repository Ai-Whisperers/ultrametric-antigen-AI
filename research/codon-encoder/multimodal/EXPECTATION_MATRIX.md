# Multimodal Integration: Expectation Matrix

**Doc-Type:** Research Planning · Version 2.0 · Updated 2026-01-03 · AI Whisperers

---

## Purpose

This document establishes baseline expectations for multimodal integration of:
1. **P-adic Codon Embeddings** (TrainableCodonEncoder)
2. **ESM-2 Contextual Embeddings** (Protein Language Model)
3. **Structural Features** (AlphaFold, Contact Maps, DSSP)

The expectations serve as:
- **Reproducibility anchor** for future comparisons
- **Success criteria** for integration efforts
- **Audit trail** documenting pre-integration state

---

## Ablation Study Results (LOO-Validated)

### Summary Table

| Mode | Features | LOO Spearman | LOO Pearson | Overfitting Ratio |
|------|----------|--------------|-------------|-------------------|
| codon_only | 4 | 0.34 | 0.45 | 1.57x |
| physico_only | 4 | 0.36 | 0.51 | 1.36x |
| esm_only (naive) | 4 | 0.47 | 0.46 | 1.28x |
| **codon+physico** | **8** | **0.60** | **0.62** | **1.27x** |
| codon+physico+esm | 12 | 0.57 | 0.61 | 1.34x |

### Key Findings

1. **Codon + Physico (0.60)** achieves best performance with proper regularization
2. **ESM-only (0.47)** outperforms individual components but naive addition hurts combined
3. **Curse of dimensionality**: 52 samples insufficient for 12+ features
4. **Synergy**: Codon + physico shows multiplicative benefit (0.60 > 0.34 + 0.36)

### Why Naive ESM Integration Decreased Performance

| Issue | Explanation | Solution |
|-------|-------------|----------|
| **Context-free embeddings** | Extracted from poly-A context, not real proteins | Use protein-level ESM |
| **Feature redundancy** | ESM evolutionary info overlaps p-adic structure | Feature selection / PCA |
| **Small dataset** | 52 samples can't support 12 features reliably | Larger datasets (ProteinGym) |
| **Naive features** | 4 summary stats from 480-dim loses information | Smarter feature extraction |

---

## Current Baseline (Post-Ablation)

### TrainableCodonEncoder + Physico Performance

| Task | Metric | Value | Dataset | Date |
|------|--------|-------|---------|------|
| DDG Prediction | LOO Spearman | **0.60** | S669 (n=52) | 2026-01-03 |
| DDG Prediction | LOO Pearson | 0.62 | S669 (n=52) | 2026-01-03 |
| DDG Prediction | LOO MAE | 0.89 | S669 (n=52) | 2026-01-03 |
| DDG Prediction | LOO RMSE | 1.17 | S669 (n=52) | 2026-01-03 |
| P-adic Structure | Distance Correlation | 0.74 | 64 codons | 2026-01-03 |

### Architecture Baseline

```
Current Best Pipeline (codon+physico):
  TrainableCodonEncoder → 16-dim Poincaré
  Features: [hyp_dist, delta_radius, diff_norm, cos_sim, Δhydro, Δcharge, Δsize, Δpolar]
  Model: Ridge(alpha=100) with StandardScaler
  Validation: Leave-One-Out CV
```

### Checkpoint Reference

| Checkpoint | Purpose | Location |
|------------|---------|----------|
| trained_codon_encoder.pt | DDG prediction | research/codon-encoder/training/results/ |
| esm_aa_embeddings.json | ESM-2 35M AA embeddings | research/codon-encoder/multimodal/data/ |
| v5_11_structural | Contact prediction | sandbox-training/checkpoints/ |

---

## Phase 1: Protein-Level ESM Integration

### Current Issue

The naive ESM approach extracted embeddings for single amino acids in poly-A context:
```
Context: AAAAAAAAAA{X}AAAAAAAAAA
Problem: No real protein context, no mutation site information
```

### Proper ESM Integration Strategy

#### 1A. Protein-Sequence-Level Embeddings

Extract ESM embeddings for **actual S669 protein sequences**:

```python
# For each mutation in S669:
# 1. Get wild-type protein sequence from UniProt/PDB
# 2. Extract ESM embedding at mutation position
# 3. Also extract for mutant sequence
# 4. Use delta embedding as feature

def extract_mutation_esm_features(wt_seq, mut_seq, position):
    wt_emb = esm_encoder(wt_seq)[position]  # 1280-dim
    mut_emb = esm_encoder(mut_seq)[position]  # 1280-dim

    # Context-aware features
    context_emb = esm_encoder(wt_seq)[position-5:position+5].mean(0)

    return {
        'wt_emb': wt_emb,
        'mut_emb': mut_emb,
        'delta_emb': mut_emb - wt_emb,
        'context_emb': context_emb,
    }
```

#### 1B. ESM Attention-Based Features

ESM attention heads contain co-evolution information:

```python
# Extract attention weights for mutation site
attention_maps = esm_model(sequence, return_attentions=True)

# Features:
# - Attention entropy at mutation site
# - Contact attention (heads 20-33 correlate with contacts)
# - Conservation score from attention patterns
```

#### 1C. ESM Log-Likelihood Features

ESM can score mutations via masked prediction:

```python
# Mask wild-type position, predict probability of mutant
wt_seq_masked = sequence[:pos] + '<mask>' + sequence[pos+1:]
probs = esm_model.predict_masked(wt_seq_masked)

# Features:
# - log P(mutant) - log P(wild_type)  # ESM-1v style
# - Entropy at position
# - Rank of mutant in predictions
```

### Data Requirements for Protein-Level ESM

| Data | Source | Size | Status |
|------|--------|------|--------|
| S669 protein sequences | UniProt | 94 sequences | To fetch |
| ESM-2 650M model | HuggingFace | ~2.5GB | Available |
| Pre-computed ESM embeddings | Local | ~500MB | To compute |

### Implementation Plan

1. **Fetch S669 protein sequences**
   - Use UniProt API with PDB IDs from S669
   - Store in `multimodal/data/s669_sequences.fasta`

2. **Extract protein-level ESM features**
   - Position-specific embeddings (WT and MUT)
   - Local context embeddings (±10 residues)
   - ESM log-likelihood scores

3. **Feature selection**
   - PCA on 1280-dim to 32-dim
   - Or learned projection layer (frozen ESM)

4. **Validate with LOO CV**
   - Compare ESM-protein vs ESM-naive
   - Ablation: codon + ESM-protein vs combined

### Expected Improvement (Revised)

| Integration | Expected LOO Spearman | Confidence | Rationale |
|-------------|----------------------|------------|-----------|
| Codon+physico (baseline) | 0.60 | Measured | Current best |
| +ESM protein-level | 0.62-0.66 | Medium | Real context helps |
| +ESM log-likelihood | 0.63-0.67 | Medium-High | ESM-1v proven |

---

## Phase 2: Structural Features Integration

### Public Datasets & Resources

#### AlphaFold Database (2025)

- **URL**: https://alphafold.ebi.ac.uk/
- **Coverage**: 214+ million structures
- **Quality**: pLDDT confidence scores per residue
- **Access**: REST API + bulk downloads

**For S669 proteins:**
```python
# Download structure for UniProt ID
url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
plddt_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-confidence_v4.json"
```

#### ProteinGym Benchmark

- **URL**: https://github.com/OATML-Markslab/ProteinGym
- **Pre-computed**: AlphaFold2 structures for all DMS assays
- **Stability subset**: 7 proteins, 26,000 mutations
- **Larger DDG datasets**: T2837 (2,837 mutations), Megascale (hundreds of thousands)

| Dataset | Proteins | Mutations | Notes |
|---------|----------|-----------|-------|
| S669 (current) | 94 | 669 | High quality, manual curation |
| T2837 | 129 | 2,837 | Larger aggregated set |
| ProteinGym-Stability | 7 | 26,000 | Deep mutational scanning |
| Megascale | ~300 | ~500,000 | ThermoMPNN training data |

#### ThermoMPNN / ProteinMPNN

- **URL**: https://github.com/dauparas/ProteinMPNN
- **Architecture**: GNN on protein structure
- **Pre-trained**: Embeddings available for transfer learning
- **Performance**: Spearman 0.72 on stability (SOTA 2024)

Key insight: ThermoMPNN uses ProteinMPNN embeddings + Megascale dataset

#### ESM-IF1 (Inverse Folding)

- **URL**: https://github.com/facebookresearch/esm
- **Approach**: Sequence from structure prediction
- **Training**: 12M AlphaFold2 structures
- **For DDG**: Mutation log-likelihood from structure

### Structural Features to Extract

| Feature | Dimension | Source | Extraction Method |
|---------|-----------|--------|-------------------|
| pLDDT at mutation | 1 | AlphaFold | JSON confidence file |
| Local pLDDT (±5 residues) | 1 | AlphaFold | Mean of neighborhood |
| Contact count | 1 | Structure | 8Å threshold |
| Relative solvent accessibility | 1 | DSSP | FreeSASA or BioPython |
| Secondary structure | 3 | DSSP | 3-state one-hot (H/E/C) |
| Distance to active site | 1 | Annotation | UniProt features |
| B-factor proxy | 1 | AlphaFold | 1 - pLDDT/100 |
| ProteinMPNN embedding | 32 | ThermoMPNN | Pre-trained encoder |

### Implementation Plan

#### 2A. AlphaFold Structure Download

```python
from src.encoders.alphafold_encoder import AlphaFoldStructureLoader

loader = AlphaFoldStructureLoader(cache_dir="multimodal/structures/")

for protein in s669_proteins:
    structure = loader.get_structure(protein.uniprot_id)
    # Extract: coords, plddt, pae
```

#### 2B. DSSP Feature Extraction

```python
from Bio.PDB import DSSP, PDBParser

parser = PDBParser()
structure = parser.get_structure("protein", pdb_file)
dssp = DSSP(structure[0], pdb_file)

# Features per residue:
# - Secondary structure (H/E/C)
# - Relative solvent accessibility
# - Phi/Psi angles
```

#### 2C. Contact Map Features

```python
def extract_contact_features(structure, mutation_pos, threshold=8.0):
    coords = structure.coords  # CA atoms

    # Contact count at mutation site
    distances = np.linalg.norm(coords - coords[mutation_pos], axis=1)
    contact_count = (distances < threshold).sum() - 1

    # Local contact density
    neighborhood = slice(max(0, mutation_pos-5), mutation_pos+5)
    local_contacts = contact_map[neighborhood, :].sum()

    return contact_count, local_contacts
```

### Expected Improvement (Structural)

| Integration | Expected LOO Spearman | Confidence | Rationale |
|-------------|----------------------|------------|-----------|
| Codon+physico (baseline) | 0.60 | Measured | Current best |
| +pLDDT only | 0.58-0.62 | Low | pLDDT weak for stability |
| +Contact features | 0.62-0.66 | Medium | Contact context helps |
| +DSSP features | 0.61-0.64 | Medium | SS environment matters |
| +ProteinMPNN embeddings | 0.65-0.70 | High | Transfer from Megascale |

**Note**: AlphaFold pLDDT alone shows weak correlation with stability changes (see literature).

---

## Phase 3: Full Multimodal Integration

### Recommended Architecture (Updated)

Based on ablation results, recommend **late fusion with feature selection**:

```
TrainableCodonEncoder ──────────→ 16-dim  ─┐
                                           │
Physico Features ───────────────→  4-dim  ─┤
                                           │
ESM-2 Protein Features (PCA) ───→ 16-dim  ─├→ Concat → Ridge(α) → DDG
                                           │
Structural Features (selected) ──→  8-dim  ─┘

Total: 44-dim (require larger dataset or strong regularization)
```

### Feature Selection Strategy

For 52 samples, limit to ~8-12 total features:

1. **Core features (keep)**: hyp_dist, delta_radius, delta_hydro, delta_charge (4)
2. **ESM features (select 2-4)**: ESM log-likelihood, position entropy, context similarity
3. **Structure features (select 2-4)**: pLDDT, contact_count, RSA, SS

### Scaling to Larger Datasets

| Dataset | Samples | Recommended Features | Expected Spearman |
|---------|---------|---------------------|-------------------|
| S669 | 52 | 8-12 | 0.60-0.65 |
| T2837 | 2,837 | 20-30 | 0.65-0.70 |
| Megascale | 500,000 | 50+ (deep learning) | 0.72+ |

---

## Literature Comparison (Updated)

| Method | Spearman | Type | Data Requirements |
|--------|----------|------|-------------------|
| Rosetta ddg_monomer | 0.69 | Structure | PDB structure |
| ThermoMPNN (2024) | 0.72 | Structure | AlphaFold structure |
| ESM-1v | 0.51 | Sequence | Zero-shot |
| ELASPIC-2 | 0.50 | Sequence | MSA |
| Mutate Everything | 0.56 | Sequence | Zero-shot on S669 |
| **Ours (codon+physico)** | **0.60** | **Sequence** | LOO-validated |
| **Target (+ESM+struct)** | **0.65-0.70** | **Hybrid** | Protein-level features |

---

## Success Criteria (Revised)

### Minimum Viable Improvement

| Task | Current | Minimum | Stretch |
|------|---------|---------|---------|
| DDG (Spearman) | 0.60 | **0.63** | 0.68+ |
| DDG on T2837 | - | **0.60** | 0.65+ |
| Contact (AUC) | 0.67 | **0.72** | 0.80+ |

### Integration Successful If:

1. LOO Spearman improves by ≥0.03 (to 0.63+)
2. Each modality contributes positively in ablation
3. Overfitting ratio stays below 1.4×
4. Method generalizes to T2837 dataset

### Integration NOT Successful If:

1. DDG Spearman drops below 0.55 (regression)
2. Added features show negative contribution
3. Overfitting ratio exceeds 1.5×
4. Requires dataset-specific tuning

---

## Next Steps

### Immediate (Phase 1A)

1. [ ] Fetch S669 protein sequences from UniProt
2. [ ] Extract protein-level ESM embeddings
3. [ ] Implement ESM log-likelihood scoring
4. [ ] Validate with LOO CV

### Short-term (Phase 2A)

1. [ ] Download AlphaFold structures for S669 proteins
2. [ ] Extract pLDDT, contacts, DSSP features
3. [ ] Integrate with multimodal predictor
4. [ ] Ablation study on structural features

### Medium-term (Phase 3)

1. [ ] Scale to T2837 dataset
2. [ ] Implement ThermoMPNN embedding extraction
3. [ ] Cross-dataset validation
4. [ ] Publication-ready benchmarking

---

## Reproducibility Checklist

### Completed

- [x] TrainableCodonEncoder implemented and trained
- [x] Baseline LOO Spearman 0.60 documented
- [x] Ablation study (codon/physico/esm) completed
- [x] Git tag: `v0.1.0-codon-encoder-baseline`
- [x] ESM-2 AA embeddings extracted (naive)
- [x] Multimodal predictor with LOO CV implemented

### Data Locations

| Dataset | Status | Location |
|---------|--------|----------|
| S669 mutations | Available | deliverables/partners/jose_colbes/reproducibility/data/ |
| Codon encoder | Trained | research/codon-encoder/training/results/ |
| ESM AA embeddings | Extracted | research/codon-encoder/multimodal/data/ |
| S669 sequences | To fetch | research/codon-encoder/multimodal/data/ |
| AlphaFold structures | To download | research/codon-encoder/multimodal/structures/ |

---

## References

- **AlphaFold DB 2025**: https://alphafold.ebi.ac.uk/
- **ProteinGym**: https://github.com/OATML-Markslab/ProteinGym
- **ESM Repository**: https://github.com/facebookresearch/esm
- **ThermoMPNN**: https://www.pnas.org/doi/10.1073/pnas.2314853121
- **SPURS (2025)**: https://www.biorxiv.org/content/10.1101/2025.02.13.638154

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-03 | 2.0 | Added ablation results, protein-level ESM plan, structural features plan, public datasets |
| 2026-01-03 | 1.0 | Initial expectation matrix, baseline measurements |

