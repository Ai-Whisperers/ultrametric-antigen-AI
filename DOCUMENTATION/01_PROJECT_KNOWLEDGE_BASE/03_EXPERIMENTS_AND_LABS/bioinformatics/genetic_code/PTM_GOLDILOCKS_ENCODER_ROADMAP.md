# PTM-Goldilocks Encoder: Training Roadmap

**Doc-Type:** Technical Roadmap - Version 1.0 - Updated 2025-12-19 - Author AI Whisperers

**Prerequisite:** [PTM_EXTENSION_PLAN.md](./PTM_EXTENSION_PLAN.md) - Must complete ground truth extension before training

---

## Executive Summary

Train a unified PTM-Goldilocks encoder that directly outputs therapeutic classification by leveraging the natural 3-adic structure of V5.11.3 embeddings (19,683 = 3^9 lattice points) and validated ground truth from HIV, SARS-CoV-2, and RA disease domains.

---

## Foundation: V5.11.3 Embeddings

| Property | Value |
|:---------|:------|
| Lattice size | 19,683 (3^9 natural 3-adic) |
| Dimensions | 16D Poincare ball |
| Hierarchy correlation | -0.832 |
| Max radius | 0.95 |
| Separation ratio | 4.95x |

The 3-adic structure is **inherited from the embeddings**, not imposed by architecture.

---

## Ground Truth Data Available

### HIV Glycan Shield (N->Q Deglycosylation)

**Source:** `research/bioinformatics/hiv/glycan_shield/glycan_analysis_results.json`

| Metric | Count |
|:-------|:------|
| Total sites | 24 |
| Goldilocks zone (15-30%) | 7 |
| Above zone (>30%) | 17 |

**Labels per site:**
- `centroid_shift` (continuous, 0.17-0.76)
- `goldilocks_zone` (categorical: goldilocks/above)
- `goldilocks_score` (continuous, 0.2-1.2)
- `boundary_crossed` (boolean)

**Goldilocks sites:** N58, N429, N103, N204, N107, N271, N265

### SARS-CoV-2 Handshake Interface (Multiple PTMs)

**Source:** `research/bioinformatics/sars_cov_2/glycan_shield/handshake_analysis_results.json`

| Metric | Count |
|:-------|:------|
| Convergent interface pairs | 14 |
| Asymmetric targets | 40+ |
| PTM types tested | 6 |

**PTM Types with ground truth:**

| PTM | Notation | Example Shifts |
|:----|:---------|:---------------|
| Phosphoserine | S->pS (S->D) | 0.19-0.35 |
| Phosphotyrosine | Y->pY (Y->D) | 0.20-0.21 |
| Phosphothreonine | T->pT (T->D) | 0.18-0.32 |
| Deglycosylation | N->Q | 0.14-0.15 |
| Acetylation | K->Ac (K->Q) | 0.11-0.12 |
| Hydroxylation | P->Hyp | 0.10-0.12 |

**Labels per target:**
- `viral_shift` (continuous)
- `host_shift` (continuous)
- `asymmetry` (viral_shift - host_shift)
- `therapeutic_potential` (categorical: HIGH/MEDIUM/LOW)

### Rheumatoid Arthritis (R->Q Citrullination)

**Source:** Scripts 01-17 in `research/bioinformatics/rheumatoid_arthritis/`

| Metric | Status |
|:-------|:-------|
| Known ACPA targets | Fibrinogen, Vimentin, Alpha-enolase |
| Goldilocks validation | Documented in HANDSHAKE_ANALYSIS_FINDINGS.md |
| HLA correlation | r=0.751 with disease odds ratio |

**Labels:** Goldilocks zone classification (15-30% = autoimmune epitope)

---

## Proposed Architecture

### Input Layer (20D)

```
12D: Codon one-hot (4 nucleotides x 3 positions)
 8D: PTM type one-hot:
     [S->D, T->D, Y->D, N->Q, R->Q, K->Q, M->Q, P->Hyp]
```

### Hidden Layers

```
Linear(20, 32) -> ReLU
Linear(32, 32) -> ReLU
Linear(32, 16) -> Poincare projection
```

### Output Heads

| Head | Output | Loss |
|:-----|:-------|:-----|
| Cluster | 21-class softmax | Cross-entropy |
| Goldilocks | 3-class (below/within/above) | Cross-entropy |
| Asymmetry | Binary (therapeutic/non-therapeutic) | BCE |
| Shift regression | Continuous (0-1) | MSE |

### Loss Function

```python
loss = (
    1.0 * loss_cluster +           # Amino acid clustering
    0.5 * loss_goldilocks +        # Zone classification
    0.3 * loss_asymmetry +         # Therapeutic potential
    0.2 * loss_shift_regression +  # Continuous shift
    0.3 * loss_contrastive         # Poincare distance preservation
)
```

---

## Training Dataset Construction

### Step 1: Extract HIV Ground Truth

```python
# From glycan_analysis_results.json
hiv_samples = []
for site in hiv_results['results']:
    hiv_samples.append({
        'context': site['wt_context'],      # 11-mer
        'ptm_type': 'N->Q',                 # Deglycosylation
        'centroid_shift': site['centroid_shift'],
        'goldilocks_label': 0 if site['goldilocks_zone'] == 'below'
                           else 1 if site['goldilocks_zone'] == 'goldilocks'
                           else 2,  # above
        'therapeutic': site['goldilocks_zone'] == 'goldilocks'
    })
# Result: 24 samples
```

### Step 2: Extract SARS-CoV-2 Ground Truth

```python
# From handshake_analysis_results.json
sars_samples = []
for target in sars_results['asymmetric_targets']:
    ptm_map = {
        'S_to_pS': 'S->D', 'T_to_pT': 'T->D', 'Y_to_pY': 'Y->D',
        'N_to_Q': 'N->Q', 'K_to_Ac': 'K->Q', 'P_to_Hyp': 'P->Hyp'
    }
    sars_samples.append({
        'context': extract_context(target['viral_position']),
        'ptm_type': ptm_map[target['modification']],
        'viral_shift': target['viral_shift'],
        'host_shift': target['host_shift'],
        'asymmetry': target['asymmetry'],
        'therapeutic': target['therapeutic_potential'] == 'HIGH'
    })
# Result: ~40 samples with 6 PTM types
```

### Step 3: Extract RA Ground Truth

```python
# From validated citrullination sites
ra_samples = []
for site in known_acpa_targets:
    ra_samples.append({
        'context': site['context'],
        'ptm_type': 'R->Q',  # Citrullination
        'centroid_shift': site['shift'],
        'goldilocks_label': 1,  # Known autoimmune = Goldilocks
        'therapeutic': False   # These are pathogenic, not therapeutic
    })
# Result: ~20-50 samples
```

### Step 4: Augmentation from V5.11.3 Lattice

```python
# Sample additional contexts from 19,683 natural positions
# Apply each PTM type and compute shifts
# Label by Goldilocks zone (15-30% = 1, else 0 or 2)
augmented_samples = []
for position in natural_positions:
    context = decode_position_to_context(position)
    for ptm_type in ['S->D', 'T->D', 'Y->D', 'N->Q', 'R->Q', 'K->Q']:
        if applicable(context, ptm_type):
            shift = compute_shift(context, ptm_type)
            augmented_samples.append({
                'context': context,
                'ptm_type': ptm_type,
                'centroid_shift': shift,
                'goldilocks_label': classify_goldilocks(shift)
            })
# Result: ~1000-5000 augmented samples
```

### Final Dataset

| Source | Samples | PTM Types | Labels |
|:-------|:--------|:----------|:-------|
| HIV | 24 | N->Q | Goldilocks zone |
| SARS-CoV-2 | 40+ | 6 types | Therapeutic potential |
| RA | 20-50 | R->Q | Goldilocks zone |
| Augmented | 1000-5000 | All | Computed |
| **Total** | **~1100-5100** | **8 types** | **Multi-task** |

---

## Training Protocol

### Phase 1: Pre-training on Augmented Data

- Train on 5000 augmented samples
- Focus on cluster and shift regression heads
- Inherit V5.11.3 structure via center initialization

### Phase 2: Fine-tuning on Validated Data

- Fine-tune on HIV + SARS-CoV-2 + RA ground truth
- Focus on Goldilocks and asymmetry heads
- Use weighted sampling to balance disease domains

### Phase 3: Validation

- Hold out 20% of each disease domain
- Evaluate per-domain and cross-domain generalization
- Compare with current post-hoc computation

---

## Expected Outcomes

### Performance Targets

| Metric | Target |
|:-------|:-------|
| Cluster accuracy | >95% (inherited from 3-adic encoder) |
| Goldilocks classification | >85% |
| Therapeutic asymmetry | >80% AUC |
| Shift regression | r > 0.9 |

### Improvements Over Current Pipeline

| Current | Proposed |
|:--------|:---------|
| Encode -> Compute shift -> Classify | Single forward pass |
| Post-hoc asymmetry calculation | Native asymmetry prediction |
| No confidence scores | Softmax probabilities |
| Disease-specific scripts | Unified encoder |

---

## Integration Plan

### File Structure

```
research/genetic_code/
├── data/
│   ├── codon_encoder_3adic.pt          # Current encoder
│   ├── ptm_goldilocks_encoder.pt       # New encoder
│   ├── ptm_training_dataset.json       # Consolidated ground truth
│   └── v5_11_3_embeddings.pt           # Foundation embeddings
├── scripts/
│   ├── 09_train_codon_encoder_3adic.py # Current
│   ├── 10_consolidate_ptm_ground_truth.py
│   ├── 11_train_ptm_goldilocks_encoder.py
│   └── 12_evaluate_ptm_encoder.py
└── PTM_GOLDILOCKS_ENCODER_ROADMAP.md   # This document
```

### API

```python
from genetic_code import PTMGoldilocksEncoder

encoder = PTMGoldilocksEncoder.load('ptm_goldilocks_encoder.pt')

# Single prediction
result = encoder.predict(
    context='VIAWNSNNLDS',
    ptm_type='S->D',
    position=6
)
# Returns: {
#   'embedding': tensor(16D),
#   'goldilocks_zone': 'goldilocks',
#   'goldilocks_prob': [0.05, 0.85, 0.10],
#   'therapeutic_potential': True,
#   'therapeutic_prob': 0.92,
#   'predicted_shift': 0.21
# }

# Batch prediction for screening
results = encoder.screen_protein(
    sequence='MFVFLVLLPLVSS...',
    ptm_types=['S->D', 'N->Q', 'R->Q']
)
```

---

## Dependencies

- PyTorch >= 2.0
- V5.11.3 embeddings (`v5_11_3_embeddings.pt`)
- Natural positions (`natural_positions_v5_11_3.json`)
- Ground truth from HIV, SARS-CoV-2, RA analyses

---

## Timeline

| Phase | Tasks |
|:------|:------|
| 1 | Consolidate ground truth from 3 diseases |
| 2 | Build augmented dataset from V5.11.3 lattice |
| 3 | Train PTM-Goldilocks encoder |
| 4 | Validate across disease domains |
| 5 | Integrate into analysis pipelines |

---

## References

- Current 3-adic encoder: `research/genetic_code/scripts/09_train_codon_encoder_3adic.py`
- HIV analysis: `research/bioinformatics/hiv/glycan_shield/`
- SARS-CoV-2 analysis: `research/bioinformatics/sars_cov_2/glycan_shield/`
- RA analysis: `research/bioinformatics/rheumatoid_arthritis/`
- Multi-adic design: `research/p-adic-genomics/multi-adic-vaes/MULTI-ADIC-VAE-DESIGN.md`

---

## Changelog

| Date | Version | Description |
|:-----|:--------|:------------|
| 2025-12-19 | 1.0 | Initial roadmap based on ground truth analysis |
