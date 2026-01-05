# P-adic Codon Encoder Research

**Doc-Type:** Research Documentation · Version 2.0 · Updated 2026-01-03 · AI Whisperers

Research scripts for validating p-adic codon embeddings against physical ground truth.

---

## Latest: TrainableCodonEncoder (v1.0)

**Key Achievement:** LOO Spearman **0.61** on DDG prediction (S669), +105% over baseline.

| Method | LOO Spearman | Type |
|--------|--------------|------|
| Rosetta ddg_monomer | 0.69 | Structure |
| **TrainableCodonEncoder** | **0.61** | **Sequence** |
| ELASPIC-2 (2024) | 0.50 | Sequence |
| FoldX | 0.48 | Structure |
| Baseline (p-adic) | 0.30 | Sequence |

### Quick Start

```python
from src.encoders import TrainableCodonEncoder
import torch

# Load trained encoder
encoder = TrainableCodonEncoder(latent_dim=16, hidden_dim=64)
ckpt = torch.load('research/codon-encoder/training/results/trained_codon_encoder.pt')
encoder.load_state_dict(ckpt['model_state_dict'])
encoder.eval()

# Get all codon embeddings (64 × 16 on Poincaré ball)
z_hyp = encoder.encode_all()

# Get amino acid embeddings
aa_embeddings = encoder.get_all_amino_acid_embeddings()

# Compute distance between amino acids
dist = encoder.compute_aa_distance('A', 'V')
```

### Training

```bash
python research/codon-encoder/training/train_codon_encoder.py --epochs 500
```

---

## Key Discoveries

| Analysis | Finding | Correlation |
|----------|---------|-------------|
| Dimension 13 | "Physics dimension" - encodes mass, volume, force constants | ρ = -0.695 |
| Radial structure | Encodes amino acid mass | ρ = +0.760 |
| Force constant formula | `k = radius × mass / 100` | **ρ = 0.860** |
| Vibrational frequency | `ω = √(k/m)` - derivable from embeddings | ρ = 1.000 |

---

## Directory Structure

```
codon-encoder/
├── README.md               # This file
├── config.py               # Shared configuration and paths
├── __init__.py
│
├── extraction/             # Embedding extraction utilities
│   ├── ANALYSIS_SUMMARY.md          # VAE vs codon mapping analysis
│   ├── extract_hyperbolic_embeddings.py
│   └── results/
│       └── codon_embeddings_v5_12_3.json
│
├── training/               # Model training scripts
│   ├── train_codon_encoder.py       # Main TrainableCodonEncoder training
│   ├── ddg_hyperbolic_training.py   # Hyperbolic DDG baseline
│   ├── ddg_vae_embeddings.py        # VAE embeddings DDG
│   ├── ddg_predictor_training.py    # sklearn-based training
│   ├── ddg_pytorch_training.py      # PyTorch hyperparameter search
│   └── results/
│       ├── trained_codon_encoder.pt   # Best model weights
│       ├── trained_codon_encoder.json # Training metrics
│       └── ddg_*.json                 # DDG evaluation results
│
├── benchmarks/             # Validation benchmarks
│   ├── mass_vs_property_benchmark.py
│   ├── kinetics_benchmark.py
│   ├── deep_physics_benchmark.py
│   └── ddg_benchmark.py
│
├── padic_aa_validation/    # P-adic AA encoder validation (NEW)
│   ├── README.md                    # Validation suite documentation
│   ├── scripts/
│   │   ├── 01_baseline_benchmark.py     # Physico vs p-adic
│   │   ├── 02_ordered_indices.py        # AA ordering experiments
│   │   ├── 03_feature_combinations.py   # Feature combination tests
│   │   ├── 04_statistical_validation.py # Bootstrap CI, permutation
│   │   └── 05_feature_optimization.py   # Greedy selection
│   ├── results/                     # All validation results (JSON)
│   └── docs/
│       └── PADIC_ENCODER_FINDINGS.md    # Comprehensive findings
│
├── analysis/               # Embedding space analysis
│   ├── proteingym_pipeline.py
│   └── padic_dynamics_predictor.py
│
├── pipelines/              # Integration pipelines
│   ├── padic_3d_dynamics.py
│   ├── af3_pipeline.py
│   └── ptm_mapping.py
│
└── development/            # Historical development scripts
    ├── 01_bioinformatics_analysis.py
    └── ...
```

---

## Architecture: TrainableCodonEncoder

### Input Encoding (12-dim)

```
Codon: ATG (Methionine)
       ↓
Position 1 (A): [1, 0, 0, 0]  # A=0, C=1, G=2, T=3
Position 2 (T): [0, 0, 0, 1]
Position 3 (G): [0, 0, 1, 0]
       ↓
Input: [1,0,0,0, 0,0,0,1, 0,0,1,0] (12-dim)
```

### Network Architecture

```
Input:    12-dim one-hot
          ↓
Linear:   12 → 64
LayerNorm + SiLU + Dropout(0.1)
          ↓
Linear:   64 → 64
LayerNorm + SiLU + Dropout(0.1)
          ↓
Linear:   64 → 16 (tangent space)
          ↓
exp_map:  Tangent → Poincaré ball
          ↓
Output:   16-dim hyperbolic embedding
```

### Loss Function

| Component | Weight | Formula | Purpose |
|-----------|--------|---------|---------|
| Radial | 1.0 | MSE(r, target_r) | Hierarchy → radius |
| P-adic | 1.0 | 1 - corr(d_hyp, d_padic) | Distance preservation |
| Cohesion | 0.5 | mean(d_synonymous) | Same AA clusters |
| Separation | 0.3 | ReLU(margin - d_centroid) | Different AAs separate |
| Property | 0.5 | 1 - corr(d_aa, d_props) | Property correlation |

---

## Reproducibility

### Environment

```
Python: 3.12
PyTorch: 2.0+
geoopt: 0.5.0+
scipy: 1.10+
sklearn: 1.3+
```

### Training Configuration

```python
# Hyperparameters used for reported results
epochs = 500
lr = 0.001
latent_dim = 16
hidden_dim = 64
dropout = 0.1
weight_decay = 1e-4
optimizer = AdamW
scheduler = CosineAnnealingLR

# Loss weights
radial_weight = 1.0
padic_weight = 1.0
cohesion_weight = 0.5
separation_weight = 0.3
property_weight = 0.5
```

### Expected Results

```
Epoch 500/500: total=6.45, radial=5.84, padic=0.04, cohesion=0.14

Embedding Analysis:
  Radius range: [2.84, 2.94]
  P-adic correlation: 0.74

DDG Prediction (LOO):
  Spearman: 0.61
  Pearson:  0.64
  MAE:      0.81
  RMSE:     1.11
```

---

## Benchmarks

### Run all benchmarks

```bash
python benchmarks/deep_physics_benchmark.py
python benchmarks/mass_vs_property_benchmark.py
python benchmarks/kinetics_benchmark.py
python benchmarks/ddg_benchmark.py
```

### Thermodynamics vs Kinetics

| Task Type | Winner | Best Feature |
|-----------|--------|--------------|
| Stability (ΔΔG) | Mass-based | padic_mass ρ=0.94 |
| Folding rates | Property-based | property ρ=0.94 |
| Aggregation | Property-based | property |

### Physics Level Correlations

| Level | Physics | P-adic Correlation |
|-------|---------|-------------------|
| 0 | Biochemistry | ρ = 0.760 (mass) |
| 1 | Classical mechanics | ρ = 0.665 (moment) |
| 2 | Statistical mechanics | ρ = 0.517 (entropy) |
| 3 | Vibrational (force constants) | **ρ = 0.608** |
| 4 | Experimental dynamics | ρ = -0.01 (B-factor) |
| 5 | Quantum corrections | ρ = 0.466 |

**Insight:** P-adic encoder captures thermodynamic invariants but NOT local experimental dynamics.

---

## Technical Notes

### Why 12-dim Input?

| Approach | Dims | Issue |
|----------|------|-------|
| 9-dim ternary (mod 3) | 9 | Information loss: T≡A (mod 3) |
| VAE operations | 9 | 4-adic codons don't map to 3-adic |
| **12-dim one-hot** | 12 | **No information loss** |

### Overfitting Analysis

| Metric | v2 Predictor | TrainableCodonEncoder |
|--------|--------------|----------------------|
| Train Spearman | 0.80 | 0.76 |
| LOO Spearman | 0.30 | 0.61 |
| Overfitting ratio | 2.67× | **1.25×** |

---

## Future Work

1. **ESM Integration**: Combine with ESM-2 contextual embeddings
2. **Structural Features**: AlphaFold-derived contact maps, DSSP
3. **Multi-task Learning**: Joint DDG + folding kinetics + solubility
4. **Larger Datasets**: ProteinGym (OPTIONAL), Mega-scale mutational scans

**Note on ProteinGym:** The S669 dataset (43 MB, 2,800 mutations) is sufficient for current DDG validation (LOO Spearman 0.61). ProteinGym (~500 MB) is an optional future enhancement for extended protein family validation. See `data/proteingym/README.md` for details.

---

## Citation

```bibtex
@software{ternary_vae_codon_encoder,
  title={Trainable Codon Encoder for Protein Stability Prediction},
  author={AI Whisperers},
  year={2026},
  url={https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics}
}
```

---

## P-adic Amino Acid Validation (NEW)

When codon information is unavailable (proteomics-only), can we use p-adic structure on amino acids?

**Summary:** +7-9% improvement over baseline, but context-dependent.

| Mutation Type | P-adic Benefit |
|---------------|----------------|
| neutral→charged | **+159%** |
| small DDG (<1) | **+23%** |
| charge reversal | **-737%** (harmful) |

**See:** `padic_aa_validation/README.md` for full validation suite.

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-05 | 2.1 | P-adic AA validation suite with 5-stage pipeline |
| 2026-01-03 | 2.0 | TrainableCodonEncoder (LOO ρ=0.61), full documentation |
| 2026-01-03 | 1.5 | HyperbolicCodonEncoder, overfitting analysis |
| 2025-12-28 | 1.0 | Initial physics benchmarks, dimension analysis |
