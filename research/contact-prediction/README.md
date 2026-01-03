# Contact Map Prediction from P-adic Codon Embeddings

**Doc-Type:** Research Initiative · Version 1.0 · Created 2026-01-03

---

## Hypothesis

Pairwise relationships between codon embeddings in hyperbolic space can predict residue-residue contacts in 3D protein structures, enabling *de novo* structure prediction from codon sequences alone.

**Key insight:** The p-adic ultrametric structure encodes hierarchical nesting—exactly how secondary structures nest within tertiary folds. If hyperbolic distance between codon embeddings correlates with 3D contact probability, we can bypass amino-acid-only methods like AlphaFold.

---

## Why This Approach Was Not Tried Before

Previous work focused on **single-codon properties**:
- Radius → mass correlation (ρ = 0.76)
- Force constants: k = radius × mass / 100 (ρ = 0.86)
- ΔΔG prediction from mass-based features (ρ = 0.94)
- PTM effects from radial shifts

**Never explored:** Pairwise codon interactions for contact prediction.

---

## Checkpoints Selected

| Checkpoint | Coverage | Hierarchy | Richness | Purpose |
|------------|----------|-----------|----------|---------|
| **final_rich_lr5e5** | 100% | -0.6932 | **0.00858** | Highest richness - best for pairwise diversity |
| **homeostatic_rich** | 100% | -0.6944 | 0.00662 | Balanced approach |
| **v5_11_structural** | 100% | -0.8320 | ~0.003 | Ceiling hierarchy for comparison |

**Why richness matters:** High-richness models have geometric diversity within valuation levels. Low-richness (collapsed shell) models would give identical pairwise distances for all codons at the same valuation, losing discriminative power for contact prediction.

---

## Files

```
research/contact-prediction/
├── README.md                              # This file
├── checkpoints/
│   ├── final_rich_lr5e5_best.pt          # Highest richness checkpoint
│   ├── homeostatic_rich_best.pt          # Balanced checkpoint
│   └── v5_11_structural_best.pt          # Ceiling hierarchy checkpoint
├── embeddings/
│   ├── v5_11_3_embeddings.pt             # Pre-extracted embeddings (z_A_hyp, z_B_hyp)
│   └── codon_mapping_3adic.json          # Codon → position mapping
├── scripts/                               # Future analysis scripts
└── data/                                  # Contact map datasets
```

---

## Proposed Experiments

### Experiment 1: Correlation Analysis

For known protein structures with contact maps (Cα < 8Å threshold):
1. Map protein sequence to codon embeddings
2. Compute pairwise hyperbolic distances: d_hyp(codon_i, codon_j)
3. Correlate with binary contact map
4. Measure Spearman correlation and AUC-ROC

**Expected signal:** Negative correlation (closer in hyperbolic space → more likely to contact)

### Experiment 2: Valuation Difference

Test if difference in p-adic valuation predicts contact:
- v(codon_i) - v(codon_j) → contact probability
- Hypothesis: Similar valuations (same "level" in hierarchy) may contact more

### Experiment 3: Angular Separation

The 16D embedding has angular structure beyond radius:
- θ(codon_i, codon_j) = angle between embeddings
- May encode interaction preferences

### Experiment 4: Lightweight Contact Predictor

Train simple model: (emb_i, emb_j) → P(contact)
- Input: Concatenated or differenced embeddings
- Architecture: 2-layer MLP
- Data: SCOPe/CATH domains with contact maps

---

## Data Sources

**Protein structures with contacts:**
- PDB (experimental)
- AlphaFold DB (predicted, high confidence only)
- SCOPe/CATH domain databases

**Codon sequences:**
- Ensembl CDS sequences
- UniProt cross-references

---

## Success Criteria

| Metric | Threshold | Significance |
|--------|-----------|--------------|
| Spearman (distance vs contact) | < -0.3 | Signal exists |
| AUC-ROC (contact prediction) | > 0.65 | Predictive power |
| Precision@L (top L contacts) | > 0.3 | Useful for structure |

---

## Code Pattern

```python
import torch
import numpy as np
from src.geometry import poincare_distance

# Load embeddings
emb_data = torch.load('embeddings/v5_11_3_embeddings.pt')
z_hyp = emb_data['z_B_hyp']  # Use VAE-B for hierarchy

# Compute pairwise distances for a protein
def compute_pairwise_distances(codon_indices: list, c: float = 1.0) -> np.ndarray:
    """Compute hyperbolic distance matrix for codon sequence."""
    n = len(codon_indices)
    embs = z_hyp[codon_indices]

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = poincare_distance(
                embs[i:i+1], embs[j:j+1], c=c
            ).item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix

# Compare with contact map
def evaluate_correlation(dist_matrix: np.ndarray, contact_map: np.ndarray):
    """Correlate hyperbolic distances with contacts."""
    # Extract upper triangle
    triu_idx = np.triu_indices(len(dist_matrix), k=1)
    distances = dist_matrix[triu_idx]
    contacts = contact_map[triu_idx]

    from scipy.stats import spearmanr
    return spearmanr(distances, contacts)
```

---

## Next Steps

1. Download SCOPe domain dataset with contact maps
2. Map domains to CDS codon sequences (Ensembl)
3. Run Experiment 1 correlation analysis
4. If signal exists (ρ < -0.2), proceed to Experiment 4

---

## References

- AlphaFold contact prediction breakthrough
- P-adic numbers and ultrametric spaces
- Hyperbolic embeddings for hierarchical data
- This project's ΔΔG and force constant discoveries

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-03 | 1.0 | Initial creation with checkpoints and approach |
