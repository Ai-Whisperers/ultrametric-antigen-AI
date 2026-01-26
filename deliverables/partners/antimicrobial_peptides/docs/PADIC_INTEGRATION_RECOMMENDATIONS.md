# P-adic Feature Integration Recommendations for Carlos Brizuela AMP Package

**Doc-Type:** Integration Analysis | Version 1.0 | 2026-01-05

---

## Executive Summary

Based on the p-adic AA validation results from `research/codon-encoder/padic_aa_validation/` and the regime analysis from the PeptideVAE validation, we identify specific integration opportunities to improve AMP activity prediction, particularly for the underperforming regimes.

---

## Current Performance Analysis

### Regime-Specific Performance

| Regime | Spearman r | N | Status |
|--------|------------|---|--------|
| **Short (≤15)** | 0.66 | 84 | Good |
| **Medium (16-25)** | 0.61 | 143 | Good |
| **Long (>25)** | **0.14** | 83 | **Critical gap** |
| **Hydrophilic (<0.2)** | 0.68 | 182 | Good |
| **Balanced (0.2-0.5)** | 0.41 | 54 | Moderate |
| **Hydrophobic (>0.5)** | **0.15** | 74 | **Critical gap** |

### Key Findings
1. **Long peptides:** Near-random performance (r=0.14)
2. **Hydrophobic peptides:** Near-random performance (r=0.15)
3. **Gap magnitude:** 0.53 correlation difference between best/worst regimes

---

## P-adic Validation Insights

From `padic_aa_validation/docs/PADIC_ENCODER_FINDINGS.md`:

| Mutation Type | P-adic Advantage | Relevance to AMPs |
|---------------|------------------|-------------------|
| neutral→charged | **+159%** | Cationic AMP residues (K, R, H) |
| small DDG | **+23%** | Subtle MIC variations |
| charge_reversal | **-737%** | Avoid for charge-balanced peptides |

### Key Discovery: `val_product` Feature

The valuation interaction term `wt_val × mt_val` is more informative than individual valuations. This captures pairwise relationships in the hydrophobicity-ordered AA hierarchy.

---

## Integration Recommendations

### 1. For Long Peptides (r=0.14 → target: 0.40+)

**Problem:** Current model fails on long peptides.

**Solution:** Use segment-based p-adic encoding from `src/encoders/segment_codon_encoder.py`

```python
from src.encoders.segment_codon_encoder import SegmentCodonEncoder

# Split long peptides into overlapping segments
encoder = SegmentCodonEncoder(
    segment_size=15,      # Optimal short length
    overlap=7,            # 50% overlap
    latent_dim=16,
    hidden_dim=64,
)

# Encode long peptide as hierarchical aggregation of segments
z_hyp = encoder(codon_indices, lengths)
```

**Rationale:** Short segments (≤15) have r=0.66. Aggregating multiple strong predictions should improve long peptide performance.

---

### 2. For Hydrophobic Peptides (r=0.15 → target: 0.35+)

**Problem:** Model fails on hydrophobic peptides.

**Insight:** Our p-adic validation shows ordering by hydrophobicity helps (r=0.36 matches baseline).

**Solution:** Add position-specific hydrophobicity encoding

```python
# In peptide_encoder.py MultiComponentEmbedding
def forward(self, tokens, positions):
    # Existing embeddings
    aa_emb = self.aa_embedding(tokens)      # 32D
    padic_emb = self.padic_embedding(...)   # 16D
    prop_emb = self.property_encoder(...)   # 8D

    # NEW: Position-specific hydrophobicity encoding
    hydro_idx = self.get_hydrophobicity_index(tokens)  # 0-19 ordered
    hydro_emb = self.hydro_position_encoder(hydro_idx, positions)  # 8D

    return torch.cat([aa_emb, padic_emb, prop_emb, hydro_emb], dim=-1)
```

**Rationale:** Explicit hydrophobicity ordering captures what p-adic structure encodes implicitly.

---

### 3. For Cationic/Hydrophilic Peptides (r=0.68, already good)

**Problem:** Already performing well, but room for improvement.

**Solution:** Add p-adic valuation features from our validation

```python
# Add to feature extraction
def get_padic_features(sequence, encoder):
    """Extract validated p-adic features."""
    embeddings = encoder.get_aa_embeddings()
    valuations = encoder.get_aa_valuations()

    features = []
    for i, aa in enumerate(sequence):
        if i > 0:
            prev_aa = sequence[i-1]
            # Key discovery: valuation product
            val_product = valuations[aa] * valuations[prev_aa]
            features.append(val_product)

    return features
```

**Expected improvement:** +5-10% based on validation results.

---

### 4. Adaptive Model Selection

Based on p-adic validation finding that benefits are context-dependent:

```python
class AdaptivePeptideEncoder(nn.Module):
    """Route to appropriate model based on peptide properties."""

    def __init__(self, ...):
        self.short_encoder = PeptideEncoderTransformer(...)  # Standard
        self.long_encoder = SegmentCodonEncoder(...)         # Hierarchical
        self.charge_classifier = nn.Linear(...)              # Routing

    def forward(self, sequences, lengths, properties):
        # Classify peptide type
        is_long = lengths > 25
        is_hydrophobic = properties[:, 0] > 0.5  # Mean hydrophobicity

        # Route to appropriate encoder
        z_short = self.short_encoder(sequences[~is_long])
        z_long = self.long_encoder(sequences[is_long])

        # Don't use p-adic for hydrophobic (p-adic hurts there)
        z_hydrophobic = self.physico_only_encoder(sequences[is_hydrophobic])

        return combine_embeddings(z_short, z_long, z_hydrophobic)
```

---

## Feature Engineering Additions

### Recommended New Features (14 total from validation)

Add these to `scripts/dramp_activity_loader.py`:

```python
def compute_padic_features(sequence: str) -> Dict[str, float]:
    """Compute p-adic features for peptide."""

    # Hydrophobicity ordering
    HYDRO_ORDER = ['I', 'F', 'V', 'L', 'W', 'M', 'A', 'C', 'G', 'T',
                   'S', 'P', 'Y', 'H', 'N', 'Q', 'E', 'D', 'K', 'R']
    aa_to_idx = {aa: i for i, aa in enumerate(HYDRO_ORDER)}

    # Valuation (p=5 for 20 AAs)
    def valuation(n, p=5):
        if n == 0: return 5
        v = 0
        while n % p == 0:
            n //= p
            v += 1
        return v

    # Compute features
    valuations = [valuation(aa_to_idx.get(aa, 10)) for aa in sequence]

    features = {
        # Mean valuation
        'mean_valuation': np.mean(valuations),
        # Max valuation (highest hierarchy level)
        'max_valuation': max(valuations),
        # Valuation variance
        'valuation_variance': np.var(valuations),
        # Sum of pairwise val_product (key discovery!)
        'sum_val_product': sum(
            valuations[i] * valuations[i+1]
            for i in range(len(valuations)-1)
        ) / max(len(valuations)-1, 1),
        # Valuation gradient (N→C trend)
        'valuation_gradient': (
            np.mean(valuations[-5:]) - np.mean(valuations[:5])
            if len(valuations) >= 10 else 0
        ),
    }

    return features
```

---

## Implementation Priority

| Priority | Enhancement | Target Improvement | Effort |
|----------|-------------|-------------------|--------|
| **1** | Segment encoder for long peptides | r: 0.14 → 0.40 | Medium |
| **2** | val_product feature | r: +5-10% | Low |
| **3** | Adaptive model routing | Overall +10% | High |
| **4** | Hydrophobicity position encoding | r: 0.15 → 0.30 | Medium |

---

## Validation Plan

After implementing each enhancement:

```bash
# Re-run regime analysis
python validation/regime_analysis.py --checkpoint <new_checkpoint>

# Verify improvements
# - Long peptides should improve from 0.14 to 0.30+
# - Hydrophobic should improve from 0.15 to 0.25+
# - Overall should improve from 0.59 to 0.65+
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/dramp_activity_loader.py` | Add `compute_padic_features()` |
| `src/encoders/peptide_encoder.py` | Add segment encoder option |
| `training/train_peptide_encoder.py` | Add adaptive routing logic |
| `validation/regime_analysis.py` | Track p-adic feature impact |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| P-adic features hurt some peptides | Use adaptive routing (skip for hydrophobic) |
| Segment encoder adds complexity | Start with val_product feature (low effort) |
| Overfitting on small regimes | Keep stratified CV, monitor per-regime performance |

---

## Conclusion

The p-adic AA validation provides clear integration guidance:

1. **Use p-adic for:** Short, hydrophilic, cationic peptides (majority of AMPs)
2. **Skip p-adic for:** Long, hydrophobic peptides (use segment encoding instead)
3. **Key feature:** `val_product` interaction term provides orthogonal information

Expected overall improvement: **r: 0.59 → 0.65+** with targeted regime fixes.
