# Riemann Hypothesis Sandbox

**Purpose:** Investigate spectral connections between 3-adic hyperbolic embeddings and the Riemann zeta function.

---

## Model Reference

**Production Model:** v1.1.0 (V5.11.3 Structural)
**Checkpoint:** `sandbox-training/checkpoints/v5_11_structural/best.pt`
**Architecture:** `TernaryVAEV5_11_OptionC`

### Loading the Correct Model

```python
from src.models import TernaryVAEV5_11_OptionC
import torch

model = TernaryVAEV5_11_OptionC(
    latent_dim=16,
    hidden_dim=128,           # from config.projection_hidden_dim
    max_radius=0.95,
    curvature=1.0,
    use_controller=False,
    use_dual_projection=True,
    n_projection_layers=2,    # from config.projection_layers
    projection_dropout=0.1
)

ckpt = torch.load('sandbox-training/checkpoints/v5_11_structural/best.pt')
model.load_state_dict(ckpt['model_state'], strict=False)
```

### Verification Metrics

| Metric | Expected | Source |
|:-------|:---------|:-------|
| VAE-B radial_corr | -0.832 | checkpoint['metrics'] |
| v₃=9 radius | 0.087 | VAE-B embedding |
| 3-adic/Poincaré ρ | 0.837 | computed |

---

## Key Results

### What the Model Learned

1. **Perfect 3-adic Radial Hierarchy**
   ```
   v₃=0: r=0.944    v₃=5: r=0.403
   v₃=1: r=0.730    v₃=6: r=0.330
   v₃=2: r=0.617    v₃=7: r=0.257
   v₃=3: r=0.542    v₃=8: r=0.174
   v₃=4: r=0.470    v₃=9: r=0.087
   ```

2. **Radial Formula:** `r(v) = 0.929 × 3^(-0.172v)`
   - Exponent 0.172 ≈ 1/6
   - Compare to Haar measure: 3^(-v) has exponent 1

3. **Exact Ultrametric:** 0 violations in 10,000 tests

### What We Tested

| Test | Result | Interpretation |
|:-----|:-------|:---------------|
| Eigenvalue spacing vs GUE | KS=0.60 | NOT zeta-like |
| Eigenvalue spacing vs Poisson | KS=0.32 | Poisson-like |
| Direct zeta zero correlation | r=0.006 | No connection via Laplacian |
| 3-adic/Poincaré correlation | ρ=0.837 | STRONG learned structure |
| Functional equation symmetry | No | Partition function test failed |

---

## Experiments

### 01_extract_embeddings.py
Extract hyperbolic embeddings from trained model.
- Saves both VAE-A and VAE-B embeddings
- Verifies projection config matches checkpoint

### 02_compute_spectrum.py
Compute graph Laplacian eigenvalues (SLOW - use 02_fast version).

### 02_compute_spectrum_fast.py
Fast spectral analysis using 2000 random samples.

### 03_compare_zeta.py
Compare eigenvalue spacings to Riemann zeta zeros.

### 04_padic_spectral_analysis.py
Analyze 3-adic weighted Laplacian structure.

### 05_exact_padic_analysis.py
Exploit exact 3-adic arithmetic for analysis that binary FP cannot do:
- Counting function N(r) ~ r^α
- Functional equation test
- Exact valuation statistics

### 06_bioinformatics_analysis.py
Apply 3-adic embeddings to codon/protein analysis:
- Map 64 codons to ternary indices
- Test synonymous codon clustering
- Correlate with BLOSUM matrices

### 07_adelic_analysis.py
Multi-prime (adelic) structure exploration:
- Compute valuations for p=2,3,5,7,11,13
- Test adelic vs 3-adic distance correlation
- Construct adelic-weighted Laplacian

### 08_alternative_spectral_operators.py
Test 6 different operators for GUE statistics:
- 3-adic weighted Laplacian
- Hyperbolic (Poincaré) Laplacian
- Radial diagonal operator
- Multiplicative structure matrix
- Heat kernel analysis
- Normalized Laplacian

### run_all.py
Run complete pipeline (use with correct checkpoint path).

---

## Quick Start

```bash
# Step 1: Extract embeddings with correct model
python 01_extract_embeddings.py \
    --checkpoint sandbox-training/checkpoints/v5_11_structural/best.pt

# Step 2: Run exact p-adic analysis
python 05_exact_padic_analysis.py

# Step 3: Run spectral comparison
python 02_compute_spectrum_fast.py --n-samples 2000

# Step 4: Compare to zeta
python 03_compare_zeta.py
```

---

## Conclusions

### Positive Results
1. Model learned genuine 3-adic geometry (ρ = 0.837)
2. Perfect radial hierarchy matching 3-adic valuation
3. Ultrametric structure preserved exactly
4. Radial formula r(v) = 0.929 × 3^(-0.172v) discovered
5. **Radial exponent explained:** c = 1/(latent_dim - n_trits - 1) = 1/6

### Downstream Validation (Production Ready)

| Metric | VAE-A | VAE-B | Interpretation |
|:-------|:------|:------|:---------------|
| NN Same Valuation | 84.4% | **99.9%** | Near-perfect clustering |
| NN Adjacent (±1) | 94.6% | **100%** | Perfect local structure |
| Pairwise Ordering | 92.3% | **100%** | Perfect hierarchy |
| Spearman ρ | -0.728 | **-0.832** | Strong v₃→radius |
| Valuation Prediction | 97.4% | **99.9%** | Embeddings encode v₃ |
| Result Component | 64.7% | **78.7%** | 2.4× random baseline |

**Verdict:** All production readiness checks passed

### Bioinformatics Application (Positive)
1. Synonymous codons cluster in embedding space (p = 6.77e-05)
2. Chemical classes separate by radius (ANOVA p = 0.018)
3. BLOSUM correlation significant (r = -0.106)

### Negative Results (Riemann)
1. Graph Laplacian eigenvalues are Poisson, not GUE
2. No direct correlation with zeta zeros via spacing distribution
3. Partition function doesn't show functional equation symmetry
4. All 6 alternative spectral operators remain Poisson-like
5. Single-prime embedding insufficient for GUE (multi-prime needed)

### Open Questions (Remaining)
1. What spectral operator would produce GUE from this structure?
2. Does multi-prime (adelic) extension yield GUE?
3. Can the radial formula predict prime distribution?

### Answered Questions
1. **Why is the radial exponent 0.172 ≈ 1/6?**
   - **ANSWER:** `c = 1/(latent_dim - n_trits - 1) = 1/(16-9-1) = 1/6`
   - See `DISCOVERY_RADIAL_EXPONENT.md` for full derivation

---

## References

- Montgomery (1973): Pair correlation of zeta zeros
- Odlyzko (1987): Distribution of spacings between zeros
- LMFDB: https://www.lmfdb.org/zeros/zeta/

---

**Last Updated:** 2025-12-16
**Model Version:** v1.1.0 (V5.11.3 Structural)
