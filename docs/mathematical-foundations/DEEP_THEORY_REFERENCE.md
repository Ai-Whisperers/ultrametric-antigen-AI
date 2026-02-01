# Deep Theory Reference

**Doc-Type:** Mathematical Deep Dive · Extracted from CLAUDE.md · Version 1.0 · 2026-01-30

> This document contains detailed mathematical content extracted from the main CLAUDE.md for users
> who need deep technical understanding of the hyperbolic audit, fix lists, and theoretical limits.

---

## V5.12.2 Hyperbolic Audit - Complete Details

### The Problem

Many files in `src/`, `src/configs/`, and `src/scripts/` incorrectly use Euclidean `.norm()` on hyperbolic Poincaré ball embeddings instead of `poincare_distance()`. This causes:
- Incorrect radial hierarchy computation (coverage stuck at ~20%)
- Metric correlations computed in wrong geometry
- Training scripts producing misleading results

### Correct Pattern

```python
# WRONG - Euclidean norm on hyperbolic embeddings
radius = torch.norm(z_hyp, dim=-1)

# CORRECT - Hyperbolic distance from origin
from src.geometry import poincare_distance
origin = torch.zeros_like(z_hyp)
radius = poincare_distance(z_hyp, origin, c=curvature)
```

### Why This Matters

In the Poincaré ball model:
- Points near the boundary have Euclidean norm ~1 but **infinite** hyperbolic distance from origin
- Euclidean norm severely underestimates distances near the boundary
- Hierarchy metrics computed with Euclidean norm give misleading correlations

---

## V5.12.2 Complete Fix List

**Total: 278 norm() calls analyzed, ~75 need fixing**

### Priority 1: Core Losses (5 files, 7 usages)

| File | Lines | Issue |
|------|-------|-------|
| `src/losses/padic/metric_loss.py` | 84 | `d_latent = torch.norm(z[i] - z[j])` |
| `src/losses/padic/ranking_loss.py` | 97-98 | `d_anchor_pos/neg` triplet distances |
| `src/losses/padic/ranking_v2.py` | 131-132 | `d_anchor_pos/neg` triplet distances |
| `src/losses/set_theory_loss.py` | 105 | `distances = torch.norm(embeddings)` radii |
| `src/losses/objectives/solubility.py` | 231 | `norms = torch.norm(latent)` compactness |

### Priority 2: Core Geometry/Models (5 files, 6 usages)

| File | Lines | Issue |
|------|-------|-------|
| `src/geometry/holographic_poincare.py` | 251 | `holographic_distance` uses Euclidean |
| `src/models/holographic/bulk_boundary.py` | 205 | `distance_to_origin` uses Euclidean |
| `src/models/lattice_projection.py` | 205 | `_adjust_radii` uses Euclidean norms |
| `src/models/contrastive/concept_aware.py` | 445 | `distances = torch.norm(diff)` |
| `src/diseases/rheumatoid_arthritis.py` | 346 | `shift_magnitude` uses Euclidean |

### Priority 3: Encoders (1 file, 1 usage)

| File | Lines | Issue |
|------|-------|-------|
| `src/encoders/codon_encoder.py` | 413 | `emb_dist = torch.norm(emb1 - emb2)` |

### Priority 4: Visualization (2 files, 7 usages)

| File | Lines | Issue |
|------|-------|-------|
| `src/visualization/projections/poincare.py` | 67,102,108,177,260,328 | radii, norms, centroids |
| `src/visualization/plots/manifold.py` | 378 | `euclidean_norms` for radial plot |

### Priority 5: Training/Monitoring (1 file, 2 usages)

| File | Lines | Issue |
|------|-------|-------|
| `src/training/monitoring/tensorboard_logger.py` | 463,466 | `z_A/B_euc_norm` |

### Priority 6: Research Scripts - HIV (4 files, ~12 usages)

| File | Lines | Issue |
|------|-------|-------|
| `hiv/scripts/03_hiv_handshake_analysis.py` | 274,283,295 | `encode_context` clamping |
| `hiv/src/03_hiv_handshake_analysis.py` | 267,276,288 | duplicate file |
| `hiv/scripts/analyze_tropism_switching.py` | 273,340 | centroid_distance, separation |
| `hiv/scripts/esm2_integration.py` | 279,283,460,615,626,720 | ESM2 distances (verify if hyperbolic) |

### Priority 7: Research Scripts - RA (3 files, ~8 usages)

| File | Lines | Issue |
|------|-------|-------|
| `rheumatoid_arthritis/scripts/03_citrullination_analysis.py` | 273,279 | euclidean_shift, cosine_sim |
| `rheumatoid_arthritis/scripts/04_codon_optimizer.py` | 228,273,323,324 | cluster distances |
| `rheumatoid_arthritis/scripts/cross_validate_encoder.py` | 185 | euc_context |

### Priority 8: Research Scripts - Genetic Code (5 files, ~12 usages)

| File | Lines | Issue |
|------|-------|-------|
| `genetic_code/scripts/04_fast_reverse_search.py` | 138,184 | radii, centroid dists |
| `genetic_code/scripts/07_extract_v5_11_3_embeddings.py` | 136,137 | radii_A/B |
| `genetic_code/scripts/09_train_codon_encoder_3adic.py` | 412 | radii |
| `genetic_code/scripts/10_extract_fused_embeddings.py` | 151,152 | radii_A/B |
| `genetic_code/scripts/11_train_codon_encoder_fused.py` | 166,302,395,459 | radii |

### Priority 9: Research Scripts - Spectral Analysis (6 files, ~18 usages)

| File | Lines | Issue |
|------|-------|-------|
| `spectral_analysis/scripts/01_extract_embeddings.py` | 253,254 | radii_A/B |
| `spectral_analysis/scripts/04_padic_spectral_analysis.py` | 94 | radii |
| `spectral_analysis/scripts/05_exact_padic_analysis.py` | 242 | radii |
| `spectral_analysis/scripts/07_adelic_analysis.py` | 65,134,194,275,280,296,341 | radii, emb_dists |
| `spectral_analysis/scripts/08_alternative_spectral_operators.py` | 106,196 | emb_dist, radii |
| `spectral_analysis/scripts/09_binary_ternary_decomposition.py` | 115,181,289 | radii |
| `spectral_analysis/scripts/10_semantic_amplification_benchmark.py` | 195 | radii |

### Priority 10: Experimental (1 file, 1 usage)

| File | Lines | Issue |
|------|-------|-------|
| `_experimental/implementations/literature/literature_implementations.py` | 1191 | `z_norms` debug |

### VERIFIED CORRECT (No Fix Needed)

These 190+ usages are intentionally Euclidean:
- Inside exp_map/log_map/poincare_distance formulas
- `self.norm(x)` = LayerNorm/BatchNorm
- Direction normalization: `v / norm(v)`
- Ball projection/clamping: `norm < max_radius`
- 3D physical coordinates
- Intentionally Euclidean functions (`euclidean_distance`, `cosine_distance`)
- p-adic norm (different mathematical object)
- Convergence checks, gradient norms

**Audit Documents:**
- `V5_12_2_audit/V5.12.2_ALL_278_CALLS.md` - Complete listing
- `V5_12_2_audit/V5.12.2_CATEGORIZED_REVIEW.md` - Detailed categorization
- `src/scripts/audit_hyperbolic_norms.py` - AST scanner

---

## Mathematical Limits

### Hierarchy Ceiling: -0.8321

The Spearman correlation between 3-adic valuation and hyperbolic radius cannot exceed -0.8321 when ANY within-level variance exists.

**Mathematical proof:**

1. The 19,683 ternary operations distribute across valuation levels as:
   - v=0: 13,122 operations (66.7%)
   - v=1: 4,374 operations (22.2%)
   - v=2: 1,458 operations (7.4%)
   - ...
   - v=9: 1 operation (0.005%)

2. Spearman correlation assigns ranks to both variables and computes Pearson on ranks.

3. When v=0 contains 66.7% of samples, ANY variance within this level creates ties.

4. Tied ranks reduce the maximum achievable |ρ|.

5. Only `RadialSnapProjection` (hard snapping to exact radii per valuation level) achieves -1.0.

6. Hard snapping eliminates all richness (within-level variance = 0), which is a trivial solution.

**Implication:** The -0.8321 ceiling is not a bug but a fundamental property of the valuation distribution.

### Richness-Hierarchy Tradeoff

**Conventional wisdom:** You must sacrifice richness for hierarchy.

**Empirical disproof:** The `homeostatic_rich` checkpoint achieved:
- Hierarchy: -0.8321 (ceiling)
- Richness: 5.8x more than v5.11.8
- Richness: 28x more than max_hierarchy checkpoint

**Conclusion:** Hierarchy and richness are NOT mutually exclusive. The key is proper loss balancing:
```python
hierarchy_weight = 5.0
richness_weight = 2.0
separation_weight = 3.0
```

---

## V5.5 - Topological Foundation (Continuum Mesh)

**Path:** `checkpoints/v5_5/best.pt` | **Size:** 2.0 MB

V5.5 provides the **geometric substrate** for the entire Ternary VAE system. Despite pure Euclidean training (no hyperbolic components), it spontaneously develops p-adic-like geometry:

| Emergent Property | Value | Significance |
|-------------------|-------|--------------|
| Monotonic radial ordering | 10/10 levels | Perfect v=0 (outer) → v=9 (center) |
| Ultrametric compliance | 82.8% | p-adic metric signature |
| Hamming-Euclidean correlation | ρ = 0.55 | Algebraic structure preserved |
| Neighbor valuation consistency | 89.3% | Continuum mesh property |

**Architecture:** 9→256→128→64→16 (ReLU, Euclidean)

### The Continuum Mesh Property

Adjacent operations in the ternary operation space (differing by one in any digit) maintain consistent valuation relationships in the learned embedding. This means:

1. The p-adic hierarchy emerges **without explicit supervision**
2. The structure is intrinsic to the ternary operation space
3. Later versions freeze this encoder to preserve the topology

### Why This Matters

V5.5 proves that:
1. The genetic code's hierarchical structure is not imposed by our loss functions
2. It emerges naturally from the reconstruction objective
3. Hyperbolic projection (v5.11+) refines but does not create this structure

---

## Dual Manifold Organization Framework

### Two Valid Manifold Types

The project recognizes two valid organizations:

**Type 1: Valuation-Optimal (p-adic semantic)**
- Hierarchy: Negative (-0.8 to -1.0)
- Organization: Rare operations → center, frequent → edge
- Use for: Genetic code analysis, semantic reasoning, DDG prediction

**Type 2: Frequency-Optimal (Shannon information)**
- Hierarchy: Positive (+0.6 to +0.8)
- Organization: Frequent operations → center, rare → edge
- Use for: Compression, fast retrieval, similarity search

**Key insight:** v5_11_progressive (+0.78 hierarchy) is NOT broken - it's Shannon-optimal.

### Information-Theoretic Basis

**Kolmogorov complexity** (structural/semantic):
- Places fundamental/irreducible patterns at privileged positions
- Matches valuation-optimal organization

**Shannon entropy** (statistical):
- Allocates space proportional to frequency
- Matches frequency-optimal organization

Both are valid optimality criteria for different applications.

---

## Deprecated Module Warning

**`src/core/geometry_utils.py`** - DEPRECATED as of V5.12.2

- Use `src.geometry` instead (geoopt-backed implementations)
- Migration: `from src.geometry import poincare_distance, exp_map_zero`
- Will be archived in a future release

---

## Audit File Locations

All detailed audit documents are in `V5_12_2_audit/`:

| File | Contents |
|------|----------|
| `V5.12.2_HYPERBOLIC_AUDIT.md` | Main audit summary |
| `V5.12.2_ALL_278_CALLS.md` | Complete norm() call listing |
| `V5.12.2_DETAILED_AUDIT.md` | File-by-file analysis |
| `V5.12.2_CATEGORIZED_REVIEW.md` | Categorization by fix priority |
| `V5.12.2_AUDIT_FINAL_STATUS.md` | Current fix status |

---

## References

### P-adic Mathematics
- Koblitz, N. (1984). p-adic Numbers, p-adic Analysis, and Zeta-Functions
- Khrennikov, A. (2010). p-Adic Valued Distributions in Mathematical Physics
- Khrennikov (2004). Information Dynamics in Cognitive, Psychological, Social and Anomalous Phenomena

### Hyperbolic Geometry
- Nickel, M., & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations
- Ganea, O., et al. (2018). Hyperbolic Neural Networks
- Mathieu, E., et al. (2019). Continuous Hierarchical Representations with Poincare Variational Auto-Encoders

### Biological Background
- Crick, F. (1966). Codon-anticodon pairing: the wobble hypothesis
- Parisi (1987). Ultrametricity for Spin Glasses

---

*This document extracted from [CLAUDE.md](archive/CLAUDE_ORIGINAL.md) on 2026-01-30*
