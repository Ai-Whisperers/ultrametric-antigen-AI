# Ternary VAE Project - Claude Context

**Doc-Type:** Project Configuration · Version 1.6 · Updated 2026-01-02 · AI Whisperers

---

## Project Overview

This repository implements a Variational Autoencoder for learning 3-adic (p-adic) hierarchical structure over ternary operations. The model embeds 19,683 ternary operations (3^9) into a hyperbolic Poincaré ball where radial position encodes 3-adic valuation.

---

## V5.12.2 HYPERBOLIC AUDIT - NOT PRODUCTION READY

**Status:** Architecture migration in progress. Do NOT use for production training.

**Issue:** Many files in `src/`, `configs/`, and `scripts/` incorrectly use Euclidean `.norm()` on hyperbolic Poincaré ball embeddings instead of `poincare_distance()`. This causes:
- Incorrect radial hierarchy computation (coverage stuck at ~20%)
- Metric correlations computed in wrong geometry
- Training scripts producing misleading results

**Audit Document:** `DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/V5_12_2_HYPERBOLIC_AUDIT.md`

**Correct Pattern:**
```python
# WRONG - Euclidean norm on hyperbolic embeddings
radius = torch.norm(z_hyp, dim=-1)

# CORRECT - Hyperbolic distance from origin
from src.geometry import poincare_distance
origin = torch.zeros_like(z_hyp)
radius = poincare_distance(z_hyp, origin, c=curvature)
```

**Fixed (ALL CORE FILES COMPLETE):**
- HIGH: `src/api/cli/train.py`, `src/encoders/holographic_encoder.py`, `src/losses/consequence_predictor.py`
- MEDIUM: `src/analysis/crispr/embedder.py`, `src/analysis/evolution.py`, `src/analysis/hiv/analyze_all_datasets.py`, `src/geometry/holographic_poincare.py`, `src/encoders/ptm_encoder.py`
- LOW: `src/visualization/plots/manifold.py`, `src/training/monitoring/tensorboard_logger.py`

**Remaining (LOWEST - research scripts only):**
- ~40 research scripts in `src/research/` use `radii = norm()` patterns

---

## Deprecated Modules

**`src/core/geometry_utils.py`** - DEPRECATED as of V5.12.2
- Use `src.geometry` instead (geoopt-backed implementations)
- Migration: `from src.geometry import poincare_distance, exp_map_zero`
- Will be archived in a future release

---

## Architecture Components

### Dual-Encoder System (v5.11+)

The TernaryVAEV5_11 architecture uses two complementary encoders:

**VAE-A (Coverage Encoder)**
- Primary role: Reconstruct all 19,683 operations (coverage)
- Behavior: Often learns frequency-based ordering (can show inverted/positive hierarchy)
- Freezing: Freeze to preserve coverage while training VAE-B for hierarchy

**VAE-B (Hierarchy Encoder)**
- Primary role: Learn correct p-adic radial ordering
- Behavior: Should achieve negative Spearman correlation (v0 at edge, v9 at center)
- Training: Benefits from slower learning rate (encoder_b_lr_scale=0.1)

**DifferentiableController**
- Architecture: 8→32→32→6 MLP
- Outputs: rho, weight_geodesic, beta_A, beta_B, tau (loss weights)
- Usage: `use_controller=True` in model config

**HomeostasisController**
- Role: Training orchestrator for freeze/unfreeze decisions
- Triggers: Based on coverage thresholds and hierarchy plateau detection
- Q-metric: `Q = dist_corr + 1.5 × |hierarchy|`

**V5.12.4 Improved Encoder/Decoder** - The `ImprovedEncoder` and `ImprovedDecoder` classes (`src/models/improved_components.py`) replace ReLU with SiLU activation for smoother gradients, add LayerNorm for stable training, include Dropout (default 0.1) for regularization, and clamp logvar to [-10, 2] to prevent KL collapse/explosion. These components can load v5.5 checkpoint weights (Linear layers only) with fresh LayerNorm initialization, enabling backwards-compatible upgrades. Enable via `encoder_type: improved` and `decoder_type: improved` in config, or see `configs/v5_12_4.yaml` for full example.

---

## Key Metrics

**Coverage**: Percentage of 19,683 operations correctly reconstructed (target: 100%)

**Hierarchy**: Spearman correlation between 3-adic valuation and radius
- Negative = correct (v0 at outer edge, v9 at center)
- Target: -0.83 to -1.0
- Ceiling: -0.8321 with any within-level variance (mathematical limit due to v=0 having 66.7% of samples)

**Richness**: Average within-valuation-level variance of radii
- Measures geometric diversity beyond ordering
- Higher = more meaningful structure preserved
- Zero = collapsed to trivial shells (bad)

---

## v5.11 Checkpoint Reference

### CRITICAL WARNING: v5_11_overnight

**DO NOT USE v5_11_overnight as a reference for good training.**

Despite appearing to have good metrics (100% coverage, -0.83 hierarchy), this checkpoint is problematic:
- Training collapsed during the run
- Coverage came purely from frozen checkpoint initialization
- The hierarchy correlation is an artifact, not learned structure
- Richness appears high (0.0085) but is not from genuine training

---

### Validated Checkpoints

| Version | Checkpoint | Coverage | Hier_B | Richness | Status |
|---------|------------|----------|--------|----------|--------|
| **v5.11.3** | v5_11_structural | 100% | -0.40 | 0.00304 | Moderate hierarchy, good richness |
| **v5.11.8** | v5_11_homeostasis | 99.9% | -0.82 | 0.00136 | Good hierarchy, moderate richness |
| **homeostatic_rich** | homeostatic_rich | 100% | -0.8321 | 0.00787 | **BEST BALANCE** - ceiling hier + high richness |
| **v5_11_progressive** | v5_11_progressive | 99.9% | +0.78 (B) | 0.00145 | VAE-B inverted, not usable |

---

### Detailed Version Notes

**v5.11.3 (v5_11_structural)**
- Architecture: hidden_dim=128, n_projection_layers=2
- Stored metrics: radial_corr_A=-0.730, radial_corr_B=-0.832
- Actual evaluation: hier_A=-0.69, hier_B=-0.74
- Use case: Bioinformatics codon encoder training (stable embeddings)

**v5.11.8 (v5_11_homeostasis)**
- Architecture: hidden_dim=64, use_controller=True
- Uses HomeostasisController for dynamic freeze management
- Stored metrics: radial_corr_A=-0.816, radial_corr_B=-0.832, radius_v0=0.932, radius_v9=0.013
- Actual evaluation: hier_A=-0.51, hier_B=-0.83
- Best radial separation (v0→v9 spread)

**homeostatic_rich (RECOMMENDED)**
- Training script: `scripts/epsilon_vae/train_homeostatic_rich.py`
- Loss weights: hierarchy=5.0, coverage=1.0, richness=2.0, separation=3.0
- Achieved: -0.8321 hierarchy (ceiling) with 5.8x more richness than v5.11.8
- Proves hierarchy and richness are NOT mutually exclusive

**v5_11_progressive**
- WARNING: VAE-B shows inverted hierarchy (+0.78)
- VAE-A shows slight negative (-0.04)
- Not suitable for p-adic applications

---

## Training Parameters for Hierarchy

To prioritize hierarchy while preserving richness:

```python
# Loss weights (from homeostatic_rich)
hierarchy_weight = 5.0      # Push toward target radii per valuation
coverage_weight = 1.0       # Maintain reconstruction
richness_weight = 2.0       # Preserve within-level variance
separation_weight = 3.0     # Ensure level ordering

# Freeze strategy
freeze_encoder_a = True     # Preserve coverage
freeze_encoder_b = False    # Allow B to learn hierarchy
encoder_b_lr_scale = 0.1    # Slower adaptation preserves structure

# Variance control
variance_weight = 50-100    # Higher = more collapse (less richness)
min_richness_ratio = 0.5    # Keep at least 50% of original variance
```

---

## Mathematical Limits

**Hierarchy Ceiling: -0.8321**

The Spearman correlation cannot exceed -0.8321 when ANY within-level variance exists. This is due to:
- v=0 contains 66.7% of all samples (13,122 of 19,683)
- Even tiny variance in this large group creates ties
- Only RadialSnapProjection (hard snapping to exact radii) achieves -1.0
- Hard snapping eliminates all richness (trivial solution)

**Richness-Hierarchy Tradeoff**

NOT mutually exclusive. homeostatic_rich proved:
- Same ceiling hierarchy (-0.8321) as collapsed models
- 5.8x more richness than v5.11.8
- 28x more richness than max_hierarchy

---

## File Locations

**Training Scripts**
- `scripts/epsilon_vae/train_homeostatic_rich.py` - Best balance approach
- `scripts/epsilon_vae/train_hierarchy_focused.py` - Hierarchy-first approach
- `scripts/epsilon_vae/analyze_all_checkpoints.py` - Checkpoint comparison tool

**Model Definitions**
- `src/models/ternary_vae.py` - TernaryVAEV5_11_PartialFreeze
- `src/models/homeostasis.py` - HomeostasisController, compute_Q

**Checkpoints**
- `sandbox-training/checkpoints/` - All training runs
- Production: Use `homeostatic_rich/best.pt` or `v5_11_homeostasis/best.pt`

---

## P-adic Codon Encoder Research

Research scripts validating p-adic embeddings against physical ground truth are in `research/codon-encoder/`.

### Key Discoveries

| Invariant | Finding | Correlation |
|-----------|---------|-------------|
| Dim 13 | "Physics dimension" - encodes mass, volume, force constants | ρ = -0.695 |
| Radial structure | Encodes amino acid mass | ρ = +0.760 |
| Force constant | `k = radius × mass / 100` | **ρ = 0.860** |
| Vibrational freq | `ω = √(k/m)` derivable | ρ = 1.000 |

### Directory Structure

```
research/codon-encoder/
├── config.py           # Centralized paths and shared data
├── benchmarks/         # Validation benchmarks
│   ├── mass_vs_property_benchmark.py
│   ├── kinetics_benchmark.py
│   ├── deep_physics_benchmark.py
│   └── ddg_benchmark.py
├── training/           # Model training
│   ├── ddg_predictor_training.py      # sklearn
│   └── ddg_pytorch_training.py        # PyTorch hyperparameter search
├── analysis/           # Embedding analysis
│   ├── proteingym_pipeline.py         # Dimension correlations
│   └── padic_dynamics_predictor.py    # Force constant prediction
├── pipelines/          # Integration pipelines
│   ├── padic_3d_dynamics.py           # 3D structure + dynamics
│   ├── af3_pipeline.py                # AlphaFold3
│   └── ptm_mapping.py                 # PTM analysis
└── results/            # Output directories
```

### Usage

```bash
cd research/codon-encoder
python benchmarks/deep_physics_benchmark.py
python training/ddg_pytorch_training.py
python analysis/proteingym_pipeline.py
```

### Key Results

- **Thermodynamics (ΔΔG)**: Mass-based features win (padic_mass ρ=0.94)
- **Kinetics (folding)**: Property-based features win (property ρ=0.94)
- **Physics Levels**: P-adic encodes force constants (Level 3) but NOT B-factors (Level 4)

---

## Quick Evaluation

```python
from src.models import TernaryVAEV5_11_PartialFreeze
from src.core import TERNARY
from src.geometry import poincare_distance

model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16, hidden_dim=64, max_radius=0.99,
    curvature=1.0, use_controller=True, use_dual_projection=True
)
model.load_state_dict(torch.load('path/to/best.pt')['model_state_dict'])

# Get embeddings
out = model(ops, compute_control=False)

# V5.12.2: Use hyperbolic distance for radii (NOT .norm())
origin_A = torch.zeros_like(out['z_A_hyp'])
origin_B = torch.zeros_like(out['z_B_hyp'])
radii_A = poincare_distance(out['z_A_hyp'], origin_A, c=1.0)
radii_B = poincare_distance(out['z_B_hyp'], origin_B, c=1.0)

# Compute hierarchy
from scipy.stats import spearmanr
valuations = TERNARY.valuation(indices)
hierarchy = spearmanr(valuations.cpu(), radii_B.cpu())[0]  # Use VAE-B for p-adic
```

---

## V5.12.2 COMPLETE FIX LIST

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
- `V5.12.2_ALL_278_CALLS.md` - Complete listing
- `V5.12.2_CATEGORIZED_REVIEW.md` - Detailed categorization
- `scripts/audit_hyperbolic_norms.py` - AST scanner

---

## Partner Packages (deliverables/partners/)

Consolidated structure for CONACYT and stakeholder deliverables:

| Partner | Focus | Scripts | Results | src.core Integration |
|---------|-------|---------|---------|---------------------|
| **jose_colbes** | Protein stability (DDG, Rosetta-blind) | C1-C4 | validation/, rosetta_blind/ | padic_math ✓ |
| **carlos_brizuela** | AMP design (NSGA-II optimization) | B1, B8, B10 | pareto, microbiome, pathogen | VAE interface |
| **alejandra_rojas** | Arbovirus (DENV, ZIKV primers) | A2, trajectory, scanner | pan_arbovirus_primers/ | padic_math ✓ |
| **hiv_research_package** | Drug resistance, LA selection | H6, H7 | tdr_screening/, la_selection/ | Stanford HIVdb |

---

## Remaining Tasks (Next Dev Session)

| Priority | Task | Category | Details |
|:--------:|------|----------|---------|
| **1** | Sync stakeholder-portfolio branch | Git | Cherry-pick/merge organized main to stakeholder-portfolio |
| **2** | Create HIV deep dive document | Docs | Missing from DOCUMENTATION tier 3 (01, 02, 04 gap) |
| **3** | Fix tier numbering | Docs | Renumber 04 → 03 or add missing 03_HIV_DEEP_DIVE.md |
| **4** | V5.12.2 research script fixes | Code | ~40 files in src/research/ use Euclidean norm() |
| **5** | Partner README updates | Docs | Add usage examples, benchmark results to each partner |
| **6** | Stakeholder portfolio curation | Git | Remove dev artifacts from stakeholder-portfolio branch |
| **7** | Publication figures organization | Assets | Move deliverables/results/figures/ to appropriate location |

### Completed This Session (2026-01-02)

- Consolidated all partner results to respective folders
- Refactored Colbes scripts (C1, C4, scoring.py) → src.core.padic_math
- Refactored Rojas scripts (trajectory, scanner) → src.core.padic_math
- Pushed 4 commits to main: Colbes, Brizuela, Rojas, HIV consolidation

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-02 | 1.6 | Added Partner Packages table, Remaining Tasks section, session summary |
| 2025-12-30 | 1.5 | V5.12.4 ImprovedEncoder/Decoder with SiLU, LayerNorm, Dropout, logvar clamping |
| 2025-12-29 | 1.4 | V5.12.2 COMPLETE FIX LIST - all 75 files needing fixes documented |
| 2025-12-29 | 1.3 | V5.12.2 audit COMPLETE - all core files fixed, deprecated geometry_utils.py |
| 2025-12-29 | 1.2 | V5.12.2 hyperbolic audit warning, fixed Quick Evaluation example |
| 2025-12-28 | 1.1 | Added codon-encoder research section with key discoveries |
| 2025-12-27 | 1.0 | Initial CLAUDE.md with architecture docs, checkpoint reference, overnight warning |

