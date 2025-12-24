# RA Visualization Implementation Plan

**Doc-Type:** Implementation Tracker · Version 1.1 · Updated 2025-12-16 · Author Claude

---

## Directory Structure

```
visualizations/
├── TODO.md                          # This file
├── generate_all.py                  # Master generator script
├── utils/                           # Shared utilities
│   ├── __init__.py
│   ├── plotting.py                  # Styling, palettes, export
│   └── data_loader.py               # Data loading helpers
├── pitch/                           # Non-scientific visualizations
│   ├── 01_pathophysiology_funnel/
│   ├── 02_hla_risk_charts/
│   ├── 03_intervention_pathways/
│   ├── 04_safety_comparisons/
│   └── 05_goldilocks_radar/
└── scientific/                      # Advanced technical visualizations
    ├── 01_hla_pca_projections/
    ├── 02_cluster_boundary_3d/
    ├── 03_codon_embedding_space/
    ├── 04_calabi_yau_manifolds/
    ├── 05_distance_heatmaps/
    └── 06_position_discriminative_maps/
```

---

## Implementation Status

### Phase 1: Infrastructure & Pitch Visualizations

| ID | Task | Status | Output Files |
|:---|:-----|:-------|:-------------|
| P1.1 | Create directory structure | DONE | directories created |
| P1.2 | Pathophysiology funnel (6-stage cascade) | DONE | pathophysiology_funnel.png/svg |
| P1.3 | HLA risk stratification charts | DONE | hla_risk_stratification.png/svg, hla_position_importance.png/svg |
| P1.4 | Intervention pathway diagrams (3-tier) | DONE | intervention_pathway.png/svg, pathway_distances.png/svg |
| P1.5 | Codon safety comparison bars | DONE | safety_comparison.png/svg, safety_summary.png/svg |
| P1.6 | Goldilocks zone radar/gauge | DONE | goldilocks_gauge.png/svg, epitope_shifts.png/svg, goldilocks_statistics.png/svg |

### Phase 2: Scientific Visualizations - 2D

| ID | Task | Status | Output Files |
|:---|:-----|:-------|:-------------|
| S2.1 | HLA PCA projections with risk contours | DONE | hla_pca_projection.png/svg, hla_risk_contours.png/svg |
| S2.2 | Distance matrix heatmaps with JS divergence | DONE | epitope_distance_matrix.png/svg, statistical_comparison.png/svg |
| S2.3 | Position discriminative power maps (84-codon) | TODO | - |

### Phase 3: Scientific Visualizations - 3D Interactive

| ID | Task | Status | Output Files |
|:---|:-----|:-------|:-------------|
| S3.1 | 21-cluster boundary surfaces (plotly) | DONE | cluster_boundary_3d.html |
| S3.2 | Codon embedding t-SNE/UMAP projections | TODO | - |
| S3.3 | Citrullination transition vectors | DONE | (included in S3.1) |

### Phase 4: Advanced Manifold Projections

| ID | Task | Status | Output Files |
|:---|:-----|:-------|:-------------|
| S4.1 | Regenerative axis 3D pathway geometry | DONE | pathway_manifold_distances.html |
| S4.2 | Calabi-Yau projection utilities | DONE | calabi_yau_projection.html |
| S4.3 | 16D→3D manifold embedding visualization | DONE | (included in S4.2) |

### Phase 5: Integration & Export

| ID | Task | Status | Output Files |
|:---|:-----|:-------|:-------------|
| I5.1 | Shared plotting utilities module | DONE | utils/plotting.py, utils/data_loader.py |
| I5.2 | HTML export for interactive plots | DONE | 3 HTML files |
| I5.3 | PDF/SVG export for publications | DONE | all .svg files |

---

## Generated Files Summary

### Pitch Visualizations (11 PNG + 11 SVG)

| File | Description | Audience |
|:-----|:------------|:---------|
| `pathophysiology_funnel.png` | 6-stage disease cascade | General |
| `hla_risk_stratification.png` | Allele odds ratio comparison | Labs |
| `hla_position_importance.png` | Position 65 vs classical SE | Labs |
| `intervention_pathway.png` | 3-tier therapeutic protocol | Clinicians |
| `pathway_distances.png` | Autonomic geometry bars | Clinicians |
| `safety_comparison.png` | Codon optimization safety | Biotech |
| `safety_summary.png` | Visual safety profile | Biotech |
| `goldilocks_gauge.png` | Immune recognition zones | General |
| `epitope_shifts.png` | Perturbation magnitudes | Labs |
| `goldilocks_statistics.png` | Statistical evidence summary | Labs |

### Scientific Visualizations (4 PNG + 4 SVG + 3 HTML)

| File | Description | Interactivity |
|:-----|:------------|:--------------|
| `hla_pca_projection.png` | PCA with risk categories | Static |
| `hla_risk_contours.png` | Risk landscape contours | Static |
| `epitope_distance_matrix.png` | Distance heatmap | Static |
| `statistical_comparison.png` | Multi-metric comparison | Static |
| `cluster_boundary_3d.html` | 3D cluster structure | Interactive (plotly) |
| `calabi_yau_projection.html` | CY manifold embedding | Interactive (plotly) |
| `pathway_manifold_distances.html` | Pathway geometry | Interactive (plotly) |

---

## Usage

**Generate all visualizations:**
```bash
cd research/bioinformatics/rheumatoid_arthritis/visualizations
python generate_all.py
```

**Generate specific category:**
```bash
python pitch/01_pathophysiology_funnel/generate.py
python scientific/04_calabi_yau_manifolds/generate.py
```

**View interactive visualizations:**
Open `.html` files in any modern browser. Supports rotation, zoom, and hover tooltips.

---

## Technical Notes

**Dependencies:**
- matplotlib >= 3.5
- numpy >= 1.21
- seaborn >= 0.11
- plotly >= 5.0 (for interactive 3D)
- scikit-learn >= 1.0 (for PCA)

**Optional:**
- kaleido (for static PNG export from plotly)

**Font Warnings:**
Some Unicode characters (subscripts, checkmarks) may show warnings on systems without full font support. Visual output is unaffected.

---

## Remaining Work

| Priority | Task | Estimated Effort |
|:---------|:-----|:-----------------|
| P1 | Position discriminative power maps (S2.3) | 2h |
| P1 | Codon embedding t-SNE/UMAP (S3.2) | 3h |
| P2 | Install kaleido for static 3D exports | 10min |
| P2 | Add animation to CY manifold rotation | 2h |

---

## Changelog

| Date | Version | Changes |
|:-----|:--------|:--------|
| 2025-12-16 | 1.1 | Phase 1-4 complete, 25 visualization files generated |
| 2025-12-16 | 1.0 | Initial plan created |
