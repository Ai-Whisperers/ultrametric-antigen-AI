# Visualization Library Comparison & Implementation Guide

**Purpose**: Research report for professional visualization module implementation
**Branch**: `feature/visualization-refactor`
**Date**: 2025-12-25

---

## Executive Summary

This report compares Python visualization libraries and relevant GitHub repositories to inform the design of a professional, beautiful visualization module for the Ternary VAE project. The goal is to create publication-quality and presentation-ready figures with consistent styling, beautiful color palettes, and professional output.

---

## 1. Core Visualization Libraries Comparison

### 1.1 Matplotlib (Foundation)

| Aspect | Details |
|:-------|:--------|
| **Best For** | Static plots, full customization, animations |
| **Interactivity** | Limited (requires additional backends) |
| **Learning Curve** | Moderate |
| **Publication Quality** | Excellent with proper configuration |
| **3D Support** | Built-in via `mpl_toolkits.mplot3d` |

**Key Strengths**:
- Most widely-used Python plotting library (released 2003)
- Foundation for Seaborn, pandas plotting, and many others
- Extremely customizable - control every element
- Strong integration with NumPy and scientific stack
- Native animation support

**Key Patterns to Adopt**:
```python
# Publication-ready configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
```

### 1.2 Seaborn (Statistical Visualization)

| Aspect | Details |
|:-------|:--------|
| **Best For** | Statistical plots, attractive defaults |
| **Interactivity** | Static only |
| **Learning Curve** | Easy |
| **Publication Quality** | Very good out-of-box |
| **Pandas Integration** | Native |

**Key Strengths** (v0.14, 2024-2025):
- One-function-per-plot API for streamlined statistical visualization
- Native Polars DataFrame support
- Auto-generates uncertainty intervals
- Pre-defined themes: `whitegrid`, `darkgrid`, `ticks`, `white`
- Built-in colorblind-safe palettes

**Key Patterns to Adopt**:
```python
# Publication theme setup
sns.set_theme(style='ticks', context='paper')
sns.set_palette('colorblind')  # Accessibility
sns.despine()  # Clean appearance
```

### 1.3 Plotly (Interactive Web)

| Aspect | Details |
|:-------|:--------|
| **Best For** | Interactive web-based visualizations |
| **Interactivity** | Excellent (zoom, pan, tooltips, animations) |
| **Learning Curve** | Easy |
| **3D Support** | Excellent, GPU-accelerated |
| **Export** | HTML, PNG, SVG, PDF |

**Key Strengths** (v7.0, 2025):
- GPU-accelerated rendering (new in 7.0)
- Native DuckDB connectors
- Interactive 3D charts perfect for manifold exploration
- Dash integration for full web apps
- Built-in animation support

**Key Patterns to Adopt**:
```python
import plotly.graph_objects as go

# Interactive 3D manifold
fig = go.Figure(data=[go.Surface(z=manifold_data)])
fig.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    template='plotly_white'
)
```

### 1.4 Altair (Declarative Grammar)

| Aspect | Details |
|:-------|:--------|
| **Best For** | Declarative, reproducible visualizations |
| **Interactivity** | Good (via Vega-Lite) |
| **Learning Curve** | Moderate |
| **Data Handling** | Excellent pandas integration |

**Key Strengths**:
- Declarative grammar of graphics (like ggplot2)
- Automatic encoding of data types
- Built-in interactivity
- Concise, readable code

### 1.5 Bokeh (Web Dashboards)

| Aspect | Details |
|:-------|:--------|
| **Best For** | Large datasets, streaming data |
| **Interactivity** | Excellent |
| **Server Support** | Built-in Bokeh Server |
| **3D Support** | Limited |

---

## 2. Specialized Libraries for Scientific Visualization

### 2.1 SciencePlots (Publication Styles)

**GitHub**: [garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots)

**What It Offers**:
- Pre-configured styles for Nature, IEEE, Science journals
- Paul Tol's colorblind-safe discrete rainbow palettes
- LaTeX-quality typography
- Easy style switching

**Key Implementation**:
```python
import scienceplots
plt.style.use(['science', 'ieee'])  # IEEE journal format
plt.style.use(['science', 'nature'])  # Nature format
```

**What We Can Learn**:
- Journal-specific figure dimensions and fonts
- Systematic style management via `.mplstyle` files
- Color cycles designed for accessibility

### 2.2 Scientific Visualization Book

**GitHub**: [rougier/scientific-visualization-book](https://github.com/rougier/scientific-visualization-book)

**What It Offers**:
- Complete open-access book on matplotlib
- Covers coordinate systems, projections, typography
- 3D figures, optimization, animation
- Extensive showcase gallery

**What We Can Learn**:
- Figure design principles
- Matplotlib architecture deep-dive
- Advanced techniques (custom projections, complex annotations)

### 2.3 PyPalettes (Color Management)

**What It Offers**:
- ~2,500 unique named palettes
- Works with matplotlib, Plotly, seaborn, altair
- Systematic palette browsing

**Key Implementation**:
```python
from pypalettes import load_cmap
cmap = load_cmap("Sunset")  # Named palette
```

### 2.4 CMasher (Scientific Colormaps)

**What It Offers**:
- Perceptually uniform colormaps
- Colorblind-friendly
- Readable in black & white print
- Sequential, diverging, and cyclic options

---

## 3. Hyperbolic & Manifold Visualization Libraries

### 3.1 hyperbolic (Poincare Disk)

**GitHub**: [cduck/hyperbolic](https://github.com/cduck/hyperbolic)

**What It Offers**:
- Python library for hyperbolic geometry
- Poincare disk and half-plane models
- SVG rendering via drawsvg

**Relevance**: Direct use for our Poincare ball visualizations.

### 3.2 hyperbolic-learning

**GitHub**: [drewwilimitis/hyperbolic-learning](https://github.com/drewwilimitis/hyperbolic-learning)

**What It Offers**:
- Hyperbolic MDS, K-Means, SVM implementations
- Embedding visualizations
- Distance computations in hyperbolic space

**What We Can Learn**:
- Hyperbolic distance visualization techniques
- Geodesic path rendering
- Curvature-aware projections

### 3.3 Geomstats

**What It Offers**:
- Riemannian geometry computations
- `PoincareBallMetric` class
- Exponential/logarithmic map implementations
- Graph embedding in hyperbolic space

**Relevance**: Mathematical foundation for hyperbolic VAE visualizations.

---

## 4. VAE & Latent Space Visualization

### 4.1 VAE Explainer

**Link**: [arxiv.org/html/2409.09011v1](https://arxiv.org/html/2409.09011v1)

**What It Offers**:
- Browser-based interactive VAE visualization
- Latent space hovering with live reconstruction
- Interpolation visualization
- Open-source implementation

**What We Can Learn**:
- Interactive latent space exploration UX
- Smooth interpolation rendering
- Real-time reconstruction display

### 4.2 VAE-Latent-Space-Explorer

**GitHub**: [tayden/VAE-Latent-Space-Explorer](https://github.com/tayden/VAE-Latent-Space-Explorer)

**What It Offers**:
- React + TensorFlow.js browser implementation
- WebGL-accelerated generation
- Interactive latent vector manipulation

**What We Can Learn**:
- Web-exportable visualization formats
- Client-side inference for demos

### 4.3 latent-musicvis

**GitHub**: [lyramakesmusic/latent-musicvis](https://github.com/lyramakesmusic/latent-musicvis)

**What It Offers**:
- 3D UMAP projection of latent vectors
- Audio playback sync with points
- Latent resynthesis

**What We Can Learn**:
- 3D point cloud with metadata overlay
- Synchronized data exploration

---

## 5. Bioinformatics-Specific Visualization

### 5.1 ProS2Vi (Protein Secondary Structure)

**What It Offers**:
- DSSP-based secondary structure visualization
- Flask GUI with Biopython backend
- PDF export via wkhtmltopdf

### 5.2 Melodia (Protein Shape Analysis)

**What It Offers**:
- Differential geometry of 3D protein curves
- Knot theory integration
- nglview Jupyter widget support

### 5.3 PyMOL / nglview

**What They Offer**:
- 3D molecular structure visualization
- Python scriptable
- Jupyter notebook integration (nglview)

---

## 6. Best Practices Summary

### 6.1 Color Selection

| Type | Recommendation |
|:-----|:---------------|
| **Sequential** | viridis, plasma, magma, inferno |
| **Diverging** | coolwarm, RdBu (centered at meaningful value) |
| **Categorical** | Paul Tol palettes, colorblind-safe |
| **Avoid** | Rainbow/jet (perceptual non-uniformity) |
| **Print-Safe** | Test in grayscale |

### 6.2 Figure Sizing

```python
# Set figure size to final output size
# This ensures fonts scale correctly
COLUMN_WIDTH = 3.5  # inches (single column)
DOUBLE_WIDTH = 7.0  # inches (full page)
ASPECT_RATIO = 0.618  # Golden ratio

fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * ASPECT_RATIO))
```

### 6.3 Typography

- **Nature/Science**: Sans-serif (Helvetica, Arial)
- **IEEE**: Sans-serif, specific point sizes
- **Presentations**: Larger fonts (14-18pt titles)

### 6.4 Export Formats

| Format | Use Case |
|:-------|:---------|
| **PNG** | Web, presentations (300 DPI for print) |
| **SVG** | Scalable web graphics |
| **PDF** | Publications, vector quality |
| **HTML** | Interactive Plotly figures |

---

## 7. Recommended Architecture for src/visualization/

Based on this research, here's the proposed module structure:

```
src/visualization/
├── __init__.py          # Public API exports
├── config.py            # Global configuration, constants
├── styles/
│   ├── __init__.py
│   ├── themes.py        # Theme definitions (scientific, pitch, dark)
│   ├── palettes.py      # Color palettes (risk, safety, semantic)
│   └── presets.py       # Pre-configured style contexts
├── core/
│   ├── __init__.py
│   ├── base.py          # Base figure/axes classes
│   ├── annotations.py   # Significance bars, labels, legends
│   └── export.py        # Multi-format export functions
├── plots/
│   ├── __init__.py
│   ├── manifold.py      # Poincare ball, latent space
│   ├── surfaces.py      # 3D surfaces, loss landscapes
│   ├── distributions.py # Histograms, KDE, violin plots
│   ├── heatmaps.py      # Correlation, density maps
│   └── training.py      # Loss curves, metrics over time
├── interactive/
│   ├── __init__.py
│   └── plotly_utils.py  # Plotly-based interactive figures
└── projections/
    ├── __init__.py
    ├── poincare.py      # Hyperbolic disk projections
    ├── calabi_yau.py    # Calabi-Yau surface rendering
    └── hopf.py          # Hopf fibration visualization
```

---

## 8. Implementation Priorities

### Phase 1: Foundation (Essential)
1. **styles/themes.py**: Scientific + pitch themes
2. **styles/palettes.py**: Color palettes (extend existing PALETTE dict)
3. **core/base.py**: Figure creation with automatic styling
4. **core/export.py**: Multi-format export (PNG, SVG, PDF)

### Phase 2: Core Plots
1. **plots/manifold.py**: Latent space visualizations
2. **plots/surfaces.py**: 3D loss landscapes
3. **plots/training.py**: Training progress plots

### Phase 3: Advanced
1. **projections/poincare.py**: Hyperbolic geometry
2. **interactive/plotly_utils.py**: Web-ready interactive plots
3. **projections/calabi_yau.py**: Mathematical surface rendering

---

## 9. Key Takeaways

### What to Adopt

1. **From SciencePlots**: Style management via `.mplstyle` files
2. **From Seaborn**: Context-aware themes (`paper`, `poster`, `talk`)
3. **From Plotly**: Interactive 3D for manifold exploration
4. **From PyPalettes**: Systematic color palette management
5. **From hyperbolic-learning**: Hyperbolic distance visualizations
6. **From VAE Explainer**: Latent space interaction patterns

### Libraries to Install

```bash
# Core
pip install matplotlib seaborn plotly

# Scientific styling
pip install scienceplots pypalettes cmocean

# Hyperbolic geometry
pip install geomstats

# Interactive notebooks
pip install nglview ipywidgets
```

---

## Sources

- [Top 10 Python Data Visualization Libraries 2025](https://reflex.dev/blog/2025-01-27-top-10-data-visualization-libraries/)
- [SciencePlots GitHub](https://github.com/garrettj403/SciencePlots)
- [Scientific Visualization Book](https://github.com/rougier/scientific-visualization-book)
- [PyPalettes Blog](https://blog.scientific-python.org/matplotlib/pypalettes/)
- [hyperbolic-learning GitHub](https://github.com/drewwilimitis/hyperbolic-learning)
- [Geomstats Documentation](https://geomstats.github.io/)
- [VAE Explainer Paper](https://arxiv.org/html/2409.09011v1)
- [Making Publication-Quality Figures in Python](https://towardsdatascience.com/making-publication-quality-figures-in-python-part-i-fig-and-axes-d86c3903ad9b/)
- [Tutorial: Publication-Ready Figures](https://github.com/ICWallis/tutorial-publication-ready-figures)
