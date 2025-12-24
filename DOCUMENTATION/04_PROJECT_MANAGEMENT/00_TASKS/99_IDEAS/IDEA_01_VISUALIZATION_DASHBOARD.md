# Idea: Unified Interactive Visualization Dashboard

**Status:** Concept
**Driver:** Need for better observability of the 16D Manifold.

## Concept

Currently, we have dozens of standalone visualization scripts (`analyze_*.py`).
We should build a single **Streamlit** or **Dash** application that allows interactive exploration of the model.

## Features

- [ ] **Latent Space Explorer**: 3D scatter plot of the Poincare ball (using `plotly`).
- [ ] **Trajectory Replay**: Slider to watch 3-adic evolution over training epochs.
- [ ] **Hypothesis Testing**: Input a gene sequence, see its projection and neighbors.
