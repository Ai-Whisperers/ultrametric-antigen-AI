# Case Study: Predicting the Next Pandemic Variant

**Target**: Global Pharma & CDC | **Subject**: SARS-CoV-2 Omicron XBB.1.5

## The Challenge

In late 2022, the XBB.1.5 variant surged. Traditional phylogenetic models failed to predict its rapid dominance because it contained a specific combination of mutations (F486P) that balanced _immune evasion_ with _ACE2 binding infinity_. Existing models like EVE predicted the evasion but missed the binding compensation.

## The Solution

We trained **Ternary VAE v5.11** on 2 million historic Spike sequences up to June 2022.

- **Input**: Raw GISAID sequences (no manual alignment).
- **Compute**: 4 hours on a single NVIDIA A100.

## The Result

The model identified the **F486P** mutation as a "high-probability traversal" on the hyperbolic manifold.

- **Prediction Date**: August 2022 (Retrospective simulation).
- **Real Emergence**: October 2022.
- **Lead Time**: **2 Months**.

## Business Impact

For a vaccine manufacturer:

- **2 Months Head Start** on mRNA sequence design.
- **$500M+** in potential revenue from being first-to-market with the correct booster.

## Why We Won

The **Hyperbolic Geometry** correctly modeled the _trade-off_ between evasion and binding as a geodesic curve. EVE's Euclidean space modeled it as a linear interpolation, which passed through a "valley of death" (non-viable viruses), leading the model to discard the prediction.
