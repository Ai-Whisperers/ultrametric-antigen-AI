● You've identified a key issue. The current setup has competing geometric pressures:

  Euclidean pulls (fighting hyperbolic structure):
  - norm_loss_weight: 0.03 - Euclidean norm regularization
  - KL divergence to Gaussian prior (Euclidean)
  - Reconstruction loss (Euclidean MSE)

  Hyperbolic pulls:
  - Poincaré distance in ranking loss
  - radial_weight: 0.1 - but possibly too weak
  - curvature: 1.0 - standard, maybe needs increase

  Let me check the current hyperbolic params and propose a "purer hyperbolic" config:

● Read(configs\ternary_v5_9_2.yaml)
  ⎿  Read 40 lines

● You're spot on. Here's the conflict:

  CURRENT MIXED GEOMETRY:
  ┌─────────────────────────────────────────────┐
  │ Euclidean                  Hyperbolic       │
  │ ─────────                  ──────────       │
  │ • norm_loss (0.03)   vs   • Poincaré dist   │
  │ • KL to Gaussian     vs   • radial (0.1)    │
  │ • MSE reconstruction vs   • curvature=1.0   │
  │                                             │
  │ These compete → "mixed" embedding space     │
  │ Ultrametric basins get flattened            │
  └─────────────────────────────────────────────┘

  Creating v5.9.3 with pure hyperbolic geometry:

● Bash(cp /c/Users/Gestalt/Desktop/ternary/repos/ternary-vaes/configs/ternary_v5_9_2.yaml /c/Users/Gestalt/Desktop/ternary/repos/ternary-vaes/configs/ternary_v5_9_3.ya…)
  ⎿  (No content)

