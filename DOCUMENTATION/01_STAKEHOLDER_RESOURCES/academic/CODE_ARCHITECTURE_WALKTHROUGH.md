# Codebase Architecture Walkthrough (Due Diligence Guide)

> **Audience**: Technical Assessors, CTOs, Lead Engineers.
> **Context**: How to verify the claims in the Whitepaper within the actual code.

## 1. The Core Engine: `src/model/`

The heart of the system lies in `ternary_vae.py`.

- **Encoder**: Look for `TernaryEncoder` class. Note the use of `Softmax(dim=-1)` to enforce the probabilistic 3-adic representation before the latent sampling.
- **Decoder**: The `HyperbolicDecoder` uses a custom `PoincareLinear` layer (defined in `geometry.py`) to map the flat latent vector back to the codon space.

## 2. The Geometry Layer: `src/geometry/`

This module implements the non-Euclidean math.

- **`poincare.py`**: Contains the `add`, `log_map`, and `exp_map` functions for the Poincaré ball. Verify the stability epsilon ($\epsilon=1e-5$) used to prevent division by zero at the boundary.

## 3. The Meta-Controller: `src/meta/statenet.py`

This is the "AI watching the AI".

- **Input**: It takes a vector of `[entropy_A, entropy_B, kl_div, gradient_norm]`.
- **Output**: It emits `[delta_lr, delta_beta, delta_temp]`.
- **logic**: Check the `update()` step in the main training loop to see how these deltas are applied to the optimizer.

## 4. Running the Reproduction Script

To verify the "Goldilocks" coverage claim:

1.  Navigate to `experiments/mathematics/`.
2.  Run `python verify_coverage.py --dim 16`.
3.  Output will generate `coverage_heatmap.png`. Look for the "diagonal" activation pattern.

## 5. Viral Inference

To run the SARS-CoV-2 benchmarks:

1.  Navigate to `experiments/bioinformatics/sars_cov_2/`.
2.  Run `python score_variants.py --input omicron_ba5.fasta`.
3.  Compare the `likelihood_score` column with the known binding affinity data in `data/ground_truth.csv`.
