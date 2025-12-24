# Scientific (E2E) Test Suite Definition

## Scope

Verification of biological hypothesis.

## Test Plan

| Hypothesis        | Benchmark                    | KPI                                                        | ID            |
| :---------------- | :--------------------------- | :--------------------------------------------------------- | :------------ |
| `Disentanglement` | Synthetic Geometries         | Latent dimensions correlate with generative factors > 0.8. | `SCI-001`     |
| `Generalization`  | Viral Mutation               | Reconstruction error on future strains < Baseline.         | `SCI-002`     |
| `Hierarchy`       | Radius-Valuation Correlation | Spearman Rho > 0.9 (Radius correlates with Val).           | `SCI-HIE-001` |
| `Hierarchy`       | Pairwise Order Accuracy      | > 95% of pairs correctly ordered by radius.                | `SCI-HIE-002` |
| `Arithmetic`      | Operation Prediction         | Given (a, b), predict c. Top-1 Acc > 80%.                  | `SCI-ARI-001` |
| `Arithmetic`      | Zero Prediction              | Predict if result is 0. F1 > 0.9.                          | `SCI-ARI-002` |
| `Retrieval`       | KNN Valuation Consistency    | > 80% of K=10 neighbors share valuation.                   | `SCI-RET-001` |
| `Generalization`  | Unseen Depths                | Val Loss on unseen depths < 1.1x training loss.            | `SCI-GEN-001` |
