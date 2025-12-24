# 02 Validation Suite: Mathematical Stress Tests

> **Objective**: Prove the "Goldilocks Zone" hypothesis rigorously through 40+ synthetic scenarios.

## A. Dimensionality & Curvature Grid

_Test the limits of the engine._

| Latent Dim (D) | Curvature (c) | p-adic Base | Topology | Test ID |
| :------------- | :------------ | :---------- | :------- | :------ |
| **8**          | -1            | 2           | Tree     | M-01    |
| **16**         | -1            | 2           | Tree     | M-02    |
| **32**         | -1            | 2           | Tree     | M-03    |
| **64**         | -1            | 2           | Tree     | M-04    |
| **128**        | -1            | 2           | Tree     | M-05    |
| **256**        | -1            | 2           | Tree     | M-06    |
| **512**        | -1            | 2           | Tree     | M-07    |
| **1024**       | -1            | 2           | Tree     | M-08    |
| **8**          | -0.1          | 3           | Cycle    | M-09    |
| **16**         | -0.1          | 3           | Cycle    | M-10    |
| **32**         | -0.1          | 3           | Cycle    | M-11    |
| **64**         | -0.1          | 3           | Cycle    | M-12    |
| **128**        | -0.1          | 3           | Cycle    | M-13    |
| **8**          | -2.0          | 5           | Mesh     | M-14    |
| **16**         | -2.0          | 5           | Mesh     | M-15    |
| **32**         | -2.0          | 5           | Mesh     | M-16    |
| **64**         | -2.0          | 5           | Mesh     | M-17    |
| **8**          | -5.0          | 7           | Random   | M-18    |
| **16**         | -5.0          | 7           | Random   | M-19    |
| **32**         | -5.0          | 7           | Random   | M-20    |
| **64**         | -5.0          | 7           | Random   | M-21    |
| **8**          | -10.0         | 11          | Star     | M-22    |
| ...            | ...           | ...         | ...      | ...     |

## B. Metric Preservation

For each of the 40 scenarios above, we calculate:

1.  **δ-Hyperbolicity**: (Target: < 0.1)
2.  **Ultrametricity Score**: (Target: > 0.99)
3.  **Correlation with Tree Distance**: (Target: > 0.95)
