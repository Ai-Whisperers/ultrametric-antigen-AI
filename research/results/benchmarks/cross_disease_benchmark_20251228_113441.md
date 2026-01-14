# Cross-Disease Benchmark Results

**Generated**: 2025-12-28T11:34:41.839556
**Total Runtime**: 0.35s

## Summary

- **Overall Spearman**: 0.5439 ± 0.2670
- **Diseases Evaluated**: 9

## Per-Disease Results

| Disease | Samples | Spearman | RMSE | Runtime |
|---------|---------|----------|------|---------|
| rsv | 16 | 0.8940 ± 0.0488 | 0.2352 | 0.01s |
| cancer | 7 | 0.7722 ± 0.2278 | 0.2908 | 0.00s |
| hcv | 16 | 0.5676 ± 0.1179 | 0.3751 | 0.01s |
| malaria | 21 | 0.2493 ± 0.0310 | 0.2700 | 0.01s |
| sars_cov_2 | 23 | 0.2366 ± 0.1436 | 0.1895 | 0.31s |
| influenza | 6 | 0.0000 ± 0.0000 | 0.2158 | 0.00s |
| hbv | 11 | 0.0000 ± 0.0000 | 0.3959 | 0.00s |
| mrsa | 5 | 0.0000 ± 0.0000 | 0.5544 | 0.00s |
| candida | 14 | 0.0000 ± 0.0000 | 0.3426 | 0.00s |

## Configuration

```json
{
  "diseases": [
    "sars_cov_2",
    "tuberculosis",
    "influenza",
    "hcv",
    "hbv",
    "malaria",
    "mrsa",
    "candida",
    "rsv",
    "cancer"
  ],
  "n_folds": 2,
  "n_repeats": 1,
  "seed": 42
}
```