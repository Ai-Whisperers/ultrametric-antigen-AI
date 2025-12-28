# Cross-Disease Benchmark Results

**Generated**: 2025-12-28T11:59:15.487345
**Total Runtime**: 0.34s

## Summary

- **Overall Spearman**: 0.7248 ± 0.1850
- **Diseases Evaluated**: 9

## Per-Disease Results

| Disease | Samples | Spearman | RMSE | Runtime |
|---------|---------|----------|------|---------|
| cancer | 7 | 1.0000 ± 0.0000 | 0.3095 | 0.02s |
| hcv | 16 | 0.9004 ± 0.1688 | 0.2990 | 0.02s |
| rsv | 16 | 0.8538 ± 0.1857 | 0.2254 | 0.03s |
| sars_cov_2 | 23 | 0.6612 ± 0.2713 | 0.1777 | 0.06s |
| hbv | 50 | 0.6476 ± 0.2453 | 0.1308 | 0.04s |
| malaria | 21 | 0.5757 ± 0.1973 | 0.2647 | 0.04s |
| mrsa | 50 | 0.4346 ± 0.3116 | 0.1337 | 0.04s |
| candida | 50 | 0.0000 ± 0.0000 | 0.2334 | 0.05s |
| influenza | 50 | -0.3412 ± 0.3198 | 0.1923 | 0.03s |

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
  "n_folds": 5,
  "n_repeats": 3,
  "seed": 42
}
```