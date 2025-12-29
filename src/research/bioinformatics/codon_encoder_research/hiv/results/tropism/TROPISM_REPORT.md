# HIV Tropism Analysis Report

Generated: 2025-12-26 03:29:43

## Summary Statistics

- Total V3 sequences: 2932
- CCR5 (R5): 2699
- CXCR4 (X4): 702
- Sequences with embeddings: 2932

## Tropism Separation Analysis

| Metric | CCR5 | CXCR4 |
|--------|------|-------|
| Mean Radius | 0.9345 | 0.9339 |
| Std Radius | 0.0149 | 0.0179 |

- Centroid Distance: 0.0222
- Mann-Whitney p-value: 0.992237

## Key Positions for Tropism

Top 10 most discriminative positions:

| Position | Separation | p-value | Key Position? |
|----------|------------|---------|---------------|
| 22 | 0.5909 | 0.000000 |  |
| 8 | 0.4315 | 0.000000 |  |
| 20 | 0.4058 | 0.000000 |  |
| 19 | 0.3734 | 0.821665 |  |
| 11 | 0.3405 | 0.000000 | Yes |
| 16 | 0.3142 | 0.000000 |  |
| 18 | 0.3086 | 0.000000 |  |
| 13 | 0.2787 | 0.468931 | Yes |
| 12 | 0.2622 | 0.000000 |  |
| 23 | 0.2449 | 0.000000 |  |

## Tropism Classifier Performance

| Classifier | Accuracy | AUC | CV Mean | CV Std |
|------------|----------|-----|---------|--------|
| Logistic Regression | 0.850 | 0.848 | 0.859 | 0.013 |
| Random Forest | 0.850 | 0.843 | 0.868 | 0.009 |

## Key Findings

1. **Geometric Separation**: CCR5 and CXCR4 sequences show 0.0222 distance in embedding space
2. **Tropism Prediction**: Best classifier achieves AUC of 0.848
3. **Most Discriminative Position**: Position 22

## Generated Files

- `tropism_separation.png` - CCR5 vs CXCR4 visualization
- `position_importance.png` - Per-position discrimination
- `classifier_performance.png` - ML classifier results
- `v3_data.csv` - V3 sequence data with embeddings info
- `position_importance.csv` - Position importance data
