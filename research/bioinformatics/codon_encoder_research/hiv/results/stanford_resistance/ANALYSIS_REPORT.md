# Stanford HIVDB Drug Resistance Analysis Report

Generated: 2025-12-26 03:29:01

## Summary Statistics

- Total mutation occurrences: 90,269
- Unique mutations: 3,647
- Mutations with valid distances: 75,440

### By Drug Class

| Drug Class | Mutations | Unique | Mean Distance | Std Distance |
|------------|-----------|--------|---------------|---------------|
| PI | 22,795 | 869 | 5.533 | 1.620 |
| NRTI | 28,704 | 1891 | 5.583 | 1.187 |
| NNRTI | 33,077 | 2063 | 5.584 | 1.190 |
| INI | 5,693 | 660 | 5.610 | 1.311 |

## Distance-Resistance Correlations

### PI

| Drug | N | Pearson r | p-value | Spearman r | p-value |
|------|---|-----------|---------|------------|----------|
| FPV | 18908 | -0.037 | 0.0000*** | -0.041 | 0.0000 |
| ATV | 14875 | -0.019 | 0.0220* | -0.031 | 0.0001 |
| IDV | 18992 | -0.023 | 0.0017** | -0.015 | 0.0417 |
| LPV | 17399 | -0.030 | 0.0001*** | -0.024 | 0.0013 |
| NFV | 19393 | -0.019 | 0.0085** | -0.010 | 0.1456 |
| SQV | 18991 | 0.004 | 0.5486 | -0.010 | 0.1676 |
| TPV | 12600 | -0.019 | 0.0305* | -0.024 | 0.0061 |
| DRV | 9680 | -0.021 | 0.0364* | -0.041 | 0.0001 |

### NRTI

| Drug | N | Pearson r | p-value | Spearman r | p-value |
|------|---|-----------|---------|------------|----------|
| ABC | 22141 | 0.038 | 0.0000*** | 0.029 | 0.0000 |
| AZT | 23568 | 0.033 | 0.0000*** | 0.039 | 0.0000 |
| D4T | 23648 | 0.022 | 0.0007*** | 0.036 | 0.0000 |
| DDI | 23682 | 0.014 | 0.0287* | 0.023 | 0.0004 |
| 3TC | 23298 | 0.028 | 0.0000*** | 0.018 | 0.0072 |
| TDF | 20120 | 0.020 | 0.0053** | 0.034 | 0.0000 |

### NNRTI

| Drug | N | Pearson r | p-value | Spearman r | p-value |
|------|---|-----------|---------|------------|----------|
| DOR | 247 | 0.294 | 0.0000*** | 0.199 | 0.0017 |
| EFV | 26727 | 0.020 | 0.0011** | 0.010 | 0.1167 |
| ETR | 11747 | 0.016 | 0.0913 | -0.010 | 0.2678 |
| NVP | 26447 | 0.017 | 0.0072** | 0.014 | 0.0244 |
| RPV | 2333 | 0.031 | 0.1343 | -0.005 | 0.8212 |

### INI

| Drug | N | Pearson r | p-value | Spearman r | p-value |
|------|---|-----------|---------|------------|----------|
| BIC | 1109 | -0.030 | 0.3179 | -0.109 | 0.0003 |
| CAB | 120 | 0.005 | 0.9607 | -0.069 | 0.4535 |
| DTG | 1753 | -0.035 | 0.1485 | -0.066 | 0.0054 |
| EVG | 4258 | -0.050 | 0.0010** | -0.005 | 0.7638 |
| RAL | 4434 | 0.001 | 0.9427 | -0.003 | 0.8635 |

## Primary vs Accessory Mutations

| Drug Class | N Primary | N Accessory | Primary Mean | Accessory Mean | p-value |
|------------|-----------|-------------|--------------|----------------|----------|
| PI | 3354 | 16425 | 5.031 | 5.635 | 0.0000*** |
| NRTI | 4665 | 19084 | 5.859 | 5.516 | 0.0000*** |
| NNRTI | 1647 | 25356 | 5.076 | 5.617 | 0.0000*** |
| INI | 541 | 4368 | 5.182 | 5.663 | 0.0000*** |

## Top Cross-Resistance Mutations

Mutations conferring resistance to 3+ drugs:

| Class | Mutation | Drugs | Distance | Max FC |
|-------|----------|-------|----------|--------|
| PI | C67CG | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | nan | 50.5 |
| PI | Q58E | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 4.317 | 62.5 |
| PI | Q92D | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 5.751 | 100.0 |
| PI | C95L | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 6.425 | 100.0 |
| PI | C95CF | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | nan | 83.0 |
| PI | C67Y | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 4.446 | 84.0 |
| PI | E34A | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 5.397 | 100.0 |
| PI | R41IK | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | nan | 100.0 |
| PI | R41E | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 5.581 | 100.0 |
| PI | P79QH | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | nan | 100.0 |
| PI | P79N | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 7.337 | 70.5 |
| PI | Q61NH | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | nan | 100.0 |
| PI | Q61HN | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | nan | 100.0 |
| PI | P39PT | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | nan | 50.3 |
| PI | P79S | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 6.650 | 100.0 |
| PI | Q92R | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 5.482 | 35.0 |
| PI | T12TAS | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | nan | 100.0 |
| PI | N37Q | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 4.113 | 100.0 |
| PI | P79A | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | 4.814 | 100.0 |
| PI | E35X | FPV, ATV, IDV, LPV, NFV, SQV, TPV, DRV | nan | 50.3 |

## Key Findings

1. **Distance-Resistance Correlation**: Mean Spearman r = -0.003
2. **Primary vs Accessory**: Primary mutations have 5.287 mean distance vs 5.608 for accessory
3. **Cross-Resistance**: 1803 mutations confer resistance to 3+ drugs

## Generated Files

- `distance_distributions.png` - Hyperbolic distance histograms by drug class
- `primary_vs_accessory.png` - Comparison of primary and accessory mutations
- `distance_vs_resistance.png` - Scatter plots of distance vs fold-change
- `cross_resistance_heatmap.png` - Drug cross-resistance correlation matrix
- `mutation_distances.csv` - Complete mutation data with distances
- `cross_resistance.csv` - Cross-resistance mutation details
