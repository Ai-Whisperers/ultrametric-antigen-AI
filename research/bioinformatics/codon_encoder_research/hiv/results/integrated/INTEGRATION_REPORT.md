# Cross-Dataset Integration Analysis Report

Generated: 2025-12-26 03:34:57

## Summary

This report integrates multiple HIV datasets to identify:
1. Drug resistance vs immune escape trade-offs
2. Multi-pressure constraint landscape
3. Optimal vaccine targets

## Resistance-Immunity Trade-offs

- Resistance-epitope overlaps found: 16054
- Unique mutations with overlaps: 3074
- Unique epitopes affected: 298

### Top Trade-off Positions

| Mutation | Drug Class | Epitope | HLAs | Fold-Change | Score |
|----------|------------|---------|------|-------------|-------|
| S283R | INI | TAFTIPSI... | 15 | 94.5 | 5.629 |
| D67NS | NNRTI | ITLWQRPLV... | 15 | 83.7 | 5.554 |
| Q61NH | PI | ITLWQRPLV... | 15 | 79.0 | 5.518 |
| Q61G | PI | ITLWQRPLV... | 15 | 75.4 | 5.489 |
| Q61HN | PI | ITLWQRPLV... | 15 | 72.2 | 5.462 |
| K65KE | NNRTI | ITLWQRPLV... | 15 | 64.5 | 5.391 |
| D60K | PI | ITLWQRPLV... | 15 | 63.9 | 5.385 |
| I66T | PI | ITLWQRPLV... | 15 | 63.5 | 5.381 |
| I66V | PI | ITLWQRPLV... | 15 | 57.9 | 5.322 |
| K66K* | NNRTI | ITLWQRPLV... | 15 | 57.5 | 5.317 |
| C67W | PI | ITLWQRPLV... | 15 | 56.3 | 5.304 |
| D60Q | PI | ITLWQRPLV... | 15 | 55.8 | 5.298 |
| D60Y | PI | ITLWQRPLV... | 15 | 54.8 | 5.286 |
| I66IN | PI | ITLWQRPLV... | 15 | 54.3 | 5.280 |
| K65KR | NNRTI | ITLWQRPLV... | 15 | 54.0 | 5.277 |

## Constraint Landscape

### PR
- Positions analyzed: 99
- CTL epitopes overlapping: 37
- Mean conservation: 0.895

### RT
- Positions analyzed: 0
- CTL epitopes overlapping: 207

### IN
- Positions analyzed: 0
- CTL epitopes overlapping: 0

## Vaccine Target Identification

- Total candidates: 387
- Without resistance overlap: 328
- Minimum HLA restrictions: 3

### Top 20 Vaccine Targets

| Rank | Epitope | Protein | HLAs | Resistance Overlap | Score |
|------|---------|---------|------|--------------------|-------|
| 1 | TPQDLNTML | Gag | 25 | No | 2.238 |
| 2 | AAVDLSHFL | Nef | 19 | No | 1.701 |
| 3 | YPLTFGWCF | Nef | 19 | No | 1.701 |
| 4 | YFPDWQNYT | Nef | 19 | No | 1.701 |
| 5 | QVPLRPMTYK | Nef | 19 | No | 1.701 |
| 6 | RAIEAQQHL | Env | 18 | No | 1.611 |
| 7 | ITKGLGISYGR | Tat | 17 | No | 1.522 |
| 8 | RPQVPLRPM | Nef | 17 | No | 1.522 |
| 9 | GHQAAMQML | Gag | 16 | No | 1.432 |
| 10 | YPLTFGWCY | Nef | 16 | No | 1.432 |
| 11 | RYPLTFGW | Nef | 16 | No | 1.432 |
| 12 | HPVHAGPIA | Gag | 15 | No | 1.343 |
| 13 | RLRPGGKKKY | Gag | 15 | No | 1.343 |
| 14 | ISPRTLNAW | Gag | 15 | No | 1.343 |
| 15 | RGPGRAFVTI | Env | 15 | No | 1.343 |
| 16 | EAVRHFPRI | Vpr | 14 | No | 1.253 |
| 17 | WASRELERF | Gag | 14 | No | 1.253 |
| 18 | TPGPGVRYPL | Nef | 14 | No | 1.253 |
| 19 | VPLRPMTY | Nef | 14 | No | 1.253 |
| 20 | LTFGWCFKL | Nef | 13 | No | 1.164 |

## Key Findings

1. **16054 resistance-epitope overlaps** identified, representing potential evolutionary trade-offs
2. **Top vaccine target**: TPQDLNTML in Gag (Score: 2.238)
3. **328 targets** without resistance mutation overlap

## Generated Files

- `tradeoff_landscape.png` - Resistance vs immunity visualization
- `constraint_map.png` - Multi-pressure constraint landscape
- `vaccine_targets.png` - Vaccine target rankings
- `resistance_epitope_overlaps.csv` - Detailed overlap data
- `tradeoff_scores.csv` - Trade-off scoring results
- `vaccine_targets.csv` - Ranked vaccine target list
