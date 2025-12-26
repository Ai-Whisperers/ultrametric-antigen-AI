# CATNAP Antibody Neutralization Analysis Report

Generated: 2025-12-26 03:29:23

## Summary Statistics

- Total neutralization records: 189,879
- Records with valid IC50: 182,453
- Unique antibodies: 1,123
- Unique viruses: 2,960
- Known bnAbs analyzed: 16

## Neutralization Breadth

Breadth calculated at IC50 < 50.0 ug/mL

- Antibodies with >50% breadth: 917
- Antibodies with >80% breadth: 864

### Known bnAb Profiles

| Antibody | Class | N Tested | % Sensitive | Geo Mean IC50 |
|----------|-------|----------|-------------|---------------|
| 3BNC117 | CD4bs | 5212 | 78.8% | 0.242 |
| NIH45-46 | CD4bs | 2437 | 77.4% | 0.249 |
| 10E8 | MPER | 9563 | 76.7% | 0.221 |
| PG9 | V2-glycan | 8451 | 70.9% | 0.300 |
| VRC01 | CD4bs | 9724 | 68.9% | 0.580 |
| 10-1074 | V3-glycan | 4969 | 66.4% | 0.385 |
| PGT128 | V3-glycan | 4698 | 62.9% | 0.424 |
| PG16 | V2-glycan | 3735 | 60.2% | 0.504 |
| PGT121 | V3-glycan | 7522 | 59.2% | 0.566 |
| PGT145 | V2-glycan | 2658 | 55.2% | 0.763 |
| 8ANC195 | interface | 1220 | 40.4% | 2.845 |
| 35O22 | interface | 1246 | 36.1% | 2.893 |
| VRC03 | CD4bs | 1477 | 33.9% | 4.118 |
| 4E10 | MPER | 3585 | 31.6% | 2.120 |
| b12 | CD4bs | 3913 | 18.2% | 10.334 |
| 2F5 | MPER | 3306 | 16.9% | 8.142 |

## Potency by Epitope Class

| Epitope Class | Antibodies | Records | Geo Mean IC50 | Median IC50 |
|---------------|------------|---------|---------------|-------------|
| V2-glycan | 3 | 10,064 | 0.689 | 0.378 |
| V3-glycan | 3 | 12,071 | 0.745 | 0.340 |
| CD4bs | 5 | 17,899 | 1.121 | 0.566 |
| MPER | 3 | 10,875 | 1.815 | 1.700 |
| interface | 2 | 1,989 | 3.597 | 10.000 |

## Virus Susceptibility

### Most Susceptible Viruses

| Virus | Antibodies Tested | % Sensitive |
|-------|-------------------|-------------|
| 100155M_33 | 4 | 100.0% |
| 11_3_J9 | 4 | 100.0% |
| 11_5_J12 | 6 | 100.0% |
| 1536_BL_E11_4 | 4 | 100.0% |
| 1536_BL_E5_3 | 4 | 100.0% |
| 1536_BL_E6_1 | 4 | 100.0% |
| 1536_BL_G9_4 | 4 | 100.0% |
| 1536_V15_D11_4 | 4 | 100.0% |
| 1536_V15_D6_1 | 4 | 100.0% |
| 1536_V16_B6_1 | 4 | 100.0% |

### Least Susceptible Viruses (min 5 antibodies tested)

| Virus | Antibodies Tested | % Sensitive |
|-------|-------------------|-------------|
| 0735_V4_C1 | 5 | 0.0% |
| 1105_P17_1 | 5 | 0.0% |
| 1209_BM_A5 | 12 | 0.0% |
| 1656_P21 | 5 | 0.0% |
| 191727_D1_12 | 5 | 0.0% |
| 192018_B1_9 | 5 | 0.0% |
| 193003_B10 | 5 | 0.0% |
| 2705_P18_1 | 5 | 0.0% |
| 3226_P15 | 5 | 0.0% |
| 3233_P12 | 5 | 0.0% |

## Key Findings

1. **Most Potent bnAb**: 10E8 (IC50 = 0.221 ug/mL)
2. **Broadest bnAb**: 3BNC117 (78.8% sensitive)
3. **Most Potent Epitope Class**: V2-glycan (IC50 = 0.689 ug/mL)

## Generated Files

- `breadth_distribution.png` - Antibody breadth histogram
- `bnab_sensitivity.png` - bnAb sensitivity profiles
- `antibody_clustering.png` - Cross-neutralization clustering
- `virus_susceptibility.png` - Virus susceptibility patterns
- `potency_by_class.png` - Potency by epitope class
- `breadth_data.csv` - Antibody breadth data
- `bnab_sensitivity.csv` - bnAb sensitivity profiles
- `virus_susceptibility.csv` - Virus susceptibility data
