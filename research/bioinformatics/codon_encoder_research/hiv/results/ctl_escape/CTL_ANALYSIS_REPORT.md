# CTL Epitope Escape Analysis Report

Generated: 2025-12-26 03:29:12

## Summary Statistics

- Total epitopes: 2,115
- Epitopes with HLA data: 1,532
- Epitopes with embeddings: 2,115
- Unique HLA types: 214

### Epitopes by Protein

| Protein | Epitopes | Mean Length | Mean Radius |
|---------|----------|-------------|-------------|
| Gag | 723 | 9.9 | 0.9250 |
| Pol | 497 | 9.5 | 0.9270 |
| Env | 353 | 9.7 | 0.9276 |
| Nef | 305 | 9.7 | 0.9334 |
| Tat | 47 | 9.7 | 0.9323 |
| Rev | 67 | 9.4 | 0.9304 |
| Vif | 71 | 9.4 | 0.9291 |
| Vpr | 37 | 9.5 | 0.9212 |
| Vpu | 14 | 9.4 | 0.9266 |

### Top HLA Restrictions

| HLA Type | Epitope Count | Mean Radius | Std Radius |
|----------|---------------|-------------|------------|
| A*02 | 467 | 0.9294 | 0.0184 |
| A*03 | 228 | 0.9283 | 0.0202 |
| B*35 | 152 | 0.9300 | 0.0177 |
| A*11 | 132 | 0.9301 | 0.0162 |
| B*57 | 123 | 0.9197 | 0.0293 |
| B*07 | 99 | 0.9308 | 0.0167 |
| B*08 | 89 | 0.9250 | 0.0216 |
| A*24 | 83 | 0.9266 | 0.0210 |
| B*27 | 83 | 0.9220 | 0.0168 |
| B*58 | 83 | 0.9233 | 0.0187 |
| B*44 | 52 | 0.9162 | 0.0230 |
| B*51 | 48 | 0.9291 | 0.0199 |

## Escape Velocity Analysis

Escape velocity measures the geometric spread of epitopes, indicating evolutionary flexibility.

| Protein | Epitopes | Mean Distance | Escape Velocity |
|---------|----------|---------------|------------------|
| Tat | 47 | 0.4461 | 0.1880 |
| Rev | 67 | 0.4306 | 0.1852 |
| Vpu | 14 | 0.4205 | 0.1562 |
| Vpr | 37 | 0.3652 | 0.1456 |
| Env | 353 | 0.4117 | 0.1388 |
| Gag | 723 | 0.3877 | 0.1340 |
| Pol | 497 | 0.3747 | 0.1338 |
| Nef | 305 | 0.3752 | 0.1319 |
| Vif | 71 | 0.3540 | 0.1131 |

## Conservation vs Radial Position

Testing hypothesis: More conserved (immunogenic) epitopes have smaller radial positions.

| Protein | Mean Radius | Correlation (r) | p-value |
|---------|-------------|-----------------|----------|
| Gag | 0.9250 | 0.021 | 0.5780 |
| Pol | 0.9270 | 0.000 | 0.9964 |
| Env | 0.9276 | 0.044 | 0.4130 |
| Nef | 0.9334 | 0.010 | 0.8687 |
| Tat | 0.9323 | -0.029 | 0.8445 |
| Rev | 0.9304 | 0.199 | 0.1070 |
| Vif | 0.9291 | -0.068 | 0.5716 |
| Vpr | 0.9212 | 0.114 | 0.5000 |
| Vpu | 0.9266 | 0.215 | 0.4600 |

## Key Findings

1. **Highest Escape Velocity**: Tat (0.1880)
2. **Most Common HLA Restriction**: A*02 (467 epitopes)

## Generated Files

- `hla_landscape_comparison.png` - HLA-specific radial positions
- `protein_escape_velocity.png` - Escape velocity by protein
- `conservation_analysis.png` - Conservation vs radius
- `epitope_length_distribution.png` - Epitope length analysis
- `radius_by_protein.png` - Radial position distributions
- `epitope_data.csv` - Complete epitope data with embeddings info
- `hla_summary.csv` - HLA restriction summary
