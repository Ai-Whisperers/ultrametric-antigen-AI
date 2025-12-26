# Results Directory

> **All outputs from training, analysis, and research pipelines**

**Last Updated:** December 26, 2025

---

## Directory Structure

```
results/
├── clinical/               # Clinical decision support outputs
│   ├── clinical_applications/   # Main clinical reports
│   ├── clinical_dashboard/      # Dashboard data
│   └── clinical_integration/    # Integration outputs
│
├── research/               # Research discovery outputs
│   ├── research_discoveries/    # Latest findings (Dec 2025)
│   ├── advanced_research/       # Advanced analysis
│   ├── comprehensive_analysis/  # Comprehensive reports
│   └── discoveries/             # Discovery records
│
├── training/               # Model training outputs
│   ├── checkpoints/        # Model checkpoints
│   ├── training_runs/      # Training run logs
│   ├── run_history/        # Historical runs
│   ├── logs/               # TensorBoard logs
│   └── metrics/            # Training metrics
│
├── validation/             # Validation outputs
│   ├── alphafold_inputs/   # AlphaFold input files
│   ├── alphafold_predictions/  # AlphaFold outputs
│   └── benchmarks/         # Benchmark results
│
└── literature/             # Literature implementation outputs
    ├── literature_implementations/
    ├── advanced_literature_implementations/
    └── cutting_edge_implementations/
```

---

## Key Reports

### Clinical (2025-12-26)
- `clinical/clinical_applications/CLINICAL_REPORT.md` - Main clinical findings
- `clinical/clinical_applications/clinical_decision_support.json` - Machine-readable

### Research (2025-12-26)
- `research/research_discoveries/RESEARCH_FINDINGS.md` - Key discoveries
- `research/COMPREHENSIVE_HIV_RESEARCH_REPORT.md` - Full research report

### Validation
- `validation/benchmarks/RESOLUTION_COMPARISON.md` - Method comparison

---

## Quick Access

```bash
# View latest clinical report
cat results/clinical/clinical_applications/CLINICAL_REPORT.md

# View research findings
cat results/research/research_discoveries/RESEARCH_FINDINGS.md

# Load clinical JSON
python -c "import json; print(json.load(open('results/clinical/clinical_applications/clinical_decision_support.json')))"
```

---

## Notes

- Original flat structure preserved for backwards compatibility
- Use new categorized structure for new work
- See individual subdirectory READMEs for details
