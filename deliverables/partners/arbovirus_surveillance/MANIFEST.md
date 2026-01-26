# Package Manifest: Alejandra Rojas

**Generated:** January 19, 2026
**Package:** `deliverables/partners/alejandra_rojas`

This manifest lists all files included in the deliverable package and their specific purpose.

## 📂 Root Directory
- `README.md`: Central documentation hub. Start here.
- `MANIFEST.md`: This file.

## 📂 scripts/ (Core Executables)
Primary tools for running analysis and primer design.
- `A2_pan_arbovirus_primers.py`: **Main Tool.** Generates pan-arbovirus primer libraries.
- `arbovirus_hyperbolic_trajectory.py`: **Main Tool.** Performs hyperbolic trajectory analysis for serotype forecasting.
- `primer_stability_scanner.py`: **Main Tool.** Scans genomes for stable regions suitable for primers.
- `ingest_arboviruses.py`: Utility to download viral genomes from NCBI.
- `denv4_*.py`: Specific scripts for Dengue Serotype 4 analysis (various conjectures).

## 📂 docs/ (Documentation)
Detailed user guides and technical reports.
- `A2_USER_GUIDE.md`: User guide for the Pan-Arbovirus Primer tool.
- `A2_TECHNICAL_DOCS.md`: Technical details for the A2 tool.
- `A2_RESULTS_INTERPRETATION.md`: How to interpret A2 results.
- `UNIFIED_RESEARCH_ROADMAP.md`: Future research directions.
- `VALIDATION_METHODOLOGY.md`: Details on the validation process.

## 📂 notebooks/ (Interactive)
- `rojas_serotype_forecast.ipynb`: Jupyter notebook for interactive exploration of forecasts.

## 📂 results/ (Outputs)
Generated results, figures, and datasets.
- `dengue_forecast.json`: Forecast results for Dengue serotypes.
- `primer_candidates.csv`: List of generated primer candidates.
- `pan_arbovirus_primers/`: Directory containing generated primer libraries for multiple viruses.
- `phylogenetic/`: Phylogenetic analysis results and figures.
- `ml_ready/`: Processed datasets ready for machine learning.

## 📂 research/ (Experimental)
Research sub-projects and explorations.
- `clade_classification/`: Code for K-mer based clade classification.
- `dual_metric/`: Dual metric scoring research.
- `e_protein_dhf/`: Analysis of E-protein in DHF.
- `functional_convergence/`: Research on functional convergence points.
- `validation/`: Validation scripts and reports.

## 📂 src/ (Library)
Shared Python modules used by scripts.
- `primer_designer.py`: Logic for primer design.
- `geometry.py`: Geometric utility functions.
- `data_pipeline.py`: Data processing pipeline components.
- `ncbi_client.py`: Client for NCBI API interactions.

## 📂 validation/ (Reports)
Comprehensive validation reports.
- `VALIDATION_MASTER_REPORT.md`: Master validation document.
- `CDC_PRIMER_RECOVERY_REPORT.md`: Validation against CDC primers.
- `DENGUE_STRAIN_VARIATION_REPORT.md`: Analysis of strain variation.
