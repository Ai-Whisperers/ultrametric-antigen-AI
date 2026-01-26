# Package Manifest: Carlos Brizuela

**Generated:** January 19, 2026
**Package:** `deliverables/partners/carlos_brizuela`

This manifest lists all files included in the deliverable package and their specific purpose.

## 📂 Root Directory
- `README.md`: Central documentation hub. Start here.
- `MANIFEST.md`: This file.
- `INTEGRATION_PLAN.md`: Plan for integrating with the wider system.
- `paths_critical.md`: Documentation of critical file paths.

## 📂 scripts/ (Core Executables)
Primary tools for AMP design and optimization.
- `B1_pathogen_specific_design.py`: **Main Tool.** Designs AMPs for specific pathogens.
- `B8_microbiome_safe_amps.py`: **Main Tool.** Designs microbiome-safe AMPs.
- `B10_synthesis_optimization.py`: **Main Tool.** Optimizes for synthesis feasibility.
- `latent_nsga2.py`: The NSGA-II optimization algorithm implementation.
- `predict_mic.py`: Utility for MIC prediction.
- `sequence_nsga2.py`: Sequence-based NSGA-II implementation.
- `dramp_activity_loader.py`: Utility to load DRAMP activity data.

## 📂 docs/ (Documentation)
Detailed user guides and technical reports.
- `B1_USER_GUIDE.md`: Guide for pathogen-specific design.
- `B8_USER_GUIDE.md`: Guide for microbiome-safe design.
- `B10_USER_GUIDE.md`: Guide for synthesis optimization.
- `AMP_RESULTS_INTERPRETATION.md`: How to interpret results.
- `MODEL_COMPARISON_ANALYSIS.md`: Comparison of different models.
- `PADIC_INTEGRATION_RECOMMENDATIONS.md`: Recommendations for p-adic integration.

## 📂 notebooks/ (Interactive)
- `brizuela_amp_navigator.ipynb`: Jupyter notebook for exploring AMP designs.

## 📂 results/ (Outputs)
Generated results and datasets.
- `pathogen_specific/`: Results from B1 tool.
- `microbiome_safe/`: Results from B8 tool.
- `synthesis_optimized/`: Results from B10 tool.
- `pareto_peptides.csv`: List of Pareto-optimal peptide candidates.
- `validation_batch/`: Validation results.

## 📂 src/ (Library)
Shared Python modules.
- `vae_interface.py`: Interface to the Variational Autoencoder.
- `objectives.py`: Definition of objective functions for optimization.

## 📂 training/ (Model Training)
Scripts for training the models.
- `train_definitive.py`: Main training script.
- `train_improved.py`: Improved training script.
- `dataset.py`: Dataset handling for training.

## 📂 validation/ (Verification)
Validation scripts and reports.
- `comprehensive_validation.py`: Runs full validation suite.
- `bootstrap_test.py`: Statistical bootstrap testing.
- `SCIENTIFIC_VALIDATION_REPORT.md`: Detailed validation report.
