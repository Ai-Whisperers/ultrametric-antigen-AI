# Package Manifest: Jose Colbes

**Generated:** January 19, 2026
**Package:** `deliverables/partners/jose_colbes`

This manifest lists all files included in the deliverable package and their specific purpose.

## 📂 Root Directory
- `README.md`: Central documentation hub. Start here.
- `MANIFEST.md`: This file.
- `DISRUPTION_POTENTIAL.md`: Analysis of competitive advantages.
- `VALIDATION_SUMMARY.md`: Summary of validation results.

## 📂 scripts/ (Core Executables)
Production tools for stability analysis.
- `C1_rosetta_blind_detection.py`: **Main Tool.** Identifies "Rosetta-blind" unstable residues.
- `C4_mutation_effect_predictor.py`: **Main Tool.** Predicts DDG for mutations.
- `proth`: (Likely a script or binary, check file type).

## 📂 docs/ (Documentation)
Detailed guides and reports.
- `C1_USER_GUIDE.md`: Guide for Rosetta-blind detection.
- `C4_USER_GUIDE.md`: Guide for DDG prediction.
- `PADIC_DECISION_GUIDE.md`: Decision trees for using p-adic methods.
- `BENCHMARK_COMPARISON.md`: Comparison with other methods (FoldX, etc.).
- `PROTEIN_STABILITY_RESULTS_INTERPRETATION.md`: Guide to interpreting results.

## 📂 notebooks/ (Interactive)
- `colbes_scoring_function.ipynb`: Jupyter notebook for the scoring function.

## 📂 reproducibility/ (Scientific Validation)
Scripts and data to reproduce benchmark results.
- `README.md`: Guide to reproducibility.
- `train_padic_ddg_predictor_v2.py`: Script to train the predictor.
- `extract_aa_embeddings_v2.py`: Script to extract amino acid embeddings.
- `validate_padic_s669.py`: Validation against S669 dataset.
- `download_s669.py`: Script to download benchmark data.
- `generate_benchmark_report.py`: Generates the benchmark report.

## 📂 results/ (Outputs)
Demo results and figures.
- `rosetta_blind/`: Results from Rosetta-blind analysis.
- `mutation_effects/`: Results from mutation effect prediction.
- `figures/`: Generated figures (e.g., Figure 5).
- `rotamer_analysis/`: Analysis of rotamer stability.

## 📂 src/ (Library)
Core library code.
- `validated_ddg_predictor.py`: Main class for DDG prediction.
- `scoring.py`: Scoring utilities.

## 📂 validation/ (Statistical)
Rigorous statistical validation scripts.
- `bootstrap_test.py`: Bootstrap significance testing.
- `alphafold_validation_pipeline.py`: Validation using AlphaFold structures.
- `test_alternative_hypotheses.py`: Testing alternative scientific hypotheses.
