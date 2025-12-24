# Comprehensive Code Health Report
**Generated:** 2025-12-24 06:41
**Scope:** Full Repository Audit (Security & Complexity)

## 1. Security Analysis (Bandit)
- **Total Issues:** 143
- **High Severity:** 5
- **Medium Severity:** 54

| Severity | Confidence | Issue | File |
| :--- | :--- | :--- | :--- |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\.claude\exemplars\pytorch_module.py:36` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\.claude\exemplars\test_pattern.py:17` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\.claude\exemplars\test_pattern.py:18` |
| 🟡 LOW | HIGH | Consider possible security implications associated with the subprocess module. | `.\.cursor\scripts\validate-cursor-config.py:16` |
| 🟡 LOW | HIGH | subprocess call - check for execution of untrusted input. | `.\.cursor\scripts\validate-cursor-config.py:22` |
| 🟠 MEDIUM | LOW | Possible SQL injection vector through string-based query construction. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\03_citrullination_analysis.py:669` |
| 🟠 MEDIUM | LOW | Call to requests without timeout | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\12_download_human_proteome.py:98` |
| 🟠 MEDIUM | LOW | Call to requests without timeout | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\12_download_human_proteome.py:145` |
| 🟡 LOW | HIGH | Consider possible security implications associated with pickle module. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\15_predict_immunogenicity.py:19` |
| 🟡 LOW | HIGH | Try, Except, Continue detected. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\15_predict_immunogenicity.py:262` |
| 🟠 MEDIUM | HIGH | Audit url open for permitted schemes. Allowing use of file:/ or custom schemes is often unexpected. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\18_extract_acpa_proteins.py:133` |
| 🟡 LOW | HIGH | Try, Except, Continue detected. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\19_alphafold_structure_mapping.py:104` |
| 🟡 LOW | HIGH | Try, Except, Continue detected. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\19_alphafold_structure_mapping.py:302` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\hyperbolic_utils.py:289` |
| 🟡 LOW | HIGH | Consider possible security implications associated with the subprocess module. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\generate_all.py:7` |
| 🟡 LOW | HIGH | subprocess call - check for execution of untrusted input. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\generate_all.py:21` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\01_bioinformatics_analysis.py:739` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\02_genetic_code_padic.py:653` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\03_reverse_padic_search.py:596` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\04_fast_reverse_search.py:98` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\06_learn_codon_mapping.py:211` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\06_learn_codon_mapping.py:645` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\07_extract_v5_11_3_embeddings.py:77` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\07_extract_v5_11_3_embeddings.py:182` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\08_find_natural_positions_v5_11_3.py:323` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\09_train_codon_encoder_3adic.py:183` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\09_train_codon_encoder_3adic.py:508` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\p-adic-genomics\validations\deep_encoder_validation.py:258` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\p-adic-genomics\validations\encoder_cross_validation.py:114` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\01_extract_embeddings.py:29` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\01_extract_embeddings.py:93` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\02_compute_spectrum.py:326` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\02_compute_spectrum_fast.py:106` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\04_padic_spectral_analysis.py:83` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\05_exact_padic_analysis.py:235` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\07_adelic_analysis.py:387` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\08_alternative_spectral_operators.py:427` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\09_binary_ternary_decomposition.py:391` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\11_variational_orthogonality_test.py:45` |
| 🟡 LOW | HIGH | Consider possible security implications associated with the subprocess module. | `.\scripts\analysis\analyze_external_tools.py:2` |
| 🟡 LOW | HIGH | subprocess call - check for execution of untrusted input. | `.\scripts\analysis\analyze_external_tools.py:95` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\analysis\analyze_zero_structure.py:64` |
| 🟡 LOW | HIGH | Consider possible security implications associated with the subprocess module. | `.\scripts\analysis\audit_repo.py:1` |
| 🔴 HIGH | HIGH | subprocess call with shell=True identified, security issue. | `.\scripts\analysis\audit_repo.py:18` |
| 🔴 HIGH | HIGH | Use of weak MD5 hash for security. Consider usedforsecurity=False | `.\scripts\analysis\code_stats.py:62` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\analysis\compare_options.py:37` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\analysis\compare_options.py:56` |
| 🟡 LOW | HIGH | Consider possible security implications associated with the subprocess module. | `.\scripts\analysis\comprehensive_audit.py:1` |
| 🔴 HIGH | HIGH | subprocess call with shell=True identified, security issue. | `.\scripts\analysis\comprehensive_audit.py:48` |
| 🟡 LOW | HIGH | Try, Except, Pass detected. | `.\scripts\analysis\comprehensive_audit.py:95` |
| 🟡 LOW | HIGH | Consider possible security implications associated with the subprocess module. | `.\scripts\analysis\generate_all_reports.py:4` |
| 🟡 LOW | HIGH | subprocess call - check for execution of untrusted input. | `.\scripts\analysis\generate_all_reports.py:25` |
| 🟡 LOW | HIGH | subprocess call - check for execution of untrusted input. | `.\scripts\analysis\generate_all_reports.py:65` |
| 🟡 LOW | HIGH | Consider possible security implications associated with the subprocess module. | `.\scripts\analysis\run_metrics.py:1` |
| 🟡 LOW | HIGH | subprocess call - check for execution of untrusted input. | `.\scripts\analysis\run_metrics.py:14` |
| 🟡 LOW | HIGH | subprocess call - check for execution of untrusted input. | `.\scripts\analysis\run_metrics.py:35` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\analysis\verify_mathematical_proofs.py:70` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\benchmark\run_benchmark.py:89` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\eval\downstream_validation.py:56` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\analyze_3adic_deep.py:31` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\analyze_3adic_structure.py:36` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\analyze_advanced_manifold.py:34` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\analyze_v5_5_quality.py:30` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\calabi_yau_fibration.py:35` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\calabi_yau_projection.py:31` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\calabi_yau_surface_mesh.py:37` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\calabi_yau_v58_extended.py:23` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\calabi_yau_v58_fast.py:20` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\calabi_yau_v58_fibration.py:62` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\plot_training_artifacts.py:428` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\visualize_ternary_manifold.py:36` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\viz_v58_v59.py:14` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\viz_v58_v59.py:15` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\scripts\visualization\viz_v59_hyperbolic.py:14` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\src\artifacts\checkpoint_manager.py:213` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\src\artifacts\checkpoint_manager.py:255` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\src\models\ternary_vae.py:92` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\src\models\ternary_vae.py:153` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\src\models\ternary_vae.py:274` |
| 🟡 LOW | HIGH | Standard pseudo-random generators are not suitable for security/cryptographic purposes. | `.\src\training\monitor.py:842` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\tests\fixtures\legacy\test_generalization.py:142` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_generalization.py:189` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_generalization.py:225` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_generalization.py:288` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_generalization.py:336` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_generalization.py:391` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_generalization.py:430` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_generalization.py:483` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_generalization.py:529` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_generalization.py:533` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:53` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:54` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:70` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:71` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:72` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:73` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:94` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:95` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:127` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:131` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:140` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:143` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\tests\fixtures\legacy\test_reproducibility.py:161` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:170` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:171` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:178` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:183` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:189` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:194` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:200` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:229` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_reproducibility.py:230` |
| 🟠 MEDIUM | HIGH | Use of unsafe PyTorch load | `.\tests\fixtures\legacy\test_training_validation.py:43` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_training_validation.py:134` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_training_validation.py:135` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_training_validation.py:212` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_training_validation.py:263` |
| 🔴 HIGH | HIGH | Use of weak MD5 hash for security. Consider usedforsecurity=False | `.\tests\fixtures\legacy\test_training_validation.py:305` |
| 🔴 HIGH | HIGH | Use of weak MD5 hash for security. Consider usedforsecurity=False | `.\tests\fixtures\legacy\test_training_validation.py:306` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_training_validation.py:331` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_training_validation.py:332` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_training_validation.py:387` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\fixtures\legacy\test_training_validation.py:388` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\e2e\test_scientific.py:54` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\e2e\test_scientific.py:55` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\e2e\test_scientific.py:60` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\integration\test_ai_environment_smoke.py:15` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\integration\test_generalization_v5_11.py:60` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\integration\test_generalization_v5_11.py:63` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\integration\test_generalization_v5_11.py:64` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\integration\test_generalization_v5_11.py:83` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_geometry.py:23` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_geometry.py:26` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_geometry.py:36` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_geometry.py:60` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_models.py:28` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_models.py:29` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_models.py:44` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_models.py:45` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_models.py:46` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_models.py:47` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_models.py:77` |
| 🟡 LOW | HIGH | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. | `.\tests\suites\unit\test_models.py:98` |

## 2. Complexity Analysis (Radon)
**Cyclomatic Complexity (CC) Violations:** 157 functions/methods with CC > 10.

| Complexity | Rank | Location | Function |
| :---: | :---: | :--- | :--- |
| 68 | ☢️ **F** | `scripts\train\train.py` | `function main` |
| 57 | ☢️ **F** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\neurodegeneration\alzheimers\02_tau_mtbr_interface.py` | `function main` |
| 56 | ☢️ **F** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\cross_validation.py` | `function main` |
| 55 | ☢️ **F** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\neurodegeneration\alzheimers\01_tau_phospho_sweep.py` | `function main` |
| 39 | ☢️ **E** | `src\training\trainer.py` | `method train_epoch` |
| 38 | ☢️ **E** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\09_immunogenicity_analysis_augmented.py` | `function main` |
| 36 | ☢️ **E** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\06_autoantigen_epitope_analysis.py` | `function main` |
| 36 | ☢️ **E** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\genetic_code\scripts\04_fast_reverse_search.py` | `function main` |
| 33 | ☢️ **E** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\neurodegeneration\alzheimers\04_tau_combinatorial.py` | `function main` |
| 33 | ☢️ **E** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\03_citrullination_analysis.py` | `function create_visualization` |
| 33 | ☢️ **E** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\spectral_analysis_over_models\scripts\10_semantic_amplification_benchmark.py` | `function run_benchmark` |
| 32 | ☢️ **E** | `src\losses\dual_vae_loss.py` | `method forward` |
| 30 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\02_hla_expanded_analysis.py` | `function create_expanded_visualization` |
| 30 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\07_citrullination_shift_analysis.py` | `function main` |
| 30 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\sars_cov_2\glycan_shield\03_deep_handshake_sweep.py` | `function main` |
| 29 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\neurodegeneration\alzheimers\03_tau_vae_trajectory.py` | `function main` |
| 26 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\03_citrullination_analysis.py` | `function main` |
| 26 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\04_codon_optimizer.py` | `method optimize` |
| 25 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\20_deep_structural_analysis.py` | `function compute_spatial_clustering` |
| 25 | ☢️ **D** | `scripts\analysis\analyze_zero_structure.py` | `function analyze_checkpoint` |
| 24 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\p-adic-genomics\validations\padic_biology_validation.py` | `function run_validation_suite` |
| 23 | ☢️ **D** | `src\losses\dual_vae_loss.py` | `class DualVAELoss` |
| 22 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\15_predict_immunogenicity.py` | `function load_training_data` |
| 21 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\14_compute_geometric_features.py` | `function compute_site_features` |
| 21 | ☢️ **D** | `DOCUMENTATION\01_PROJECT_KNOWLEDGE_BASE\03_EXPERIMENTS_AND_LABS\bioinformatics\codon_encoder_research\sars_cov_2\glycan_shield\02_handshake_interface_analysis.py` | `function main` |

*...and 132 more.*
