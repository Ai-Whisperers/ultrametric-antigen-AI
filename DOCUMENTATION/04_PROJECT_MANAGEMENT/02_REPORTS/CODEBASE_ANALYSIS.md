# Codebase Analysis Report

## File Lengths (Lines of Code)

| File | Lines |
|------|-------|
| `src\models\archive\ternary_vae_v5_10.py` | 1266 |
| `src\training\hyperbolic_trainer.py` | 1033 |
| `scripts\train\train.py` | 1029 |
| `src\training\monitor.py` | 1003 |
| `src\losses\padic_losses.py` | 969 |
| `scripts\visualization\visualize_ternary_manifold.py` | 817 |
| `scripts\visualization\calabi_yau_v58_fibration.py` | 764 |
| `src\models\ternary_vae.py` | 749 |
| `scripts\visualization\analyze_3adic_deep.py` | 718 |
| `src\losses\padic_geodesic.py` | 681 |
| `scripts\visualization\analyze_3adic_structure.py` | 680 |
| `src\training\trainer.py` | 677 |
| `src\training\archive\appetitive_trainer.py` | 674 |
| `scripts\visualization\analyze_advanced_manifold.py` | 643 |
| `src\models\archive\ternary_vae_v5_7.py` | 625 |
| `scripts\train\archive\train_ternary_v5_7.py` | 616 |
| `src\losses\appetitive_losses.py` | 599 |
| `src\losses\components.py` | 593 |
| `scripts\visualization\plot_training_artifacts.py` | 555 |
| `src\losses\hyperbolic_recon.py` | 554 |
| `scripts\train\archive\train_ternary_v5_5.py` | 548 |
| `scripts\train\archive\train_ternary_v5_9.py` | 547 |
| `tests\integration\test_generalization.py` | 540 |
| `src\models\archive\ternary_vae_v5_6.py` | 538 |
| `scripts\visualization\calabi_yau_fibration.py` | 538 |
| `src\losses\dual_vae_loss.py` | 528 |
| `scripts\train\archive\train_ternary_v5_9_1.py` | 528 |
| `scripts\visualization\calabi_yau_projection.py` | 514 |
| `scripts\visualization\calabi_yau_v58_extended.py` | 504 |
| `src\models\homeostasis.py` | 488 |
| `scripts\visualization\calabi_yau_surface_mesh.py` | 473 |
| `scripts\benchmark\run_benchmark.py` | 472 |
| `scripts\train\archive\train_ternary_v5_8.py` | 453 |
| `scripts\train\archive\train_hyperbolic_structure.py` | 448 |
| `src\training\config_schema.py` | 441 |
| `scripts\train\archive\train_purposeful.py` | 439 |
| `scripts\benchmark\measure_manifold_resolution.py` | 436 |
| `scripts\benchmark\measure_coupled_resolution.py` | 430 |
| `scripts\eval\downstream_validation.py` | 399 |
| `src\geometry\poincare.py` | 395 |
| `scripts\visualization\calabi_yau_v58_fast.py` | 394 |
| `tests\integration\test_training_validation.py` | 392 |
| `src\core\ternary.py` | 379 |
| `src\losses\registry.py` | 376 |
| `src\losses\hyperbolic_prior.py` | 367 |
| `scripts\analysis\compare_options.py` | 354 |
| `scripts\visualization\analyze_v5_5_quality.py` | 351 |
| `src\losses\consequence_predictor.py` | 341 |
| `src\losses\zero_structure.py` | 317 |
| `src\models\archive\appetitive_vae.py` | 311 |
| `src\models\differentiable_controller.py` | 301 |
| `scripts\train\archive\train_ternary_v5_10.py` | 290 |
| `src\models\hyperbolic_projection.py` | 288 |
| `src\utils\metrics.py` | 277 |
| `scripts\analysis\analyze_zero_structure.py` | 273 |
| `src\observability\async_writer.py` | 257 |
| `src\artifacts\checkpoint_manager.py` | 256 |
| `src\training\environment.py` | 237 |
| `tests\integration\test_reproducibility.py` | 234 |
| `src\data\gpu_resident.py` | 225 |
| `src\losses\base.py` | 225 |
| `src\README.md` | 224 |
| `src\models\curriculum.py` | 217 |
| `src\training\schedulers.py` | 215 |
| `src\observability\metrics_buffer.py` | 198 |
| `scripts\train\archive\train_appetitive_vae.py` | 185 |
| `src\observability\coverage.py` | 179 |
| `scripts\visualization\viz_v58_v59.py` | 179 |
| `src\losses\__init__.py` | 175 |
| `src\metrics\hyperbolic.py` | 173 |
| `src\utils\ternary_lut.py` | 169 |
| `scripts\visualization\viz_v59_hyperbolic.py` | 166 |
| `src\losses\radial_stratification.py` | 158 |
| `scripts\analysis\code_stats.py` | 150 |
| `src\data\loaders.py` | 128 |
| `scripts\train\archive\train_ternary_v5_6.py` | 116 |
| `src\data\dataset.py` | 79 |
| `src\data\generation.py` | 62 |
| `src\training\__init__.py` | 57 |
| `src\__init__.py` | 53 |
| `src\utils\reproducibility.py` | 51 |
| `src\observability\__init__.py` | 48 |
| `src\models\README.md` | 44 |
| `src\utils\__init__.py` | 44 |
| `src\core\__init__.py` | 42 |
| `src\data\__init__.py` | 37 |
| `src\geometry\__init__.py` | 36 |
| `scripts\train\archive\README.md` | 35 |
| `src\models\__init__.py` | 33 |
| `src\training\README.md` | 31 |
| `src\losses\README.md` | 26 |
| `src\metrics\__init__.py` | 23 |
| `src\artifacts\__init__.py` | 13 |

**Total Lines:** 34695

## Code Duplication (Shared 6-line blocks)

This section shows pairs of files that share significant chunks of identical code (ignoring whitespace).

| File A | File B | Shared Blocks (approx lines) |
|--------|--------|------------------------------|
| `src\models\archive\ternary_vae_v5_6.py` | `src\models\archive\ternary_vae_v5_7.py` | 280 blocks |
| `src\models\archive\ternary_vae_v5_10.py` | `src\models\archive\ternary_vae_v5_7.py` | 252 blocks |
| `src\models\archive\ternary_vae_v5_10.py` | `src\models\archive\ternary_vae_v5_6.py` | 195 blocks |
| `scripts\train\archive\train_ternary_v5_8.py` | `scripts\train\archive\train_ternary_v5_9.py` | 130 blocks |
| `src\training\trainer.py` | `src\training\archive\appetitive_trainer.py` | 120 blocks |
| `scripts\train\archive\train_purposeful.py` | `scripts\train\archive\train_ternary_v5_8.py` | 105 blocks |
| `scripts\train\archive\train_ternary_v5_7.py` | `scripts\train\archive\train_ternary_v5_8.py` | 103 blocks |
| `scripts\train\archive\train_purposeful.py` | `scripts\train\archive\train_ternary_v5_7.py` | 94 blocks |
| `scripts\train\archive\train_ternary_v5_9.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 84 blocks |
| `src\training\trainer.py` | `scripts\train\archive\train_ternary_v5_7.py` | 82 blocks |
| `scripts\train\archive\train_ternary_v5_7.py` | `scripts\train\archive\train_ternary_v5_9.py` | 81 blocks |
| `scripts\train\archive\train_purposeful.py` | `scripts\train\archive\train_ternary_v5_9.py` | 79 blocks |
| `scripts\benchmark\measure_coupled_resolution.py` | `scripts\benchmark\measure_manifold_resolution.py` | 74 blocks |
| `scripts\visualization\calabi_yau_v58_extended.py` | `scripts\visualization\calabi_yau_v58_fast.py` | 73 blocks |
| `scripts\train\archive\train_ternary_v5_5.py` | `scripts\train\archive\train_ternary_v5_7.py` | 72 blocks |
| `scripts\train\archive\train_appetitive_vae.py` | `scripts\train\archive\train_ternary_v5_6.py` | 69 blocks |
| `src\training\archive\appetitive_trainer.py` | `scripts\train\archive\train_ternary_v5_7.py` | 66 blocks |
| `scripts\train\archive\train_appetitive_vae.py` | `scripts\train\archive\train_ternary_v5_5.py` | 63 blocks |
| `scripts\train\archive\train_ternary_v5_5.py` | `scripts\train\archive\train_ternary_v5_6.py` | 63 blocks |
| `src\training\schedulers.py` | `scripts\train\archive\train_ternary_v5_5.py` | 55 blocks |
| `scripts\train\archive\train_ternary_v5_8.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 50 blocks |
| `scripts\train\archive\train_ternary_v5_6.py` | `scripts\train\archive\train_ternary_v5_7.py` | 46 blocks |
| `scripts\train\archive\train_ternary_v5_6.py` | `scripts\train\archive\train_ternary_v5_8.py` | 46 blocks |
| `scripts\train\archive\train_ternary_v5_6.py` | `scripts\train\archive\train_ternary_v5_9.py` | 46 blocks |
| `scripts\train\archive\train_appetitive_vae.py` | `scripts\train\archive\train_ternary_v5_7.py` | 44 blocks |
| `src\training\hyperbolic_trainer.py` | `scripts\train\archive\train_ternary_v5_9.py` | 43 blocks |
| `scripts\train\archive\train_appetitive_vae.py` | `scripts\train\archive\train_ternary_v5_8.py` | 43 blocks |
| `scripts\train\archive\train_appetitive_vae.py` | `scripts\train\archive\train_ternary_v5_9.py` | 43 blocks |
| `scripts\train\archive\train_purposeful.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 41 blocks |
| `scripts\train\archive\train_ternary_v5_5.py` | `scripts\train\archive\train_ternary_v5_8.py` | 41 blocks |
| `scripts\train\archive\train_ternary_v5_5.py` | `scripts\train\archive\train_ternary_v5_9.py` | 41 blocks |
| `scripts\train\archive\train_ternary_v5_7.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 39 blocks |
| `scripts\train\archive\train_purposeful.py` | `scripts\train\archive\train_ternary_v5_6.py` | 36 blocks |
| `scripts\train\archive\train_appetitive_vae.py` | `scripts\train\archive\train_purposeful.py` | 35 blocks |
| `scripts\train\archive\train_ternary_v5_6.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 33 blocks |
| `src\training\trainer.py` | `scripts\train\archive\train_ternary_v5_5.py` | 32 blocks |
| `scripts\train\archive\train_purposeful.py` | `scripts\train\archive\train_ternary_v5_5.py` | 30 blocks |
| `scripts\train\archive\train_ternary_v5_5.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 26 blocks |
| `scripts\train\archive\train_appetitive_vae.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 24 blocks |
| `scripts\visualization\calabi_yau_projection.py` | `scripts\visualization\calabi_yau_surface_mesh.py` | 23 blocks |
| `src\training\archive\appetitive_trainer.py` | `scripts\train\archive\train_ternary_v5_5.py` | 22 blocks |
| `src\metrics\hyperbolic.py` | `scripts\train\archive\train_ternary_v5_9.py` | 18 blocks |
| `tests\integration\test_generalization.py` | `tests\integration\test_training_validation.py` | 17 blocks |
| `tests\integration\test_generalization.py` | `tests\integration\test_reproducibility.py` | 17 blocks |
| `scripts\visualization\calabi_yau_fibration.py` | `scripts\visualization\calabi_yau_surface_mesh.py` | 16 blocks |
| `tests\integration\test_reproducibility.py` | `tests\integration\test_training_validation.py` | 15 blocks |
| `scripts\visualization\calabi_yau_fibration.py` | `scripts\visualization\calabi_yau_projection.py` | 14 blocks |
| `scripts\visualization\calabi_yau_fibration.py` | `scripts\visualization\calabi_yau_v58_fibration.py` | 13 blocks |
| `scripts\visualization\calabi_yau_v58_extended.py` | `scripts\visualization\calabi_yau_v58_fibration.py` | 13 blocks |
| `scripts\visualization\calabi_yau_v58_fast.py` | `scripts\visualization\calabi_yau_v58_fibration.py` | 13 blocks |
| `scripts\benchmark\run_benchmark.py` | `scripts\train\archive\train_purposeful.py` | 12 blocks |
| `scripts\visualization\analyze_3adic_deep.py` | `scripts\visualization\analyze_3adic_structure.py` | 12 blocks |
| `src\training\archive\appetitive_trainer.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 11 blocks |
| `src\training\trainer.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 10 blocks |
| `src\geometry\poincare.py` | `src\geometry\__init__.py` | 6 blocks |
| `src\losses\padic_losses.py` | `src\losses\radial_stratification.py` | 6 blocks |
| `scripts\train\archive\train_hyperbolic_structure.py` | `scripts\visualization\analyze_v5_5_quality.py` | 6 blocks |
| `src\training\hyperbolic_trainer.py` | `scripts\train\archive\train_ternary_v5_8.py` | 6 blocks |
| `scripts\benchmark\run_benchmark.py` | `scripts\train\archive\train_ternary_v5_5.py` | 6 blocks |
| `scripts\visualization\analyze_3adic_structure.py` | `scripts\visualization\calabi_yau_fibration.py` | 6 blocks |
| `scripts\visualization\viz_v58_v59.py` | `scripts\visualization\viz_v59_hyperbolic.py` | 6 blocks |
| `src\losses\base.py` | `src\losses\components.py` | 5 blocks |
| `scripts\benchmark\run_benchmark.py` | `scripts\train\archive\train_appetitive_vae.py` | 5 blocks |
| `scripts\benchmark\run_benchmark.py` | `scripts\train\archive\train_ternary_v5_6.py` | 5 blocks |
| `scripts\benchmark\run_benchmark.py` | `scripts\train\archive\train_ternary_v5_7.py` | 5 blocks |
| `scripts\benchmark\run_benchmark.py` | `scripts\train\archive\train_ternary_v5_8.py` | 5 blocks |
| `scripts\benchmark\run_benchmark.py` | `scripts\train\archive\train_ternary_v5_9.py` | 5 blocks |
| `scripts\benchmark\run_benchmark.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 5 blocks |
| `src\losses\components.py` | `src\losses\__init__.py` | 4 blocks |
| `src\models\archive\ternary_vae_v5_10.py` | `scripts\train\archive\train_hyperbolic_structure.py` | 4 blocks |
| `src\models\archive\ternary_vae_v5_6.py` | `scripts\train\archive\train_hyperbolic_structure.py` | 4 blocks |
| `src\models\archive\ternary_vae_v5_7.py` | `scripts\train\archive\train_hyperbolic_structure.py` | 4 blocks |
| `src\training\hyperbolic_trainer.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 4 blocks |
| `scripts\train\archive\train_hyperbolic_structure.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 4 blocks |
| `src\core\ternary.py` | `src\core\__init__.py` | 3 blocks |
| `src\geometry\poincare.py` | `src\losses\hyperbolic_recon.py` | 3 blocks |
| `src\losses\components.py` | `src\losses\dual_vae_loss.py` | 3 blocks |
| `src\losses\dual_vae_loss.py` | `scripts\train\archive\train_ternary_v5_9.py` | 3 blocks |
| `src\losses\padic_geodesic.py` | `src\losses\radial_stratification.py` | 3 blocks |
| `src\losses\__init__.py` | `src\models\archive\appetitive_vae.py` | 3 blocks |
| `src\metrics\hyperbolic.py` | `scripts\train\archive\train_ternary_v5_9_1.py` | 3 blocks |
| `src\models\hyperbolic_projection.py` | `src\models\ternary_vae.py` | 3 blocks |
| `src\training\hyperbolic_trainer.py` | `src\training\trainer.py` | 3 blocks |
| `scripts\visualization\analyze_3adic_deep.py` | `scripts\visualization\calabi_yau_surface_mesh.py` | 3 blocks |
