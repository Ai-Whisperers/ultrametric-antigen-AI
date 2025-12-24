# Code Health Dashboard
**Date:** 2025-12-24 00:22

**Total Issues Found:** 1111
- **Ruff (Lint/Style):** 933
- **Mypy (Type Safety):** 178

## Top Offenders
| File | Issue Count | Primary Issues |
| :--- | :---: | :--- |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\data\natural_positions_v5_11_3.py` | 64 | Undefined name `np` |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\02_genetic_code_padic.py` | 23 | Module level import not at top of file |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\01_bioinformatics_analysis.py` | 21 | Module level import not at top of file |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\05_analyze_natural_positions.py` | 20 | `json` imported but unused |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\embeddings_analysis\01_extract_compare_embeddings.py` | 19 | Module level import not at top of file |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\03_reverse_padic_search.py` | 19 | Module level import not at top of file |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\10_immunogenicity_visualizations.py` | 18 | `collections.defaultdict` imported but unused |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\20_deep_structural_analysis.py` | 18 | `typing.Tuple` imported but unused |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_ternary_v5_10.py` | 18 | `torch` imported but unused |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\tests\integration\test_training_validation.py` | 18 | `typing.Tuple` imported but unused |

## Detailed Verification Audit
> Issues grouped by file. Fix priority: Type Errors > Syntax Errors > Style.

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\data\natural_positions_v5_11_3.py` (64 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |
| 8 | **ruff** | `F821` | Undefined name `np` |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\02_genetic_code_padic.py` (23 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 27 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 30 | **ruff** | `F401` | `scipy.spatial.distance.squareform` imported but unused |
| 31 | **ruff** | `E402` | Module level import not at top of file |
| 213 | **ruff** | `F541` | f-string without any placeholders |
| 228 | **ruff** | `F541` | f-string without any placeholders |
| 234 | **ruff** | `F541` | f-string without any placeholders |
| 282 | **ruff** | `F541` | f-string without any placeholders |
| 283 | **ruff** | `F541` | f-string without any placeholders |
| 341 | **ruff** | `F541` | f-string without any placeholders |
| 356 | **ruff** | `F541` | f-string without any placeholders |
| 418 | **ruff** | `F541` | f-string without any placeholders |
| 422 | **ruff** | `F541` | f-string without any placeholders |
| 424 | **ruff** | `F541` | f-string without any placeholders |
| 460 | **ruff** | `F541` | f-string without any placeholders |
| 465 | **ruff** | `F541` | f-string without any placeholders |
| 477 | **ruff** | `F541` | f-string without any placeholders |
| 478 | **ruff** | `F541` | f-string without any placeholders |
| 480 | **ruff** | `F541` | f-string without any placeholders |
| 482 | **ruff** | `F541` | f-string without any placeholders |
| 558 | **ruff** | `F841` | Local variable `sorted_labels` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\01_bioinformatics_analysis.py` (21 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 25 | **ruff** | `E402` | Module level import not at top of file |
| 26 | **ruff** | `E402` | Module level import not at top of file |
| 27 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `F401` | `scipy.cluster.hierarchy.fcluster` imported but unused |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 451 | **ruff** | `F541` | f-string without any placeholders |
| 486 | **ruff** | `F541` | f-string without any placeholders |
| 488 | **ruff** | `F541` | f-string without any placeholders |
| 490 | **ruff** | `F541` | f-string without any placeholders |
| 492 | **ruff** | `F541` | f-string without any placeholders |
| 577 | **ruff** | `F541` | f-string without any placeholders |
| 580 | **ruff** | `F541` | f-string without any placeholders |
| 583 | **ruff** | `F541` | f-string without any placeholders |
| 608 | **ruff** | `F841` | Local variable `codons` is assigned to but never used |
| 621 | **ruff** | `F841` | Local variable `chem_class` is assigned to but never used |
| 626 | **ruff** | `F541` | f-string without any placeholders |
| 640 | **ruff** | `F541` | f-string without any placeholders |
| 712 | **ruff** | `F541` | f-string without any placeholders |
| 713 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\05_analyze_natural_positions.py` (20 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 16 | **ruff** | `F401` | `json` imported but unused |
| 22 | **ruff** | `E402` | Module level import not at top of file |
| 267 | **ruff** | `F541` | f-string without any placeholders |
| 273 | **ruff** | `F541` | f-string without any placeholders |
| 276 | **ruff** | `F541` | f-string without any placeholders |
| 298 | **ruff** | `F541` | f-string without any placeholders |
| 307 | **ruff** | `F541` | f-string without any placeholders |
| 331 | **ruff** | `F541` | f-string without any placeholders |
| 364 | **ruff** | `F541` | f-string without any placeholders |
| 385 | **ruff** | `F541` | f-string without any placeholders |
| 400 | **ruff** | `F541` | f-string without any placeholders |
| 401 | **ruff** | `F541` | f-string without any placeholders |
| 418 | **ruff** | `F541` | f-string without any placeholders |
| 420 | **ruff** | `F541` | f-string without any placeholders |
| 434 | **ruff** | `F841` | Local variable `nucleotides` is assigned to but never used |
| 470 | **ruff** | `F541` | f-string without any placeholders |
| 519 | **ruff** | `F541` | f-string without any placeholders |
| 520 | **ruff** | `F541` | f-string without any placeholders |
| 521 | **ruff** | `F541` | f-string without any placeholders |
| 557 | **ruff** | `F841` | Local variable `valuations` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\embeddings_analysis\01_extract_compare_embeddings.py` (19 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **ruff** | `E402` | Module level import not at top of file |
| 19 | **ruff** | `E402` | Module level import not at top of file |
| 20 | **ruff** | `E402` | Module level import not at top of file |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 22 | **ruff** | `E402` | Module level import not at top of file |
| 109 | **ruff** | `F841` | Local variable `direction_layers` is assigned to but never used |
| 110 | **ruff** | `F841` | Local variable `radius_layers` is assigned to but never used |
| 133 | **ruff** | `F841` | Local variable `b` is assigned to but never used |
| 217 | **ruff** | `F841` | Local variable `n_layers` is assigned to but never used |
| 390 | **ruff** | `F401` | `scipy.cluster.hierarchy.fcluster` imported but unused |
| 391 | **ruff** | `F841` | Local variable `Z` is assigned to but never used |
| 535 | **ruff** | `F541` | f-string without any placeholders |
| 539 | **ruff** | `F541` | f-string without any placeholders |
| 544 | **ruff** | `F541` | f-string without any placeholders |
| 549 | **ruff** | `F541` | f-string without any placeholders |
| 554 | **ruff** | `F541` | f-string without any placeholders |
| 634 | **ruff** | `F541` | f-string without any placeholders |
| 642 | **ruff** | `F541` | f-string without any placeholders |
| 644 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\03_reverse_padic_search.py` (19 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 27 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `F401` | `scipy.stats` imported but unused |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `F401` | `scipy.spatial.distance.squareform` imported but unused |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 88 | **ruff** | `F841` | Local variable `n_points` is assigned to but never used |
| 210 | **ruff** | `F541` | f-string without any placeholders |
| 296 | **ruff** | `F541` | f-string without any placeholders |
| 356 | **ruff** | `F541` | f-string without any placeholders |
| 383 | **ruff** | `F841` | Local variable `start_idx` is assigned to but never used |
| 474 | **ruff** | `F541` | f-string without any placeholders |
| 477 | **ruff** | `F541` | f-string without any placeholders |
| 483 | **ruff** | `F541` | f-string without any placeholders |
| 492 | **ruff** | `F541` | f-string without any placeholders |
| 707 | **ruff** | `F541` | f-string without any placeholders |
| 710 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\10_immunogenicity_visualizations.py` (18 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 12 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 16 | **ruff** | `E402` | Module level import not at top of file |
| 17 | **ruff** | `E402` | Module level import not at top of file |
| 17 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 18 | **ruff** | `E402` | Module level import not at top of file |
| 19 | **ruff** | `E402` | Module level import not at top of file |
| 20 | **ruff** | `E402` | Module level import not at top of file |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 22 | **ruff** | `E402` | Module level import not at top of file |
| 25 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `F401` | `hyperbolic_utils.poincare_distance` imported but unused |
| 34 | **ruff** | `E402` | Module level import not at top of file |
| 133 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 134 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 178 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 251 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 315 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 445 | **ruff** | `E741` | Ambiguous variable name: `l` |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\20_deep_structural_analysis.py` (18 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 24 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 24 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 25 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 27 | **ruff** | `F401` | `scipy.cluster.hierarchy.linkage` imported but unused |
| 27 | **ruff** | `F401` | `scipy.cluster.hierarchy.fcluster` imported but unused |
| 230 | **ruff** | `F841` | Local variable `threshold` is assigned to but never used |
| 321 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 322 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 508 | **ruff** | `F841` | Local variable `markers` is assigned to but never used |
| 551 | **ruff** | `F541` | f-string without any placeholders |
| 568 | **ruff** | `F841` | Local variable `summary_data` is assigned to but never used |
| 594 | **ruff** | `F541` | f-string without any placeholders |
| 598 | **ruff** | `F541` | f-string without any placeholders |
| 625 | **ruff** | `F841` | Local variable `sites_by_protein` is assigned to but never used |
| 644 | **ruff** | `F541` | f-string without any placeholders |
| 706 | **ruff** | `F541` | f-string without any placeholders |
| 716 | **ruff** | `F541` | f-string without any placeholders |
| 726 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_ternary_v5_10.py` (18 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 22 | **ruff** | `F401` | `torch` imported but unused |
| 23 | **ruff** | `F401` | `torch.optim` imported but unused |
| 28 | **ruff** | `F401` | `src.models.curriculum.ContinuousCurriculumModule` imported but unused |
| 42 | **ruff** | `F401` | `src.losses.ConsequencePredictor` imported but unused |
| 43 | **ruff** | `F401` | `src.losses.evaluate_addition_accuracy` imported but unused |
| 44 | **ruff** | `F401` | `src.losses.RadialStratificationLoss` imported but unused |
| 45 | **ruff** | `F401` | `src.losses.PurposefulRankingLoss` imported but unused |
| 55 | **ruff** | `F841` | Local variable `validated` is assigned to but never used |
| 155 | **ruff** | `F541` | f-string without any placeholders |
| 168 | **ruff** | `F541` | f-string without any placeholders |
| 171 | **ruff** | `F541` | f-string without any placeholders |
| 172 | **ruff** | `F541` | f-string without any placeholders |
| 178 | **ruff** | `F541` | f-string without any placeholders |
| 184 | **ruff** | `F541` | f-string without any placeholders |
| 185 | **ruff** | `F541` | f-string without any placeholders |
| 187 | **ruff** | `F541` | f-string without any placeholders |
| 189 | **ruff** | `F541` | f-string without any placeholders |
| 195 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\tests\integration\test_training_validation.py` (18 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 18 | **ruff** | `F401` | `typing.Dict` imported but unused |
| 18 | **ruff** | `F401` | `typing.List` imported but unused |
| 113 | **ruff** | `F541` | f-string without any placeholders |
| 118 | **ruff** | `F541` | f-string without any placeholders |
| 202 | **ruff** | `F541` | f-string without any placeholders |
| 207 | **ruff** | `F541` | f-string without any placeholders |
| 260 | **ruff** | `F541` | f-string without any placeholders |
| 265 | **ruff** | `F541` | f-string without any placeholders |
| 326 | **ruff** | `F541` | f-string without any placeholders |
| 329 | **ruff** | `F541` | f-string without any placeholders |
| 360 | **ruff** | `F541` | f-string without any placeholders |
| 366 | **ruff** | `F541` | f-string without any placeholders |
| 368 | **ruff** | `F541` | f-string without any placeholders |
| 370 | **ruff** | `F541` | f-string without any placeholders |
| 376 | **ruff** | `F541` | f-string without any placeholders |
| 382 | **ruff** | `F541` | f-string without any placeholders |
| 384 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\11_immunogenicity_predictor.py` (17 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 43 | **ruff** | `E402` | Module level import not at top of file |
| 44 | **ruff** | `E402` | Module level import not at top of file |
| 45 | **ruff** | `E402` | Module level import not at top of file |
| 46 | **ruff** | `E402` | Module level import not at top of file |
| 48 | **ruff** | `F401` | `sklearn.metrics.precision_recall_curve` imported but unused |
| 48 | **ruff** | `F401` | `sklearn.metrics.average_precision_score` imported but unused |
| 50 | **ruff** | `F401` | `sklearn.metrics.confusion_matrix` imported but unused |
| 50 | **ruff** | `F401` | `sklearn.metrics.classification_report` imported but unused |
| 53 | **ruff** | `E402` | Module level import not at top of file |
| 54 | **ruff** | `E402` | Module level import not at top of file |
| 54 | **ruff** | `F401` | `sklearn.model_selection.LeaveOneOut` imported but unused |
| 55 | **ruff** | `E402` | Module level import not at top of file |
| 55 | **ruff** | `F401` | `scipy.stats` imported but unused |
| 58 | **ruff** | `E402` | Module level import not at top of file |
| 59 | **ruff** | `E402` | Module level import not at top of file |
| 63 | **ruff** | `E402` | Module level import not at top of file |
| 72 | **ruff** | `E402` | Module level import not at top of file |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\24_model_validation_visualizations.py` (17 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 27 | **ruff** | `F401` | `typing.List` imported but unused |
| 27 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 29 | **ruff** | `F401` | `sklearn.metrics.precision_recall_curve` imported but unused |
| 29 | **ruff** | `F401` | `sklearn.metrics.average_precision_score` imported but unused |
| 119 | **ruff** | `F841` | Local variable `epitopes` is assigned to but never used |
| 173 | **ruff** | `E712` | Avoid equality comparisons to `True`; use `df['immunodominant']:` for truth checks |
| 174 | **ruff** | `E712` | Avoid equality comparisons to `False`; use `not df['immunodominant']:` for false checks |
| 313 | **ruff** | `F841` | Local variable `af3_comparisons` is assigned to but never used |
| 421 | **ruff** | `F841` | Local variable `labels` is assigned to but never used |
| 422 | **ruff** | `F841` | Local variable `sizes` is assigned to but never used |
| 439 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 542 | **ruff** | `F841` | Local variable `proteome_stats` is assigned to but never used |
| 544 | **ruff** | `F841` | Local variable `shifts` is assigned to but never used |
| 729 | **ruff** | `F841` | Local variable `shifts` is assigned to but never used |
| 746 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 767 | **ruff** | `E712` | Avoid equality comparisons to `True`; use `df['immunodominant']:` for truth checks |
| 768 | **ruff** | `E712` | Avoid equality comparisons to `False`; use `not df['immunodominant']:` for false checks |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\09_binary_ternary_decomposition.py` (16 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 33 | **ruff** | `E402` | Module level import not at top of file |
| 34 | **ruff** | `E402` | Module level import not at top of file |
| 35 | **ruff** | `E402` | Module level import not at top of file |
| 35 | **ruff** | `F401` | `scipy.linalg.svd` imported but unused |
| 36 | **ruff** | `E402` | Module level import not at top of file |
| 37 | **ruff** | `E402` | Module level import not at top of file |
| 37 | **ruff** | `F401` | `matplotlib.pyplot` imported but unused |
| 107 | **ruff** | `F841` | Local variable `corr_pc2_mod2` is assigned to but never used |
| 107 | **ruff** | `F841` | Local variable `p2` is assigned to but never used |
| 183 | **ruff** | `F541` | f-string without any placeholders |
| 220 | **ruff** | `F841` | Local variable `z_level` is assigned to but never used |
| 357 | **ruff** | `F541` | f-string without any placeholders |
| 359 | **ruff** | `F541` | f-string without any placeholders |
| 365 | **ruff** | `F541` | f-string without any placeholders |
| 373 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\analyze_3adic_deep.py` (16 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 15 | **ruff** | `F401` | `itertools.product` imported but unused |
| 19 | **ruff** | `F401` | `torch.nn.functional` imported but unused |
| 21 | **ruff** | `F401` | `scipy.stats.pearsonr` imported but unused |
| 22 | **ruff** | `F401` | `scipy.spatial.distance.cdist` imported but unused |
| 24 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 220 | **ruff** | `F841` | Local variable `scatter` is assigned to but never used |
| 229 | **ruff** | `F841` | Local variable `im` is assigned to but never used |
| 539 | **ruff** | `F841` | Local variable `nn_dist` is assigned to but never used |
| 552 | **ruff** | `F541` | f-string without any placeholders |
| 586 | **ruff** | `F541` | f-string without any placeholders |
| 698 | **ruff** | `F841` | Local variable `digit_results` is assigned to but never used |
| 699 | **ruff** | `F841` | Local variable `trajectory_results` is assigned to but never used |
| 707 | **ruff** | `F541` | f-string without any placeholders |
| 708 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `src\models\archive\ternary_vae_v5_10.py` (16 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 519 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "StateNetV5")  [assignment] |
| 558 | **mypy** | `TYPE` | Need type annotation for "statenet_corrections"  [var-annotated] |
| 688 | **mypy** | `TYPE` | Cannot determine type of "r_A_ema"  [has-type] |
| 689 | **mypy** | `TYPE` | Cannot determine type of "r_B_ema"  [has-type] |
| 696 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_A"  [has-type] |
| 697 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_B"  [has-type] |
| 698 | **mypy** | `TYPE` | Cannot determine type of "prior_sigma"  [has-type] |
| 699 | **mypy** | `TYPE` | Cannot determine type of "curvature"  [has-type] |
| 1122 | **mypy** | `TYPE` | Cannot determine type of "r_A_ema"  [has-type] |
| 1123 | **mypy** | `TYPE` | Cannot determine type of "r_B_ema"  [has-type] |
| 1134 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_A"  [has-type] |
| 1135 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_B"  [has-type] |
| 1161 | **mypy** | `TYPE` | Cannot determine type of "loss_ema"  [has-type] |
| 1164 | **mypy** | `TYPE` | Cannot determine type of "loss_prev"  [has-type] |
| 1165 | **mypy** | `TYPE` | Cannot determine type of "loss_grad_ema"  [has-type] |
| 1166 | **mypy** | `TYPE` | Cannot determine type of "loss_prev"  [has-type] |

### 📄 `scripts\benchmark\run_benchmark.py` (16 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 5 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |
| 13 | **mypy** | `TYPE` | Library stubs not installed for "tabulate"  [import-untyped] |
| 13 | **mypy** | `TYPE` | Hint: "python3 -m pip install types-tabulate" |
| 29 | **mypy** | `TYPE` | Incompatible default for argument "checkpoint_path" (default has type "None", argument has type "str")  [assignment] |
| 29 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 29 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |
| 353 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 356 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 359 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 363 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 366 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 408 | **mypy** | `TYPE` | Incompatible default for argument "output_path" (default has type "None", argument has type "str")  [assignment] |
| 408 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 408 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |
| 413 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Path", variable has type "str")  [assignment] |
| 415 | **mypy** | `TYPE` | "str" has no attribute "parent"  [attr-defined] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\06_learn_codon_mapping.py` (15 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 26 | **ruff** | `F401` | `itertools.combinations` imported but unused |
| 31 | **ruff** | `E402` | Module level import not at top of file |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 33 | **ruff** | `E402` | Module level import not at top of file |
| 34 | **ruff** | `E402` | Module level import not at top of file |
| 35 | **ruff** | `E402` | Module level import not at top of file |
| 36 | **ruff** | `E402` | Module level import not at top of file |
| 216 | **ruff** | `F841` | Local variable `codon_to_idx` is assigned to but never used |
| 412 | **ruff** | `F841` | Local variable `codons` is assigned to but never used |
| 451 | **ruff** | `F541` | f-string without any placeholders |
| 474 | **ruff** | `F841` | Local variable `clusters` is assigned to but never used |
| 508 | **ruff** | `F541` | f-string without any placeholders |
| 512 | **ruff** | `F541` | f-string without any placeholders |
| 533 | **ruff** | `F841` | Local variable `codons` is assigned to but never used |
| 666 | **ruff** | `F841` | Local variable `history` is assigned to but never used |

### 📄 `scripts\visualization\analyze_3adic_structure.py` (15 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 159 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[int]")  [assignment] |
| 160 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[int]")  [assignment] |
| 161 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, Any]", variable has type "list[floating[Any]]")  [assignment] |
| 162 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, Any]", variable has type "list[floating[Any]]")  [assignment] |
| 183 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 184 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 194 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 195 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 219 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 220 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 229 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 230 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 270 | **mypy** | `TYPE` | Need type annotation for "neighbor_dists_A"  [var-annotated] |
| 271 | **mypy** | `TYPE` | Need type annotation for "neighbor_dists_B"  [var-annotated] |
| 493 | **mypy** | `TYPE` | Module has no attribute "tab10"  [attr-defined] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\06_autoantigen_epitope_analysis.py` (14 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 19 | **ruff** | `F401` | `torch.nn` imported but unused |
| 25 | **ruff** | `F401` | `scipy.spatial.distance.pdist` imported but unused |
| 25 | **ruff** | `F401` | `scipy.spatial.distance.squareform` imported but unused |
| 30 | **ruff** | `F401` | `hyperbolic_utils.project_to_poincare` imported but unused |
| 34 | **ruff** | `F401` | `hyperbolic_utils.CodonEncoder` imported but unused |
| 44 | **ruff** | `F811` | Redefinition of unused `ARGININE_CODONS` from line 36 |
| 54 | **ruff** | `F811` | Redefinition of unused `AA_TO_CODON` from line 35 |
| 546 | **ruff** | `F841` | Local variable `script_dir` is assigned to but never used |
| 599 | **ruff** | `F541` | f-string without any placeholders |
| 606 | **ruff** | `F541` | f-string without any placeholders |
| 699 | **ruff** | `F541` | f-string without any placeholders |
| 706 | **ruff** | `F541` | f-string without any placeholders |
| 713 | **ruff** | `F541` | f-string without any placeholders |
| 720 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\07_adelic_analysis.py` (14 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 28 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `F401` | `scipy.spatial.distance.pdist` imported but unused |
| 31 | **ruff** | `F401` | `scipy.spatial.distance.squareform` imported but unused |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 33 | **ruff** | `E402` | Module level import not at top of file |
| 85 | **ruff** | `F541` | f-string without any placeholders |
| 157 | **ruff** | `F541` | f-string without any placeholders |
| 239 | **ruff** | `F541` | f-string without any placeholders |
| 246 | **ruff** | `F541` | f-string without any placeholders |
| 248 | **ruff** | `F541` | f-string without any placeholders |
| 355 | **ruff** | `F841` | Local variable `Z_full` is assigned to but never used |

### 📄 `src\losses\dual_vae_loss.py` (14 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 502 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 503 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 504 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 505 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 506 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 507 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 509 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 510 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 514 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 515 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 523 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 524 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 525 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |
| 526 | **mypy** | `TYPE` | "float" has no attribute "item"  [attr-defined] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\17_proteome_visualizations.py` (13 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 15 | **ruff** | `F401` | `pandas` imported but unused |
| 113 | **ruff** | `F541` | f-string without any placeholders |
| 157 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 177 | **ruff** | `F541` | f-string without any placeholders |
| 212 | **ruff** | `F541` | f-string without any placeholders |
| 244 | **ruff** | `F841` | Local variable `pvals` is assigned to but never used |
| 257 | **ruff** | `F841` | Local variable `scatter` is assigned to but never used |
| 276 | **ruff** | `F541` | f-string without any placeholders |
| 323 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 345 | **ruff** | `F541` | f-string without any placeholders |
| 419 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 432 | **ruff** | `F541` | f-string without any placeholders |
| 470 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\09_train_codon_encoder_3adic.py` (13 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 26 | **ruff** | `E402` | Module level import not at top of file |
| 27 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `F401` | `matplotlib.pyplot` imported but unused |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 32 | **ruff** | `F401` | `scipy.stats.spearmanr` imported but unused |
| 429 | **ruff** | `F541` | f-string without any placeholders |
| 532 | **ruff** | `F841` | Local variable `history` is assigned to but never used |
| 591 | **ruff** | `F541` | f-string without any placeholders |
| 592 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\train.py` (13 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 36 | **ruff** | `F401` | `torch.nn.functional` imported but unused |
| 38 | **ruff** | `F401` | `tqdm.tqdm` imported but unused |
| 39 | **ruff** | `F401` | `numpy` imported but unused |
| 46 | **ruff** | `E402` | Module level import not at top of file |
| 47 | **ruff** | `E402` | Module level import not at top of file |
| 47 | **ruff** | `F401` | `src.losses.CombinedGeodesicLoss` imported but unused |
| 48 | **ruff** | `E402` | Module level import not at top of file |
| 49 | **ruff** | `E402` | Module level import not at top of file |
| 50 | **ruff** | `E402` | Module level import not at top of file |
| 298 | **ruff** | `F841` | Local variable `max_v` is assigned to but never used |
| 639 | **ruff** | `F541` | f-string without any placeholders |
| 1013 | **ruff** | `F541` | f-string without any placeholders |
| 1018 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\analyze_advanced_manifold.py` (13 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 19 | **ruff** | `F401` | `matplotlib.cm` imported but unused |
| 20 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 22 | **ruff** | `F401` | `scipy.spatial.distance.cdist` imported but unused |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 347 | **ruff** | `F541` | f-string without any placeholders |
| 401 | **ruff** | `F541` | f-string without any placeholders |
| 428 | **ruff** | `F841` | Local variable `im3` is assigned to but never used |
| 442 | **ruff** | `F541` | f-string without any placeholders |
| 515 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 545 | **ruff** | `F541` | f-string without any placeholders |
| 576 | **ruff** | `F541` | f-string without any placeholders |
| 628 | **ruff** | `F841` | Local variable `summary` is assigned to but never used |
| 633 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\calabi_yau_projection.py` (13 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 15 | **ruff** | `F401` | `torch.nn.functional` imported but unused |
| 20 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 88 | **ruff** | `F841` | Local variable `N` is assigned to but never used |
| 121 | **ruff** | `F841` | Local variable `N` is assigned to but never used |
| 135 | **ruff** | `F841` | Local variable `roots` is assigned to but never used |
| 172 | **ruff** | `F841` | Local variable `q2` is assigned to but never used |
| 203 | **ruff** | `F841` | Local variable `N` is assigned to but never used |
| 218 | **ruff** | `F841` | Local variable `p3` is assigned to but never used |
| 239 | **ruff** | `F841` | Local variable `N` is assigned to but never used |
| 355 | **ruff** | `F841` | Local variable `scatter` is assigned to but never used |
| 369 | **ruff** | `F841` | Local variable `z_B` is assigned to but never used |
| 382 | **ruff** | `F841` | Local variable `scatter` is assigned to but never used |
| 404 | **ruff** | `F841` | Local variable `operations` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\19_alphafold_structure_mapping.py` (12 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 19 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 101 | **ruff** | `F841` | Local variable `e` is assigned to but never used |
| 288 | **ruff** | `E722` | Do not use bare `except` |
| 345 | **ruff** | `F541` | f-string without any placeholders |
| 348 | **ruff** | `F541` | f-string without any placeholders |
| 396 | **ruff** | `F541` | f-string without any placeholders |
| 397 | **ruff** | `F541` | f-string without any placeholders |
| 689 | **ruff** | `F541` | f-string without any placeholders |
| 755 | **ruff** | `F541` | f-string without any placeholders |
| 756 | **ruff** | `F541` | f-string without any placeholders |
| 757 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\08_find_natural_positions_v5_11_3.py` (12 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 26 | **ruff** | `E402` | Module level import not at top of file |
| 27 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `F401` | `sklearn.cluster.KMeans` imported but unused |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `F401` | `scipy.spatial.distance.pdist` imported but unused |
| 29 | **ruff** | `F401` | `scipy.spatial.distance.squareform` imported but unused |
| 190 | **ruff** | `F541` | f-string without any placeholders |
| 262 | **ruff** | `F841` | Local variable `cluster_mask` is assigned to but never used |
| 263 | **ruff** | `F841` | Local variable `cluster_indices` is assigned to but never used |
| 327 | **ruff** | `F841` | Local variable `valuations` is assigned to but never used |
| 356 | **ruff** | `E741` | Ambiguous variable name: `l` |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\analysis\compare_options.py` (12 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 14 | **ruff** | `F401` | `scipy.stats.pearsonr` imported but unused |
| 20 | **ruff** | `E402` | Module level import not at top of file |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 22 | **ruff** | `E402` | Module level import not at top of file |
| 23 | **ruff** | `E402` | Module level import not at top of file |
| 84 | **ruff** | `F541` | f-string without any placeholders |
| 89 | **ruff** | `F541` | f-string without any placeholders |
| 114 | **ruff** | `F541` | f-string without any placeholders |
| 131 | **ruff** | `F841` | Local variable `target_d` is assigned to but never used |
| 136 | **ruff** | `F541` | f-string without any placeholders |
| 140 | **ruff** | `F541` | f-string without any placeholders |
| 152 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\analyze_3adic_structure.py` (12 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 19 | **ruff** | `F401` | `itertools.combinations` imported but unused |
| 24 | **ruff** | `F401` | `scipy.stats.pearsonr` imported but unused |
| 25 | **ruff** | `F401` | `scipy.spatial.distance.pdist` imported but unused |
| 25 | **ruff** | `F401` | `scipy.spatial.distance.squareform` imported but unused |
| 27 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 33 | **ruff** | `E402` | Module level import not at top of file |
| 34 | **ruff** | `E402` | Module level import not at top of file |
| 309 | **ruff** | `F541` | f-string without any placeholders |
| 438 | **ruff** | `F841` | Local variable `dist_B` is assigned to but never used |
| 651 | **ruff** | `F841` | Local variable `special_ops` is assigned to but never used |
| 658 | **ruff** | `F541` | f-string without any placeholders |
| 661 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\08_alternative_spectral_operators.py` (11 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 26 | **ruff** | `E402` | Module level import not at top of file |
| 27 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `F401` | `scipy.special.gamma` imported but unused |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 259 | **ruff** | `F841` | Local variable `z_product` is assigned to but never used |
| 293 | **ruff** | `F841` | Local variable `n` is assigned to but never used |
| 353 | **ruff** | `F841` | Local variable `n` is assigned to but never used |

### 📄 `src\models\homeostasis.py` (11 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 95 | **mypy** | `TYPE` | Need type annotation for "coverage_history"  [var-annotated] |
| 96 | **mypy** | `TYPE` | Need type annotation for "hierarchy_A_history"  [var-annotated] |
| 97 | **mypy** | `TYPE` | Need type annotation for "hierarchy_B_history"  [var-annotated] |
| 98 | **mypy** | `TYPE` | Need type annotation for "controller_grad_history"  [var-annotated] |
| 99 | **mypy** | `TYPE` | Need type annotation for "Q_history"  [var-annotated] |
| 174 | **mypy** | `TYPE` | Dict entry 3 has incompatible type "str": "float"; expected "str": "bool"  [dict-item] |
| 175 | **mypy** | `TYPE` | Dict entry 4 has incompatible type "str": "float"; expected "str": "bool"  [dict-item] |
| 176 | **mypy** | `TYPE` | Dict entry 5 has incompatible type "str": "list[str]"; expected "str": "bool"  [dict-item] |
| 229 | **mypy** | `TYPE` | Dict entry 3 has incompatible type "str": "float"; expected "str": "bool"  [dict-item] |
| 230 | **mypy** | `TYPE` | Dict entry 4 has incompatible type "str": "float"; expected "str": "bool"  [dict-item] |
| 231 | **mypy** | `TYPE` | Dict entry 5 has incompatible type "str": "list[str]"; expected "str": "bool"  [dict-item] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\hiv\glycan_shield\02_alphafold3_input_generator.py` (10 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 41 | **ruff** | `F541` | f-string without any placeholders |
| 58 | **ruff** | `F541` | f-string without any placeholders |
| 258 | **ruff** | `F541` | f-string without any placeholders |
| 267 | **ruff** | `F541` | f-string without any placeholders |
| 282 | **ruff** | `F541` | f-string without any placeholders |
| 297 | **ruff** | `F541` | f-string without any placeholders |
| 320 | **ruff** | `F541` | f-string without any placeholders |
| 434 | **ruff** | `F541` | f-string without any placeholders |
| 435 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\23_alphafold_visualizations.py` (10 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 24 | **ruff** | `F401` | `typing.List` imported but unused |
| 24 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 164 | **ruff** | `F841` | Local variable `bars1` is assigned to but never used |
| 166 | **ruff** | `F841` | Local variable `bars2` is assigned to but never used |
| 202 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 269 | **ruff** | `F541` | f-string without any placeholders |
| 296 | **ruff** | `F541` | f-string without any placeholders |
| 333 | **ruff** | `F841` | Local variable `bars1` is assigned to but never used |
| 335 | **ruff** | `F841` | Local variable `bars2` is assigned to but never used |
| 603 | **ruff** | `F841` | Local variable `color` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\05_exact_padic_analysis.py` (10 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 33 | **ruff** | `E402` | Module level import not at top of file |
| 34 | **ruff** | `E402` | Module level import not at top of file |
| 35 | **ruff** | `E402` | Module level import not at top of file |
| 36 | **ruff** | `E402` | Module level import not at top of file |
| 71 | **ruff** | `F841` | Local variable `sorted_radii` is assigned to but never used |
| 85 | **ruff** | `E722` | Do not use bare `except` |
| 214 | **ruff** | `E722` | Do not use bare `except` |
| 222 | **ruff** | `E722` | Do not use bare `except` |
| 259 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `src\models\curriculum.py` (10 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 64 | **mypy** | `TYPE` | Need type annotation for "tau_history" (hint: "tau_history: list[<type>] = ...")  [var-annotated] |
| 85 | **mypy** | `TYPE` | Cannot determine type of "tau"  [has-type] |
| 94 | **mypy** | `TYPE` | Cannot determine type of "tau_ema"  [has-type] |
| 137 | **mypy** | `TYPE` | Function "builtins.any" is not valid as a type  [valid-type] |
| 137 | **mypy** | `TYPE` | Perhaps you meant "typing.Any" instead of "any"? |
| 143 | **mypy** | `TYPE` | Function "builtins.any" is not valid as a type  [valid-type] |
| 143 | **mypy** | `TYPE` | Perhaps you meant "typing.Any" instead of "any"? |
| 173 | **mypy** | `TYPE` | Need type annotation for "delta_history" (hint: "delta_history: list[<type>] = ...")  [var-annotated] |
| 174 | **mypy** | `TYPE` | Need type annotation for "radial_loss_history" (hint: "radial_loss_history: list[<type>] = ...")  [var-annotated] |
| 175 | **mypy** | `TYPE` | Need type annotation for "ranking_loss_history" (hint: "ranking_loss_history: list[<type>] = ...")  [var-annotated] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\neurodegeneration\alzheimers\03_tau_vae_trajectory.py` (9 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 23 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 24 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 25 | **ruff** | `F401` | `itertools.combinations` imported but unused |
| 33 | **ruff** | `E402` | Module level import not at top of file |
| 38 | **ruff** | `F401` | `tau_phospho_database.KXGS_MOTIFS` imported but unused |
| 41 | **ruff** | `E402` | Module level import not at top of file |
| 448 | **ruff** | `F541` | f-string without any placeholders |
| 468 | **ruff** | `F541` | f-string without any placeholders |
| 536 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\01_hla_functionomic_analysis.py` (9 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 15 | **ruff** | `F401` | `torch.nn` imported but unused |
| 18 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 25 | **ruff** | `F401` | `hyperbolic_utils.poincare_distance_matrix` imported but unused |
| 27 | **ruff** | `F401` | `hyperbolic_utils.load_hyperbolic_encoder` imported but unused |
| 31 | **ruff** | `F401` | `hyperbolic_utils.HyperbolicCodonEncoder` imported but unused |
| 32 | **ruff** | `F401` | `hyperbolic_utils.CodonEncoder` imported but unused |
| 496 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 543 | **ruff** | `F841` | Local variable `script_dir` is assigned to but never used |
| 548 | **ruff** | `F841` | Local variable `PROJECT_ROOT` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\04_codon_optimizer.py` (9 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 25 | **ruff** | `F401` | `torch.nn` imported but unused |
| 30 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 30 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 217 | **ruff** | `F841` | Local variable `modified_codons` is assigned to but never used |
| 562 | **ruff** | `F541` | f-string without any placeholders |
| 563 | **ruff** | `F541` | f-string without any placeholders |
| 636 | **ruff** | `F841` | Local variable `script_dir` is assigned to but never used |
| 696 | **ruff** | `F541` | f-string without any placeholders |
| 705 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\03_compare_zeta.py` (9 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 16 | **ruff** | `F401` | `urllib.request` imported but unused |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 22 | **ruff** | `E402` | Module level import not at top of file |
| 23 | **ruff** | `E402` | Module level import not at top of file |
| 153 | **ruff** | `F541` | f-string without any placeholders |
| 154 | **ruff** | `F541` | f-string without any placeholders |
| 184 | **ruff** | `F841` | Local variable `zeta_norm` is assigned to but never used |
| 326 | **ruff** | `F841` | Local variable `x_range` is assigned to but never used |
| 404 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\eval\downstream_validation.py` (9 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 19 | **ruff** | `F401` | `torch.nn.functional` imported but unused |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `E402` | Module level import not at top of file |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 33 | **ruff** | `E402` | Module level import not at top of file |
| 33 | **ruff** | `F401` | `src.losses.poincare_distance` imported but unused |
| 157 | **ruff** | `F541` | f-string without any placeholders |
| 216 | **ruff** | `F541` | f-string without any placeholders |
| 331 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\analysis\geometry.py` (9 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 2 | **ruff** | `F401` | `numpy` imported but unused |
| 3 | **ruff** | `F401` | `itertools` imported but unused |
| 4 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 4 | **ruff** | `F401` | `typing.List` imported but unused |
| 54 | **ruff** | `F841` | Local variable `max_delta` is assigned to but never used |
| 71 | **ruff** | `F811` | Redefinition of unused `itertools` from line 3 |
| 71 | **ruff** | `F401` | `itertools` imported but unused |
| 88 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 159 | **ruff** | `F841` | Local variable `S` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\training\trainer.py` (9 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 16 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 18 | **ruff** | `F401` | `sys` imported but unused |
| 177 | **ruff** | `F541` | f-string without any placeholders |
| 184 | **ruff** | `F541` | f-string without any placeholders |
| 195 | **ruff** | `F541` | f-string without any placeholders |
| 205 | **ruff** | `F541` | f-string without any placeholders |
| 208 | **ruff** | `F541` | f-string without any placeholders |
| 301 | **ruff** | `F841` | Local variable `free_bits` is assigned to but never used |
| 529 | **ruff** | `F841` | Local variable `free_bits` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\02_hla_expanded_analysis.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **ruff** | `F401` | `torch.nn` imported but unused |
| 20 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 21 | **ruff** | `F401` | `itertools.permutations` imported but unused |
| 28 | **ruff** | `F401` | `hyperbolic_utils.poincare_distance_matrix` imported but unused |
| 33 | **ruff** | `F401` | `hyperbolic_utils.CodonEncoder` imported but unused |
| 553 | **ruff** | `F401` | `sklearn.manifold.TSNE` imported but unused |
| 577 | **ruff** | `F841` | Local variable `scatter` is assigned to but never used |
| 709 | **ruff** | `F841` | Local variable `script_dir` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\07_citrullination_shift_analysis.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `F401` | `torch.nn` imported but unused |
| 23 | **ruff** | `F401` | `pathlib.Path` imported but unused |
| 30 | **ruff** | `F401` | `hyperbolic_utils.poincare_distance` imported but unused |
| 31 | **ruff** | `F401` | `hyperbolic_utils.project_to_poincare` imported but unused |
| 35 | **ruff** | `F401` | `hyperbolic_utils.CodonEncoder` imported but unused |
| 352 | **ruff** | `F541` | f-string without any placeholders |
| 361 | **ruff** | `F541` | f-string without any placeholders |
| 363 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\12_download_human_proteome.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 16 | **ruff** | `F401` | `gzip` imported but unused |
| 17 | **ruff** | `F401` | `time` imported but unused |
| 19 | **ruff** | `F401` | `typing.List` imported but unused |
| 19 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 20 | **ruff** | `F401` | `io.StringIO` imported but unused |
| 180 | **ruff** | `E731` | Do not assign a `lambda` expression, use a `def` |
| 182 | **ruff** | `E731` | Do not assign a `lambda` expression, use a `def` |
| 414 | **ruff** | `F841` | Local variable `stats` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\04_fast_reverse_search.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 20 | **ruff** | `E402` | Module level import not at top of file |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 22 | **ruff** | `E402` | Module level import not at top of file |
| 23 | **ruff** | `E402` | Module level import not at top of file |
| 179 | **ruff** | `F841` | Local variable `pattern_with_id` is assigned to but never used |
| 251 | **ruff** | `F541` | f-string without any placeholders |
| 271 | **ruff** | `F541` | f-string without any placeholders |
| 353 | **ruff** | `E741` | Ambiguous variable name: `l` |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\02_compute_spectrum.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 20 | **ruff** | `E402` | Module level import not at top of file |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 22 | **ruff** | `E402` | Module level import not at top of file |
| 23 | **ruff** | `E402` | Module level import not at top of file |
| 24 | **ruff** | `E402` | Module level import not at top of file |
| 25 | **ruff** | `E402` | Module level import not at top of file |
| 335 | **ruff** | `F541` | f-string without any placeholders |
| 366 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_purposeful.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `F401` | `torch.nn` imported but unused |
| 38 | **ruff** | `F401` | `src.metrics.compute_ranking_correlation_hyperbolic` imported but unused |
| 140 | **ruff** | `F841` | Local variable `val_losses` is assigned to but never used |
| 321 | **ruff** | `F541` | f-string without any placeholders |
| 337 | **ruff** | `F541` | f-string without any placeholders |
| 382 | **ruff** | `F541` | f-string without any placeholders |
| 390 | **ruff** | `F541` | f-string without any placeholders |
| 424 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\calabi_yau_surface_mesh.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 16 | **ruff** | `F401` | `torch.nn.functional` imported but unused |
| 21 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 23 | **ruff** | `F401` | `scipy.spatial.ConvexHull` imported but unused |
| 25 | **ruff** | `F401` | `scipy.interpolate.griddata` imported but unused |
| 76 | **ruff** | `F841` | Local variable `N` is assigned to but never used |
| 196 | **ruff** | `E722` | Do not use bare `except` |
| 245 | **ruff** | `F841` | Local variable `height` is assigned to but never used |
| 410 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\visualize_ternary_manifold.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 22 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 31 | **ruff** | `E402` | Module level import not at top of file |
| 32 | **ruff** | `E402` | Module level import not at top of file |
| 49 | **ruff** | `F541` | f-string without any placeholders |
| 58 | **ruff** | `F541` | f-string without any placeholders |
| 278 | **ruff** | `F541` | f-string without any placeholders |
| 323 | **ruff** | `F841` | Local variable `surf` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\training\archive\appetitive_trainer.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 20 | **ruff** | `F401` | `sys` imported but unused |
| 153 | **ruff** | `F541` | f-string without any placeholders |
| 438 | **ruff** | `F841` | Local variable `temp_A` is assigned to but never used |
| 439 | **ruff** | `F841` | Local variable `temp_B` is assigned to but never used |
| 440 | **ruff** | `F841` | Local variable `beta_A` is assigned to but never used |
| 441 | **ruff** | `F841` | Local variable `beta_B` is assigned to but never used |
| 496 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\19_comprehensive_ra_ptm_sweep.py` (7 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 27 | **ruff** | `E402` | Module level import not at top of file |
| 200 | **ruff** | `F841` | Local variable `sequence` is assigned to but never used |
| 301 | **ruff** | `F541` | f-string without any placeholders |
| 332 | **ruff** | `F541` | f-string without any placeholders |
| 337 | **ruff** | `F541` | f-string without any placeholders |
| 342 | **ruff** | `F541` | f-string without any placeholders |
| 349 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\22_alphafold3_analysis.py` (7 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 19 | **ruff** | `F401` | `re` imported but unused |
| 396 | **ruff** | `F541` | f-string without any placeholders |
| 409 | **ruff** | `F541` | f-string without any placeholders |
| 415 | **ruff** | `F541` | f-string without any placeholders |
| 439 | **ruff** | `F541` | f-string without any placeholders |
| 453 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\genetic_code\scripts\07_extract_v5_11_3_embeddings.py` (7 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `F401` | `numpy` imported but unused |
| 28 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 99 | **ruff** | `F541` | f-string without any placeholders |
| 189 | **ruff** | `F541` | f-string without any placeholders |
| 194 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\01_extract_embeddings.py` (7 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 19 | **ruff** | `E402` | Module level import not at top of file |
| 20 | **ruff** | `E402` | Module level import not at top of file |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 21 | **ruff** | `F401` | `tqdm.tqdm` imported but unused |
| 23 | **ruff** | `E402` | Module level import not at top of file |
| 24 | **ruff** | `E402` | Module level import not at top of file |
| 253 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_appetitive_vae.py` (7 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 32 | **ruff** | `F401` | `torch.nn` imported but unused |
| 33 | **ruff** | `F401` | `torch.optim` imported but unused |
| 34 | **ruff** | `F401` | `torch.utils.data.Dataset` imported but unused |
| 82 | **ruff** | `F841` | Local variable `test_split` is assigned to but never used |
| 165 | **ruff** | `F541` | f-string without any placeholders |
| 170 | **ruff** | `F541` | f-string without any placeholders |
| 171 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_ternary_v5_9.py` (7 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 14 | **ruff** | `F401` | `torch.nn` imported but unused |
| 15 | **ruff** | `F401` | `torch.optim` imported but unused |
| 16 | **ruff** | `F401` | `torch.nn.functional` imported but unused |
| 23 | **ruff** | `F401` | `datetime.datetime` imported but unused |
| 255 | **ruff** | `F841` | Local variable `val_losses` is assigned to but never used |
| 356 | **ruff** | `F541` | f-string without any placeholders |
| 366 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `src\geometry\poincare.py` (7 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 85 | **mypy** | `TYPE` | Item "None" of "Any | None" has no attribute "dist"  [union-attr] |
| 127 | **mypy** | `TYPE` | Item "None" of "Any | None" has no attribute "projx"  [union-attr] |
| 161 | **mypy** | `TYPE` | Item "None" of "Any | None" has no attribute "expmap"  [union-attr] |
| 191 | **mypy** | `TYPE` | Item "None" of "Any | None" has no attribute "logmap"  [union-attr] |
| 224 | **mypy** | `TYPE` | Item "None" of "Any | None" has no attribute "mobius_add"  [union-attr] |
| 257 | **mypy** | `TYPE` | Item "None" of "Any | None" has no attribute "lambda_x"  [union-attr] |
| 283 | **mypy** | `TYPE` | Item "None" of "Any | None" has no attribute "transp"  [union-attr] |

### 📄 `scripts\analysis\verify_mathematical_proofs.py` (7 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 37 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "Callable[[str, str], Any]")  [assignment] |
| 41 | **mypy** | `TYPE` | Incompatible default for argument "checkpoint_path" (default has type "None", argument has type "str")  [assignment] |
| 41 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 41 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |
| 42 | **mypy** | `TYPE` | Incompatible default for argument "model_config" (default has type "None", argument has type "dict[str, Any]")  [assignment] |
| 42 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 42 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\neurodegeneration\alzheimers\04_tau_combinatorial.py` (6 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 20 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 33 | **ruff** | `F401` | `tau_phospho_database.TAU_EPITOPES` imported but unused |
| 34 | **ruff** | `F401` | `tau_phospho_database.TAU_DOMAINS` imported but unused |
| 35 | **ruff** | `F401` | `tau_phospho_database.KXGS_MOTIFS` imported but unused |
| 38 | **ruff** | `E402` | Module level import not at top of file |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\18_goldilocks_validation.py` (6 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 20 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 22 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 295 | **ruff** | `F841` | Local variable `imm` is assigned to but never used |
| 296 | **ruff** | `F841` | Local variable `sil` is assigned to but never used |
| 330 | **ruff** | `F541` | f-string without any placeholders |
| 403 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\cross_validation.py` (6 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 80 | **ruff** | `F541` | f-string without any placeholders |
| 84 | **ruff** | `F541` | f-string without any placeholders |
| 94 | **ruff** | `F541` | f-string without any placeholders |
| 95 | **ruff** | `F541` | f-string without any placeholders |
| 167 | **ruff** | `F541` | f-string without any placeholders |
| 228 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\02_compute_spectrum_fast.py` (6 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 20 | **ruff** | `E402` | Module level import not at top of file |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 22 | **ruff** | `E402` | Module level import not at top of file |
| 23 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `F841` | Local variable `n` is assigned to but never used |
| 105 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\analyze_v5_5_quality.py` (6 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **ruff** | `F401` | `torch.nn.functional` imported but unused |
| 25 | **ruff** | `E402` | Module level import not at top of file |
| 143 | **ruff** | `F841` | Local variable `targets` is assigned to but never used |
| 179 | **ruff** | `F541` | f-string without any placeholders |
| 237 | **ruff** | `F541` | f-string without any placeholders |
| 337 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\calabi_yau_fibration.py` (6 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 22 | **ruff** | `F401` | `mpl_toolkits.mplot3d.art3d.Line3DCollection` imported but unused |
| 23 | **ruff** | `F401` | `scipy.interpolate.splprep` imported but unused |
| 23 | **ruff** | `F401` | `scipy.interpolate.splev` imported but unused |
| 25 | **ruff** | `F401` | `scipy.spatial.distance.cdist` imported but unused |
| 263 | **ruff** | `E722` | Do not use bare `except` |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\calabi_yau_v58_fibration.py` (6 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 14 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 15 | **ruff** | `F401` | `scipy.spatial.distance.cdist` imported but unused |
| 17 | **ruff** | `F401` | `scipy.ndimage.gaussian_filter` imported but unused |
| 259 | **ruff** | `F841` | Local variable `n_groups` is assigned to but never used |
| 452 | **ruff** | `E722` | Do not use bare `except` |
| 512 | **ruff** | `F841` | Local variable `scatter` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\neurodegeneration\alzheimers\01_tau_phospho_sweep.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `F401` | `typing.List` imported but unused |
| 30 | **ruff** | `E402` | Module level import not at top of file |
| 34 | **ruff** | `F401` | `tau_phospho_database.TAU_DOMAINS` imported but unused |
| 36 | **ruff** | `F401` | `tau_phospho_database.KXGS_MOTIFS` imported but unused |
| 39 | **ruff** | `E402` | Module level import not at top of file |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\03_citrullination_analysis.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 23 | **ruff** | `F401` | `torch.nn` imported but unused |
| 37 | **ruff** | `F401` | `hyperbolic_utils.CodonEncoder` imported but unused |
| 390 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 516 | **ruff** | `F841` | Local variable `script_dir` is assigned to but never used |
| 582 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\09_immunogenicity_analysis_augmented.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 19 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 26 | **ruff** | `E402` | Module level import not at top of file |
| 30 | **ruff** | `F401` | `hyperbolic_utils.poincare_distance_matrix` imported but unused |
| 33 | **ruff** | `F401` | `hyperbolic_utils.ARGININE_CODONS` imported but unused |
| 37 | **ruff** | `E402` | Module level import not at top of file |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\scientific\02_cluster_boundary_3d\generate.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 11 | **ruff** | `F401` | `json` imported but unused |
| 18 | **ruff** | `F401` | `plotly.subplots.make_subplots` imported but unused; consider using `importlib.util.find_spec` to test for availability |
| 23 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 214 | **ruff** | `F401` | `utils.plotting.PALETTE` imported but unused |
| 214 | **ruff** | `F401` | `utils.plotting.save_figure` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\p-adic-genomics\validations\deep_encoder_validation.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 24 | **ruff** | `F401` | `typing.List` imported but unused |
| 24 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 24 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 27 | **ruff** | `F401` | `scipy.spatial.distance.cdist` imported but unused |
| 662 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\04_padic_spectral_analysis.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 22 | **ruff** | `E402` | Module level import not at top of file |
| 23 | **ruff** | `E402` | Module level import not at top of file |
| 24 | **ruff** | `E402` | Module level import not at top of file |
| 102 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_hyperbolic_structure.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 26 | **ruff** | `E402` | Module level import not at top of file |
| 27 | **ruff** | `E402` | Module level import not at top of file |
| 28 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 30 | **ruff** | `E402` | Module level import not at top of file |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_ternary_v5_8.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 13 | **ruff** | `F401` | `torch.nn` imported but unused |
| 14 | **ruff** | `F401` | `torch.optim` imported but unused |
| 21 | **ruff** | `F401` | `datetime.datetime` imported but unused |
| 199 | **ruff** | `F841` | Local variable `val_losses` is assigned to but never used |
| 437 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_ternary_v5_9_1.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 14 | **ruff** | `F401` | `torch.nn` imported but unused |
| 16 | **ruff** | `F401` | `torch.nn.functional` imported but unused |
| 23 | **ruff** | `F401` | `datetime.datetime` imported but unused |
| 336 | **ruff** | `F541` | f-string without any placeholders |
| 372 | **ruff** | `F841` | Local variable `val_loader` is assigned to but never used |

### 📄 `scripts\train\archive\train_ternary_v5_7.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |
| 179 | **mypy** | `TYPE` | Need type annotation for "r_A_history" (hint: "r_A_history: list[<type>] = ...")  [var-annotated] |
| 180 | **mypy** | `TYPE` | Need type annotation for "r_B_history" (hint: "r_B_history: list[<type>] = ...")  [var-annotated] |
| 181 | **mypy** | `TYPE` | Need type annotation for "ranking_weight_history" (hint: "ranking_weight_history: list[<type>] = ...")  [var-annotated] |
| 378 | **mypy** | `TYPE` | Need type annotation for "epoch_losses"  [var-annotated] |

### 📄 `scripts\benchmark\measure_manifold_resolution.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 8 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |
| 81 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment] |
| 85 | **mypy** | `TYPE` | "list[Any]" has no attribute "mean"  [attr-defined] |
| 87 | **mypy** | `TYPE` | "list[Any]" has no attribute "max"  [attr-defined] |
| 88 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |

### 📄 `scripts\benchmark\measure_coupled_resolution.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 8 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |
| 8 | **mypy** | `TYPE` | Hint: "python3 -m pip install types-PyYAML" |
| 8 | **mypy** | `TYPE` | (or run "mypy --install-types" to install all missing stub packages) |
| 8 | **mypy** | `TYPE` | See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports |
| 243 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment] |

### 📄 `scripts\visualization\viz_v59_hyperbolic.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 38 | **mypy** | `TYPE` | Argument "where" to "fill_between" of "Axes" has incompatible type "ndarray[Any, dtype[bool_]]"; expected "Sequence[bool] | None"  [arg-type] |
| 51 | **mypy** | `TYPE` | Argument "xy" to "annotate" of "Axes" has incompatible type "tuple[signedinteger[Any], Any]"; expected "tuple[float, float]"  [arg-type] |
| 51 | **mypy** | `TYPE` | Argument "xytext" to "annotate" of "Axes" has incompatible type "tuple[signedinteger[Any], Any]"; expected "tuple[float, float] | None"  [arg-type] |
| 95 | **mypy** | `TYPE` | Argument "where" to "fill_between" of "Axes" has incompatible type "ndarray[Any, dtype[bool_]]"; expected "Sequence[bool] | None"  [arg-type] |
| 96 | **mypy** | `TYPE` | Argument "where" to "fill_between" of "Axes" has incompatible type "ndarray[Any, dtype[bool_]]"; expected "Sequence[bool] | None"  [arg-type] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\neurodegeneration\alzheimers\05_alphafold3_validation_jobs.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 28 | **ruff** | `F401` | `data.tau_phospho_database.TAU_DOMAINS` imported but unused |
| 29 | **ruff** | `F401` | `data.tau_phospho_database.TAU_PHOSPHO_SITES` imported but unused |
| 30 | **ruff** | `F401` | `data.tau_phospho_database.KXGS_MOTIFS` imported but unused |
| 229 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\05_regenerative_axis_analysis.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 36 | **ruff** | `F401` | `torch.nn` imported but unused |
| 41 | **ruff** | `F401` | `typing.List` imported but unused |
| 50 | **ruff** | `F401` | `hyperbolic_utils.CodonEncoder` imported but unused |
| 610 | **ruff** | `F841` | Local variable `script_dir` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\14_compute_geometric_features.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 159 | **ruff** | `F841` | Local variable `valid_positions` is assigned to but never used |
| 246 | **ruff** | `F541` | f-string without any placeholders |
| 425 | **ruff** | `F841` | Local variable `stats` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\15_predict_immunogenicity.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 19 | **ruff** | `F401` | `pickle` imported but unused |
| 306 | **ruff** | `F541` | f-string without any placeholders |
| 309 | **ruff** | `F541` | f-string without any placeholders |
| 463 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\20_ra_handshake_analysis.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 34 | **ruff** | `E402` | Module level import not at top of file |
| 404 | **ruff** | `F541` | f-string without any placeholders |
| 436 | **ruff** | `F541` | f-string without any placeholders |
| 441 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\pitch\02_hla_risk_charts\generate.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 13 | **ruff** | `F401` | `utils.plotting.get_risk_cmap` imported but unused |
| 14 | **ruff** | `F401` | `utils.data_loader.get_loader` imported but unused |
| 41 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 105 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\pitch\05_goldilocks_radar\generate.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 11 | **ruff** | `F401` | `matplotlib.patches` imported but unused |
| 15 | **ruff** | `F401` | `utils.plotting.add_goldilocks_zones` imported but unused |
| 137 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |
| 232 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\scientific\05_distance_heatmaps\generate.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 12 | **ruff** | `F401` | `seaborn` imported but unused |
| 81 | **ruff** | `F841` | Local variable `scatter` is assigned to but never used |
| 137 | **ruff** | `F841` | Local variable `bars1` is assigned to but never used |
| 139 | **ruff** | `F841` | Local variable `bars2` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_ternary_v5_6.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 14 | **ruff** | `F401` | `torch.nn` imported but unused |
| 15 | **ruff** | `F401` | `torch.optim` imported but unused |
| 16 | **ruff** | `F401` | `torch.utils.data.Dataset` imported but unused |
| 63 | **ruff** | `F841` | Local variable `test_split` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\losses\hyperbolic_prior.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 24 | **ruff** | `F401` | `torch.nn.functional` imported but unused |
| 25 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 35 | **ruff** | `F401` | `src.geometry.GEOOPT_AVAILABLE` imported but unused |
| 149 | **ruff** | `F841` | Local variable `dist_from_origin` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\tests\integration\test_generalization.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 19 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 19 | **ruff** | `F401` | `typing.Dict` imported but unused |
| 20 | **ruff** | `F401` | `itertools` imported but unused |
| 420 | **ruff** | `F841` | Local variable `z_with_statenet` is assigned to but never used |

### 📄 `src\training\config_schema.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 242 | **mypy** | `TYPE` | Incompatible default for argument "defaults" (default has type "None", argument has type "dict[str, Any]")  [assignment] |
| 242 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 242 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |
| 280 | **mypy** | `TYPE` | Need type annotation for "warnings" (hint: "warnings: list[<type>] = ...")  [var-annotated] |

### 📄 `src\training\archive\appetitive_trainer.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 123 | **mypy** | `TYPE` | Need type annotation for "correlation_history" (hint: "correlation_history: list[<type>] = ...")  [var-annotated] |
| 124 | **mypy** | `TYPE` | Need type annotation for "mi_history" (hint: "mi_history: list[<type>] = ...")  [var-annotated] |
| 125 | **mypy** | `TYPE` | Need type annotation for "addition_accuracy_history" (hint: "addition_accuracy_history: list[<type>] = ...")  [var-annotated] |
| 435 | **mypy** | `TYPE` | Need type annotation for "epoch_losses"  [var-annotated] |

### 📄 `src\losses\hyperbolic_prior.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 316 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_ema"  [has-type] |
| 317 | **mypy** | `TYPE` | Cannot determine type of "kl_ema"  [has-type] |
| 325 | **mypy** | `TYPE` | Cannot determine type of "adaptive_sigma"  [has-type] |
| 337 | **mypy** | `TYPE` | Cannot determine type of "adaptive_curvature"  [has-type] |

### 📄 `src\losses\hyperbolic_recon.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 316 | **mypy** | `TYPE` | Cannot determine type of "loss_ema"  [has-type] |
| 317 | **mypy** | `TYPE` | Cannot determine type of "coverage_ema"  [has-type] |
| 326 | **mypy** | `TYPE` | Cannot determine type of "adaptive_geodesic_weight"  [has-type] |
| 339 | **mypy** | `TYPE` | Cannot determine type of "adaptive_radius_power"  [has-type] |

### 📄 `src\losses\registry.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 62 | **mypy** | `TYPE` | By default the bodies of untyped functions are not checked, consider using --check-untyped-defs  [annotation-unchecked] |
| 63 | **mypy** | `TYPE` | By default the bodies of untyped functions are not checked, consider using --check-untyped-defs  [annotation-unchecked] |
| 64 | **mypy** | `TYPE` | By default the bodies of untyped functions are not checked, consider using --check-untyped-defs  [annotation-unchecked] |
| 205 | **mypy** | `TYPE` | Item "None" of "Any | None" has no attribute "device"  [union-attr] |

### 📄 `scripts\train\train.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 33 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |
| 222 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "float", variable has type "None")  [assignment] |
| 223 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "int", variable has type "None")  [assignment] |
| 285 | **mypy** | `TYPE` | Need type annotation for "valuation_groups" (hint: "valuation_groups: dict[<type>, <type>] = ...")  [var-annotated] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\embeddings_analysis\02_visualize_hierarchy.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 65 | **ruff** | `F841` | Local variable `bars1` is assigned to but never used |
| 66 | **ruff** | `F841` | Local variable `bars2` is assigned to but never used |
| 103 | **ruff** | `F841` | Local variable `bars` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\hiv\glycan_shield\01_glycan_sentinel_analysis.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **ruff** | `F401` | `typing.Dict` imported but unused |
| 26 | **ruff** | `F401` | `hyperbolic_utils.poincare_distance` imported but unused |
| 323 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\hiv\scripts\01_hiv_escape_analysis.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 8 | **ruff** | `F401` | `json` imported but unused |
| 26 | **ruff** | `F401` | `hyperbolic_utils.AA_TO_CODON` imported but unused |
| 273 | **ruff** | `F841` | Local variable `wt_aa` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\neurodegeneration\alzheimers\02_tau_mtbr_interface.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 22 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 31 | **ruff** | `E402` | Module level import not at top of file |
| 39 | **ruff** | `E402` | Module level import not at top of file |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\08_augmented_epitope_database.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 20 | **ruff** | `F401` | `typing.List` imported but unused |
| 700 | **ruff** | `F541` | f-string without any placeholders |
| 708 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\scientific\01_hla_pca_projections\generate.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 15 | **ruff** | `F401` | `utils.plotting.PALETTE` imported but unused |
| 16 | **ruff** | `F401` | `utils.data_loader.get_loader` imported but unused |
| 163 | **ruff** | `F841` | Local variable `cbar` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\scientific\04_calabi_yau_manifolds\generate.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `F401` | `mpl_toolkits.mplot3d.Axes3D` imported but unused |
| 273 | **ruff** | `F401` | `utils.plotting.save_figure` imported but unused |
| 341 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\sars_cov_2\glycan_shield\01_spike_sentinel_analysis.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `E402` | Module level import not at top of file |
| 27 | **ruff** | `F401` | `hyperbolic_utils.codon_to_onehot` imported but unused |
| 235 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\sars_cov_2\glycan_shield\02_handshake_interface_analysis.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 24 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 26 | **ruff** | `F401` | `collections.defaultdict` imported but unused |
| 32 | **ruff** | `E402` | Module level import not at top of file |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\sars_cov_2\glycan_shield\03_deep_handshake_sweep.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 19 | **ruff** | `F401` | `itertools.product` imported but unused |
| 25 | **ruff** | `E402` | Module level import not at top of file |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\p-adic-genomics\validations\encoder_cross_validation.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 23 | **ruff** | `F401` | `typing.List` imported but unused |
| 23 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 23 | **ruff** | `F401` | `typing.Optional` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\p-adic-genomics\validations\padic_biology_validation.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 27 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 27 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 528 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\10_semantic_amplification_benchmark.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 318 | **ruff** | `F841` | Local variable `ops_geometric` is assigned to but never used |
| 382 | **ruff** | `F541` | f-string without any placeholders |
| 446 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\spectral_analysis_over_models\scripts\11_variational_orthogonality_test.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 22 | **ruff** | `F401` | `torch.nn` imported but unused |
| 237 | **ruff** | `F841` | Local variable `sample_radii` is assigned to but never used |
| 331 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\demo\interactive_manifold.ipynb` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 2 | **ruff** | `F401` | `numpy` imported but unused |
| 3 | **ruff** | `F401` | `matplotlib.pyplot` imported but unused |
| 5 | **ruff** | `F401` | `src.data.generate_all_ternary_operations` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\analysis\code_stats.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 46 | **ruff** | `F841` | Local variable `duplications` is assigned to but never used |
| 50 | **ruff** | `E741` | Ambiguous variable name: `l` |
| 57 | **ruff** | `E741` | Ambiguous variable name: `l` |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\analysis\verify_mathematical_proofs.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 15 | **ruff** | `F401` | `json` imported but unused |
| 46 | **ruff** | `F541` | f-string without any placeholders |
| 97 | **ruff** | `F841` | Local variable `z_B` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_ternary_v5_5.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 14 | **ruff** | `F401` | `torch.nn` imported but unused |
| 133 | **ruff** | `F541` | f-string without any placeholders |
| 512 | **ruff** | `F841` | Local variable `test_split` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\__init__.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 26 | **ruff** | `E402` | Module level import not at top of file |
| 29 | **ruff** | `E402` | Module level import not at top of file |
| 36 | **ruff** | `E402` | Module level import not at top of file |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\losses\zero_structure.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 20 | **ruff** | `F401` | `typing.Dict` imported but unused |
| 20 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 20 | **ruff** | `F401` | `typing.Tuple` imported but unused |

### 📄 `src\models\archive\ternary_vae_v5_7.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 320 | **mypy** | `TYPE` | Need type annotation for "statenet_corrections"  [var-annotated] |
| 392 | **mypy** | `TYPE` | Cannot determine type of "r_A_ema"  [has-type] |
| 393 | **mypy** | `TYPE` | Cannot determine type of "r_B_ema"  [has-type] |

### 📄 `src\training\hyperbolic_trainer.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 273 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "PAdicRankingLossHyperbolic")  [assignment] |
| 449 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "float", variable has type "None")  [assignment] |
| 460 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "float", variable has type "None")  [assignment] |

### 📄 `src\training\environment.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 126 | **mypy** | `TYPE` | "dict[Any, Any]" has no attribute "log_dir"  [attr-defined] |
| 194 | **mypy** | `TYPE` | Assignment to variable "e" outside except: block  [misc] |
| 195 | **mypy** | `TYPE` | Trying to read deleted variable "e"  [misc] |

### 📄 `scripts\visualization\visualize_ternary_manifold.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 337 | **mypy** | `TYPE` | Argument "extent" to "imshow" of "Axes" has incompatible type "list[Any]"; expected "tuple[float, float, float, float] | None"  [arg-type] |
| 685 | **mypy** | `TYPE` | Argument "extent" to "imshow" of "Axes" has incompatible type "list[Any]"; expected "tuple[float, float, float, float] | None"  [arg-type] |
| 697 | **mypy** | `TYPE` | Argument "extent" to "imshow" of "Axes" has incompatible type "list[Any]"; expected "tuple[float, float, float, float] | None"  [arg-type] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\13_extract_arginine_contexts.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 312 | **ruff** | `F841` | Local variable `stats` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\16_enrichment_analysis.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **ruff** | `F401` | `collections.Counter` imported but unused |
| 422 | **ruff** | `F841` | Local variable `summary` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\22_deep_citrullination_analysis.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 300 | **ruff** | `F541` | f-string without any placeholders |
| 303 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\pitch\01_pathophysiology_funnel\generate.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 11 | **ruff** | `F401` | `matplotlib.patches` imported but unused |
| 13 | **ruff** | `F401` | `numpy` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\pitch\03_intervention_pathways\generate.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 11 | **ruff** | `F401` | `matplotlib.patches` imported but unused |
| 41 | **ruff** | `F841` | Local variable `tier_x` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\utils\__init__.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 2 | **ruff** | `F403` | `from .plotting import *` used; unable to detect undefined names |
| 3 | **ruff** | `F403` | `from .data_loader import *` used; unable to detect undefined names |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\import_test.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 2 | **ruff** | `F401` | `pathlib.Path` imported but unused |
| 5 | **ruff** | `F401` | `src.models.ternary_vae.TernaryVAE` imported but unused; consider using `importlib.util.find_spec` to test for availability |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\benchmark\measure_coupled_resolution.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 11 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 12 | **ruff** | `F401` | `collections.defaultdict` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\benchmark\measure_manifold_resolution.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 11 | **ruff** | `F401` | `typing.Tuple` imported but unused |
| 12 | **ruff** | `F401` | `collections.defaultdict` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\benchmark\run_benchmark.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 14 | **ruff** | `F401` | `typing.List` imported but unused |
| 21 | **ruff** | `F401` | `src.data.TernaryOperationDataset` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\core\ternary.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 35 | **ruff** | `F401` | `typing.Union` imported but unused |
| 36 | **ruff** | `F401` | `functools.lru_cache` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\data\loaders.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 10 | **ruff** | `F401` | `torch.utils.data.Dataset` imported but unused |
| 116 | **ruff** | `F841` | Local variable `base_dataset` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\losses\appetitive_losses.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 320 | **ruff** | `F841` | Local variable `pos_scores` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\losses\radial_stratification.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 16 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 16 | **ruff** | `F401` | `typing.Tuple` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\models\homeostasis.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **ruff** | `F401` | `torch` imported but unused |
| 19 | **ruff** | `F401` | `typing.Tuple` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\models\ternary_vae.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 36 | **ruff** | `F401` | `typing.Optional` imported but unused |
| 39 | **ruff** | `F401` | `.differentiable_controller.ThreeBodyController` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\training\config_schema.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 13 | **ruff** | `F401` | `pathlib.Path` imported but unused |
| 280 | **ruff** | `F841` | Local variable `warnings` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\training\environment.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 22 | **ruff** | `F401` | `.config_schema.TrainingConfig` imported but unused |
| 174 | **ruff** | `F401` | `torch.utils.tensorboard.SummaryWriter` imported but unused; consider using `importlib.util.find_spec` to test for availability |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\training\hyperbolic_trainer.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 513 | **ruff** | `F841` | Local variable `exploration_boosted` is assigned to but never used |
| 568 | **ruff** | `F841` | Local variable `val_losses` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\utils\metrics.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 5 | **ruff** | `F401` | `typing.List` imported but unused |
| 5 | **ruff** | `F401` | `typing.Set` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\utils\ternary_lut.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 23 | **ruff** | `F401` | `typing.Union` imported but unused |
| 144 | **ruff** | `F841` | Local variable `device` is assigned to but never used |

### 📄 `src\losses\consequence_predictor.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 54 | **mypy** | `TYPE` | Need type annotation for "actual_accuracy_history" (hint: "actual_accuracy_history: list[<type>] = ...")  [var-annotated] |
| 55 | **mypy** | `TYPE` | Need type annotation for "predicted_accuracy_history" (hint: "predicted_accuracy_history: list[<type>] = ...")  [var-annotated] |

### 📄 `src\losses\padic_losses.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 364 | **mypy** | `TYPE` | Need type annotation for "hard_anchors" (hint: "hard_anchors: list[<type>] = ...")  [var-annotated] |
| 774 | **mypy** | `TYPE` | Need type annotation for "hard_anchors" (hint: "hard_anchors: list[<type>] = ...")  [var-annotated] |

### 📄 `src\artifacts\checkpoint_manager.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 36 | **mypy** | `TYPE` | Need type annotation for "_queue"  [var-annotated] |
| 175 | **mypy** | `TYPE` | Item "None" of "AsyncCheckpointSaver | None" has no attribute "save_async"  [union-attr] |

### 📄 `src\training\trainer.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 121 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "RadialStratificationLoss")  [assignment] |
| 520 | **mypy** | `TYPE` | Need type annotation for "epoch_losses"  [var-annotated] |

### 📄 `scripts\train\archive\train_appetitive_vae.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 35 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |
| 45 | **mypy** | `TYPE` | Module "src.training" has no attribute "AppetitiveVAETrainer"  [attr-defined] |

### 📄 `scripts\visualization\calabi_yau_v58_fast.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 362 | **mypy** | `TYPE` | Module has no attribute "viridis"  [attr-defined] |
| 375 | **mypy** | `TYPE` | Module has no attribute "plasma"  [attr-defined] |

### 📄 `scripts\visualization\calabi_yau_v58_extended.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 369 | **mypy** | `TYPE` | Cannot call function of unknown type  [operator] |
| 439 | **mypy** | `TYPE` | Module has no attribute "viridis"  [attr-defined] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\hiv\glycan_shield\03_create_batch_json.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 79 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\18_extract_acpa_proteins.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 306 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\21_alphafold3_input_generator.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 20 | **ruff** | `F401` | `typing.Optional` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\scripts\21_ra_alphafold_jobs.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 228 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\pitch\04_safety_comparisons\generate.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 13 | **ruff** | `F401` | `utils.plotting.get_safety_cmap` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\DOCUMENTATION\02_THEORY_AND_RESEARCH\experiments_and_labs\bioinformatics\codon_encoder_research\rheumatoid_arthritis\visualizations\utils\plotting.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 9 | **ruff** | `F401` | `numpy` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\analysis\analyze_zero_structure.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 52 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\analysis\audit_repo.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 93 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\train\archive\train_ternary_v5_7.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 206 | **ruff** | `F541` | f-string without any placeholders |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\calabi_yau_v58_extended.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 302 | **ruff** | `E722` | Do not use bare `except` |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\visualization\calabi_yau_v58_fast.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 234 | **ruff** | `E722` | Do not use bare `except` |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\data\gpu_resident.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 21 | **ruff** | `F401` | `typing.Optional` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\geometry\poincare.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 27 | **ruff** | `F401` | `typing.Union` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\losses\base.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **ruff** | `F401` | `typing.List` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\losses\components.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 14 | **ruff** | `F401` | `torch.nn` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\losses\hyperbolic_recon.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 516 | **ruff** | `F841` | Local variable `batch_size` is assigned to but never used |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\losses\padic_geodesic.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 24 | **ruff** | `F401` | `typing.Optional` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\losses\padic_losses.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 16 | **ruff** | `F401` | `typing.Optional` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\models\hyperbolic_projection.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 20 | **ruff** | `F401` | `typing.Optional` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\observability\coverage.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **ruff** | `F401` | `typing.Any` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\observability\metrics_buffer.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 27 | **ruff** | `F401` | `typing.Any` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\utils\reproducibility.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 11 | **ruff** | `F401` | `typing.Optional` imported but unused |

### 📄 `src\models\ternary_vae.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 250 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "DifferentiableController")  [assignment] |

### 📄 `src\observability\async_writer.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 83 | **mypy** | `TYPE` | Need type annotation for "_queue"  [var-annotated] |

### 📄 `scripts\train\archive\train_ternary_v5_5.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |

### 📄 `src\models\archive\ternary_vae_v5_6.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 274 | **mypy** | `TYPE` | Need type annotation for "statenet_corrections"  [var-annotated] |

### 📄 `src\losses\appetitive_losses.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 192 | **mypy** | `TYPE` | Cannot determine type of "z_history"  [has-type] |

### 📄 `src\training\monitor.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 88 | **mypy** | `TYPE` | Argument 1 to "_setup_file_logging" of "TrainingMonitor" has incompatible type "str | None"; expected "str"  [arg-type] |

### 📄 `scripts\train\archive\train_ternary_v5_9_1.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |

### 📄 `scripts\train\archive\train_ternary_v5_9.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |

### 📄 `scripts\train\archive\train_ternary_v5_8.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 16 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |

### 📄 `scripts\train\archive\train_ternary_v5_6.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |

### 📄 `scripts\train\archive\train_ternary_v5_10.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 18 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |

### 📄 `scripts\train\archive\train_purposeful.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 24 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |

