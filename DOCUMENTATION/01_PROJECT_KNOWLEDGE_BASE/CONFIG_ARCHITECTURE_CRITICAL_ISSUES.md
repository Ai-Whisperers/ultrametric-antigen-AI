# Configuration Architecture Critical Issues

**Doc-Type:** Critical Technical Debt · Version 1.0 · Updated 2025-12-29 · AI Whisperers

---

## Executive Summary

The configuration system has SEVERE architectural fragmentation. The centralized config system (`src/config/`) was designed but **NEVER INTEGRATED**. Every training script bypasses schema validation, environment variable prefixes are inconsistent across modules, and config structures diverged completely from the schema design.

**Severity: CRITICAL** - This technical debt undermines:
- Configuration validation (typos go undetected)
- Environment consistency (3 different prefix conventions)
- Maintainability (each script has its own config loading)
- Testing (no unified config testing possible)

---

## Issue 1: Schema Completely Bypassed

### Problem Description

The `src/config/schema.py` defines a `TrainingConfig` dataclass with Pydantic-style validation. The `src/config/loader.py` provides `load_config()` that applies this schema. **ZERO scripts use it.**

### Evidence

**Scripts using raw `yaml.safe_load()` (BYPASSING SCHEMA):**

| File | Line | Code |
|------|------|------|
| `scripts/train.py` | 441 | `yaml.safe_load(f)` |
| `scripts/training/train_v5_11_11_homeostatic.py` | 264 | `yaml.safe_load(f)` |
| `scripts/training/train_v5_12.py` | 66 | `yaml.safe_load(f)` |
| `scripts/ARCHIVE/v5_6_era/benchmarks/run_benchmark.py` | 59 | `yaml.safe_load(f)` |

**Scripts using centralized `load_config()`:** NONE (0 scripts)

### Schema vs Reality Mismatch

**What `TrainingConfig` schema expects (src/config/schema.py:174-223):**
```python
@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    # ...
    geometry: GeometryConfig      # curvature, max_radius, latent_dim
    optimizer: OptimizerConfig    # type, learning_rate, weight_decay
    loss_weights: LossWeights     # reconstruction, kl_divergence, ranking, radial
    ranking: RankingConfig        # margin, n_triplets, hard_negative_ratio
    vae_a: VAEConfig              # beta_start, beta_end, temp_start, temp_end
    vae_b: VAEConfig
```

**What actual configs use (configs/v5_12.yaml, configs/ternary.yaml):**
```yaml
model:                    # NOT in schema
  name: TernaryVAEV5_11_PartialFreeze
  latent_dim: 16
  hidden_dim: 64          # Schema has this in GeometryConfig
  # ...
frozen_checkpoint:        # NOT in schema
  path: sandbox-training/checkpoints/v5_5/latest.pt
loss:                     # Different structure than LossWeights
  rich_hierarchy:         # NOT in schema
  radial:                 # Different keys than schema
  geodesic:               # NOT in schema
homeostasis:              # NOT in schema
option_c:                 # NOT in schema
```

### Impact

1. **No validation**: Missing keys, typos, invalid values go undetected
2. **Dead code**: `src/config/loader.py` and `TrainingConfig` are unused
3. **Fragmentation**: Each script re-implements config loading

---

## Issue 2: Environment Variable Prefix Chaos

### Problem Description

Three different modules use three different environment variable prefixes with no consistency.

### Evidence

**Module 1: `src/config/loader.py` - Uses `TVAE_` prefix**

| Location | Line | Environment Variable |
|----------|------|---------------------|
| `loader.py` | 46 | `ENV_PREFIX = "TVAE_"` |
| `loader.py` | 130 | `TVAE_EPOCHS -> epochs` |
| `loader.py` | 131 | `TVAE_BATCH_SIZE -> batch_size` |
| `loader.py` | 132 | `TVAE_LEARNING_RATE -> optimizer.learning_rate` |
| `loader.py` | 133 | `TVAE_GEOMETRY_CURVATURE -> geometry.curvature` |

**Module 2: `src/config/environment.py` - Uses NO prefix (bare names)**

| Location | Line | Environment Variable |
|----------|------|---------------------|
| `environment.py` | 20-24 | `TERNARY_VAE_ENV` (one special case) |
| `environment.py` | 121 | `CHECKPOINT_DIR` (no prefix!) |
| `environment.py` | 122 | `TENSORBOARD_DIR` (no prefix!) |
| `environment.py` | 123 | `LOG_DIR` (no prefix!) |
| `environment.py` | 116 | `LOG_LEVEL` (no prefix!) |
| `environment.py` | 107 | `CUDA_VISIBLE_DEVICES` (standard) |
| `environment.py` | 137 | `PROFILE_MODE` (no prefix!) |

**Module 3: `src/config/paths.py` - Uses `TERNARY_` prefix**

| Location | Line | Environment Variable |
|----------|------|---------------------|
| `paths.py` | 46 | `TERNARY_PROJECT_ROOT` |
| `paths.py` | 75 | `TERNARY_CONFIG_DIR` |
| `paths.py` | 81 | `TERNARY_DATA_DIR` |
| `paths.py` | 93 | `TERNARY_OUTPUT_DIR` |

**Module 4: `src/config/schema.py` - Uses `TVAE_` prefix (matches loader)**

| Location | Line | Environment Variable |
|----------|------|---------------------|
| `schema.py` | 206 | `TVAE_CHECKPOINT_DIR` |
| `schema.py` | 208 | `TVAE_LOG_DIR` |
| `schema.py` | 210 | `TVAE_TENSORBOARD_DIR` |

### Conflict Table

| Purpose | loader.py | environment.py | paths.py | schema.py |
|---------|-----------|----------------|----------|-----------|
| Checkpoint dir | `TVAE_CHECKPOINT_DIR` | `CHECKPOINT_DIR` | - | `TVAE_CHECKPOINT_DIR` |
| Log dir | `TVAE_LOG_DIR` | `LOG_DIR` | - | `TVAE_LOG_DIR` |
| TensorBoard dir | `TVAE_TENSORBOARD_DIR` | `TENSORBOARD_DIR` | - | `TVAE_TENSORBOARD_DIR` |
| Data dir | - | - | `TERNARY_DATA_DIR` | - |
| Project root | - | - | `TERNARY_PROJECT_ROOT` | - |

### Impact

1. **User confusion**: Which prefix to use?
2. **Silent failures**: Setting `TVAE_CHECKPOINT_DIR` won't work if code uses `CHECKPOINT_DIR`
3. **Untestable**: Can't test environment variable handling with conflicting conventions

---

## Issue 3: Path Constants vs Config Paths

### Problem Description

`src/config/paths.py` defines a comprehensive path structure, but configs and scripts use hardcoded legacy paths.

### Evidence

**New path structure (src/config/paths.py):**
```python
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUT_DIR / "models"      # outputs/models/
RUNS_DIR = OUTPUT_DIR / "runs"               # outputs/runs/
LOGS_DIR = OUTPUT_DIR / "logs"               # outputs/logs/
```

**Legacy paths in configs (still in use):**

| File | Line | Legacy Path | Should Be |
|------|------|-------------|-----------|
| `configs/v5_12.yaml` | 173 | `sandbox-training/checkpoints/v5_12` | `outputs/models/v5_12` |
| `configs/v5_12.yaml` | 168 | `runs/v5_12_production` | `outputs/runs/v5_12_production` |
| `configs/v5_12.yaml` | 52 | `sandbox-training/checkpoints/v5_5/latest.pt` | `outputs/models/v5_5/latest.pt` |
| `configs/ternary.yaml` | 26 | `sandbox-training/checkpoints/v5_5/latest.pt` | `outputs/models/v5_5/latest.pt` |

**Scripts mixing old and new:**

| File | Line | Usage |
|------|------|-------|
| `train_v5_12.py` | 46 | `from src.config.paths import CHECKPOINTS_DIR, RUNS_DIR` (imports new) |
| `train_v5_12.py` | 449 | `save_dir = PROJECT_ROOT / config['checkpoints']['save_dir']` (uses legacy from config) |
| `train_v5_12.py` | 454 | `log_dir = RUNS_DIR / ...` (uses new) |
| `train_v5_12.py` | 478 | `frozen_path = PROJECT_ROOT / frozen_cfg.get('path', 'sandbox-training/...')` (hardcoded legacy) |

### Impact

1. **Inconsistent file locations**: Some outputs go to `outputs/`, some to `sandbox-training/`
2. **Migration blocked**: Can't migrate to new structure while configs use old paths
3. **Legacy maintenance**: Must maintain both `sandbox-training/` and `outputs/` directories

---

## Issue 4: Constants Not Used by Configs

### Problem Description

`src/config/constants.py` defines extensive constants for homeostatic control, but configs hardcode different values without referencing these constants.

### Evidence

**Constants defined (src/config/constants.py:191-211):**
```python
HOMEOSTATIC_COVERAGE_FREEZE_THRESHOLD = 0.995
HOMEOSTATIC_COVERAGE_UNFREEZE_THRESHOLD = 1.0
HOMEOSTATIC_COVERAGE_FLOOR = 0.95
HOMEOSTATIC_HIERARCHY_PLATEAU_THRESHOLD = 0.001
HOMEOSTATIC_HIERARCHY_PLATEAU_PATIENCE = 5      # <-- Different!
HOMEOSTATIC_HIERARCHY_PATIENCE_CEILING = 15     # <-- Different!
HOMEOSTATIC_CONTROLLER_GRAD_PATIENCE = 3        # <-- Different!
HOMEOSTATIC_ANNEALING_STEP = 0.005              # <-- Different!
HOMEOSTATIC_WARMUP_EPOCHS = 5
HOMEOSTATIC_HYSTERESIS_EPOCHS = 3
```

**V5.12 config overrides without comment (configs/v5_12.yaml:57-76):**
```yaml
homeostasis:
  coverage_freeze_threshold: 0.995    # Same
  annealing_step: 0.003               # DIFFERENT (0.005 in constants)
  hierarchy_plateau_patience: 7       # DIFFERENT (5 in constants)
  hierarchy_patience_ceiling: 20      # DIFFERENT (15 in constants)
  controller_grad_patience: 5         # DIFFERENT (3 in constants)
```

### Impact

1. **Hidden divergence**: Config values differ from constants without documentation
2. **No single source of truth**: Which is correct - constant or config?
3. **Maintenance burden**: Must update both constants.py and configs

---

## Issue 5: Dead Configuration Code

### Unused Modules and Functions

| Module | What's Dead | Why |
|--------|-------------|-----|
| `src/config/loader.py` | `load_config()` | No script imports it |
| `src/config/loader.py` | `_load_env_vars()` | Never called |
| `src/config/loader.py` | `_validate_config()` | Never called |
| `src/config/loader.py` | `save_config()` | Never called |
| `src/config/schema.py` | `TrainingConfig` | Never instantiated |
| `src/config/schema.py` | `GeometryConfig` | Never instantiated |
| `src/config/schema.py` | `LossWeights` | Never instantiated |
| `src/config/schema.py` | `OptimizerConfig` | Never instantiated |
| `src/config/schema.py` | `RankingConfig` | Never instantiated |
| `src/config/schema.py` | `VAEConfig` | Never instantiated |
| `src/config/environment.py` | `EnvConfig.from_env()` | Never called from training |
| `src/config/environment.py` | `get_env_config()` | Never called from training |

### Evidence

```bash
# Search for load_config usage in scripts
grep -r "from src.config import load_config" scripts/
# Result: (empty - no matches)

grep -r "from src.config.loader import load_config" scripts/
# Result: (empty - no matches)
```

---

## Issue 6: Config Files Structure Inventory

### Active Configs (12 files)

| Config File | Structure Type | Schema Compatible |
|-------------|----------------|-------------------|
| `configs/ternary.yaml` | Custom V5.11 | NO |
| `configs/ternary_fast_test.yaml` | Custom | NO |
| `configs/v5_11_11_homeostatic_ale_device.yaml` | Custom V5.11.11 | NO |
| `configs/v5_11_11_homeostatic_rtx2060s.yaml` | Custom V5.11.11 | NO |
| `configs/v5_12.yaml` | Custom V5.12 | NO |
| `configs/archive/ternary_v5_6.yaml` | Legacy | NO |
| `configs/archive/ternary_v5_7.yaml` | Legacy | NO |
| `configs/archive/ternary_v5_8.yaml` | Legacy | NO |
| `configs/archive/ternary_v5_9.yaml` | Legacy | NO |
| `configs/archive/ternary_v5_9_2.yaml` | Legacy | NO |
| `configs/archive/ternary_v5_10.yaml` | Legacy | NO |
| `configs/archive/appetitive_vae.yaml` | Different model | NO |

**ZERO configs match the TrainingConfig schema.**

---

## Remediation Roadmap

### Phase 1: Document and Freeze (Week 1)
- [x] Create this document
- [ ] Add deprecation warnings to unused code
- [ ] Document which env vars actually work

### Phase 2: Unify Environment Variables (Week 2)
- [ ] Choose ONE prefix: `TERNARY_` (matches project name)
- [ ] Migrate all modules to use `TERNARY_` prefix
- [ ] Update documentation

### Phase 3: Schema Redesign (Week 3-4)
- [ ] Design new schema matching actual config structure
- [ ] Create V5ConfigSchema matching v5.11+ configs
- [ ] Add migration path from current configs

### Phase 4: Integration (Week 5-6)
- [ ] Update all scripts to use `load_config()`
- [ ] Add validation to all training scripts
- [ ] Remove dead code

### Phase 5: Path Migration (Week 7-8)
- [ ] Migrate checkpoints from `sandbox-training/` to `outputs/models/`
- [ ] Update all configs to use new paths
- [ ] Add backwards compatibility for existing checkpoints

---

## File Reference Quick Lookup

### src/config/ Module Structure

```
src/config/
├── __init__.py          # Exports (mostly unused)
├── constants.py         # Constants (partially used)
├── environment.py       # EnvConfig (unused by training)
├── loader.py            # load_config() (UNUSED)
├── paths.py             # Path constants (partially used)
├── schema.py            # TrainingConfig (UNUSED)
└── README.md            # Documentation
```

### Environment Variable Complete Reference

| Variable | Prefix | Module | Line | Purpose |
|----------|--------|--------|------|---------|
| `TERNARY_VAE_ENV` | `TERNARY_` | environment.py | 99 | Environment mode |
| `TERNARY_PROJECT_ROOT` | `TERNARY_` | paths.py | 46 | Project root override |
| `TERNARY_CONFIG_DIR` | `TERNARY_` | paths.py | 75 | Config dir override |
| `TERNARY_DATA_DIR` | `TERNARY_` | paths.py | 81 | Data dir override |
| `TERNARY_OUTPUT_DIR` | `TERNARY_` | paths.py | 93 | Output dir override |
| `CHECKPOINT_DIR` | (none) | environment.py | 121 | Checkpoint dir override |
| `TENSORBOARD_DIR` | (none) | environment.py | 122 | TensorBoard dir override |
| `LOG_DIR` | (none) | environment.py | 123 | Log dir override |
| `LOG_LEVEL` | (none) | environment.py | 116 | Logging level |
| `PROFILE_MODE` | (none) | environment.py | 137 | Enable profiling |
| `CUDA_VISIBLE_DEVICES` | (standard) | environment.py | 107 | GPU selection |
| `TVAE_CHECKPOINT_DIR` | `TVAE_` | schema.py | 206 | Checkpoint dir override |
| `TVAE_LOG_DIR` | `TVAE_` | schema.py | 208 | Log dir override |
| `TVAE_TENSORBOARD_DIR` | `TVAE_` | schema.py | 210 | TensorBoard dir override |
| `TVAE_EPOCHS` | `TVAE_` | loader.py | 130 | Epochs override |
| `TVAE_BATCH_SIZE` | `TVAE_` | loader.py | 131 | Batch size override |
| `TVAE_LEARNING_RATE` | `TVAE_` | loader.py | 132 | Learning rate override |
| `TVAE_GEOMETRY_CURVATURE` | `TVAE_` | loader.py | 133 | Curvature override |

---

## Issue 7: Checkpoint Chaos - 115 Directories, No Organization

### Problem Description

The `sandbox-training/checkpoints/` directory contains **115 checkpoint directories** with:
- No clear naming convention
- No documented purpose for most
- Inconsistent metric storage formats
- Mix of production, experimental, test, and dead checkpoints
- Massive disk space usage with redundant data

### Full Checkpoint Inventory

**Analysis performed:** 2025-12-29 via `scripts/analysis/analyze_all_checkpoints.py`

```
Total directories: 115
  - complete: 102 (have best.pt)
  - crashed: 8 (have epoch files but no best.pt)
  - empty: 4 (no checkpoint files)
  - partial: 1 (have latest.pt only)
```

### Category Breakdown

#### PRODUCTION (24 directories) - Core Model Versions

These are the versioned model checkpoints. Most have similar metrics but different training approaches.

| Checkpoint | Status | Coverage | Hierarchy_B | Richness | Notes |
|------------|--------|----------|-------------|----------|-------|
| `v5_5` | complete | N/A | N/A | N/A | **FOUNDATION** - 100% coverage, frozen for V5.11+ |
| `v5_6` | complete | N/A | N/A | N/A | Legacy |
| `v5_7` | complete | N/A | N/A | N/A | Legacy |
| `v5_8` | complete | N/A | N/A | N/A | Legacy |
| `v5_9` | complete | N/A | N/A | N/A | Legacy |
| `v5_9_2` | complete | N/A | N/A | N/A | Legacy |
| `v5_10` | **partial** | N/A | N/A | N/A | **INCOMPLETE** - training crashed |
| `v5_11` | complete | 100% | -0.8302 | N/A | Base V5.11 |
| `v5_11_annealing` | complete | 100% | -0.8318 | N/A | |
| `v5_11_homeostasis` | complete | 99.9% | -0.8318 | N/A | **RECOMMENDED** for hierarchy |
| `v5_11_learnable` | complete | 100% | -0.8295 | N/A | |
| `v5_11_npairs4k` | complete | 100% | -0.8314 | N/A | |
| `v5_11_npairs8300` | complete | 100% | -0.8312 | N/A | |
| `v5_11_npairs8600` | complete | 100% | -0.8313 | N/A | |
| `v5_11_npairs8k` | complete | 100% | -0.8313 | N/A | |
| `v5_11_npairs9k` | complete | 100% | -0.8315 | N/A | |
| `v5_11_progressive` | complete | 99.9% | -0.8299 | N/A | |
| `v5_11_radial05` | complete | 100% | -0.8310 | N/A | |
| `v5_11_radial09` | complete | 100% | -0.8309 | N/A | |
| `v5_11_radial1` | complete | 100% | -0.8315 | N/A | |
| `v5_11_repro` | complete | 100% | -0.8315 | N/A | |
| `v5_11_structural` | complete | 100% | -0.8320 | N/A | **BEST HIERARCHY** |
| `v5_11_thresh725` | complete | 100% | -0.8313 | N/A | |
| `v5_11_validation` | complete | 100% | -0.8308 | N/A | |

**PROBLEM:** 24 "production" checkpoints with nearly identical metrics (-0.829 to -0.832 hierarchy). Why keep all?

#### HOMEOSTATIC_EXPERIMENT (1 directory) - Best Balance Found

| Checkpoint | Status | Coverage | Hierarchy_B | Richness | Notes |
|------------|--------|----------|-------------|----------|-------|
| `homeostatic_rich` | complete | 100% | -0.6944 | 0.006615 | **BEST RICHNESS+HIERARCHY BALANCE** |

**This is the checkpoint the CLAUDE.md recommends!** But it's buried among 114 others.

#### LOSS_EXPERIMENT (11 directories) - Hierarchy vs Richness Tradeoffs

| Checkpoint | Status | Coverage | Hierarchy_B | Richness | Notes |
|------------|--------|----------|-------------|----------|-------|
| `balanced_radial` | complete | 100% | -0.8321 | 0.000048 | Ceiling hierarchy, collapsed richness |
| `final_rich_lr1e4` | complete | 100% | -0.6840 | 0.006825 | Good richness |
| `final_rich_lr3e4` | complete | 100% | -0.6691 | 0.008205 | **HIGHEST RICHNESS** |
| `final_rich_lr5e5` | complete | 100% | -0.6932 | 0.008583 | High richness |
| `hierarchy_extreme` | complete | 100% | -0.8321 | N/A | Ceiling hierarchy |
| `hierarchy_focused` | complete | 100% | -0.8320 | N/A | |
| `max_hierarchy` | complete | 100% | -0.8298 | 0.000265 | Near-collapsed richness |
| `radial_collapse` | complete | 100% | -0.8321 | N/A | Ceiling hierarchy |
| `radial_snapped` | complete | 100% | N/A | N/A | No metrics stored |
| `radial_target` | complete | 100% | -0.8321 | N/A | |
| `soft_radial` | complete | 100% | N/A | N/A | No metrics stored |

#### SWEEP_TEST (45 directories) - Hyperparameter Searches

**MASSIVE REDUNDANCY:** 45 directories from various sweeps.

| Sweep Group | Count | Status | Notes |
|-------------|-------|--------|-------|
| `stability_run_*` | 10 | complete | Similar metrics across all |
| `stable_run_*` | 10 | complete | Similar metrics across all |
| `sweep2_*` | 5 | **crashed** | All 5 crashed - DEAD |
| `sweep3_*` | 6 | complete | LR schedule experiments |
| `sweep4_*` | 8 | complete | LR value experiments |
| `sweep_curv_*` | 3 | complete | Curvature experiments |
| `sweep_latent_*` | 3 | 1 complete, 2 crashed | Latent dim experiments |

**Metrics range across sweeps:**
- Coverage: 99.9% (all similar)
- Hierarchy_B: -0.65 to -0.79 (moderate variance)
- Richness: 0.001 to 0.004 (all low)
- Q: 0.99 to 1.11 (all similar)

#### TEST (12 directories) - Validation & Debugging

| Checkpoint | Status | Coverage | Hierarchy_B | Notes |
|------------|--------|----------|-------------|-------|
| `adamw_test` | complete | 100% | -0.8317 | Optimizer test |
| `base_trainer_e2e_test` | complete | 100% | -0.8320 | |
| `base_trainer_test` | complete | 100% | -0.5115 | **LOW HIERARCHY** |
| `geometry_validation` | complete | 100% | -0.6590 | |
| `learnable_curvature_test` | complete | 100% | -0.6925 | |
| `max_radius_096_test` | complete | 100% | -0.6516 | |
| `max_radius_097_test` | complete | 100% | -0.6538 | |
| `refactor_validation` | complete | 100% | -0.6937 | |
| `riemannian_test` | complete | 100% | -0.8316 | |
| `v5_11_11_test` | complete | 100% | -0.8318 | |
| `v5_11_12_validation` | complete | 100% | -0.8320 | |
| `v5_11_9_test` | complete | 100% | -0.8184 | |

#### TRAINING_EXPERIMENT (6 directories) - Progressive/Annealing Tests

| Checkpoint | Status | Coverage | Hierarchy_B | Notes |
|------------|--------|----------|-------------|-------|
| `progressive_conservative` | complete | **0.1%** | -0.8320 | **FAILED** - coverage collapse |
| `progressive_tiny_lr` | complete | **60.4%** | -0.8320 | **FAILED** - coverage collapse |
| `v5_11_annealing_long` | complete | **98.2%** | -0.8320 | Slight coverage loss |
| `v5_11_learnable_qreg` | complete | 100% | -0.8303 | Good |
| `v5_11_progressive_50ep` | complete | **1.3%** | -0.8320 | **FAILED** - coverage collapse |
| `v5_11_progressive_non_fixed` | complete | **0.3%** | -0.8320 | **FAILED** - coverage collapse |

**4 of 6 are FAILED experiments with collapsed coverage!**

#### FINAL_PUSH (6 directories) - Recent Attempts

| Checkpoint | Status | Coverage | Hierarchy_B | Richness | Notes |
|------------|--------|----------|-------------|----------|-------|
| `final_homeo_lr1e3` | complete | 99.9% | -0.6813 | 0.001924 | |
| `final_homeo_lr3e4` | **empty** | N/A | N/A | N/A | **NO CHECKPOINTS** |
| `final_homeo_lr5e4` | complete | 99.9% | -0.6770 | 0.002229 | |
| `scratch_run_1` | **empty** | N/A | N/A | N/A | **NO CHECKPOINTS** |
| `scratch_run_2` | **empty** | N/A | N/A | N/A | **NO CHECKPOINTS** |
| `scratch_run_3` | **empty** | N/A | N/A | N/A | **NO CHECKPOINTS** |

**4 of 6 are empty or incomplete!**

#### OTHER (10 directories) - Miscellaneous

| Checkpoint | Status | Coverage | Hierarchy_B | Notes |
|------------|--------|----------|-------------|-------|
| `appetitive` | complete | N/A | N/A | Different model type |
| `hyperbolic_structure` | **crashed** | N/A | N/A | Training crashed |
| `purposeful` | complete | N/A | N/A | No metrics |
| `purposeful_v5.10` | complete | N/A | N/A | No metrics |
| `purposeful_v5.6` | complete | N/A | N/A | No metrics |
| `ternary` | complete | 100% | -0.8320 | Generic name |
| `v5_11_11_production` | complete | 100% | -0.6926 | Moderate hierarchy |
| `v5_11_9_homeo_zero` | complete | 99.5% | -0.7457 | |
| `v5_11_9_zero` | complete | 100% | -0.7583 | |
| `v5_11_epsilon_coupled` | complete | 100% | **+0.0018** | **INVERTED HIERARCHY!** |

### Metric Storage Format Chaos

Different checkpoints store metrics in different formats:

| Format | Example Checkpoints | Structure |
|--------|---------------------|-----------|
| Format 1: Direct dict | `v5_11_homeostasis` | `ckpt['metrics'] = {'coverage': 0.999, ...}` |
| Format 2: Separate keys | `v5_11_structural` | `ckpt['coverage'] = 0.999; ckpt['hierarchy'] = -0.83` |
| Format 3: eval_metrics | `sweep3_*` | `ckpt['eval_metrics'] = {...}` |
| Format 4: No metrics | `v5_5`, `v5_6`, `appetitive` | Metrics not stored at all |
| Format 5: radial_corr | Some v5.11 | `ckpt['radial_corr_A']`, `ckpt['radial_corr_B']` |

**No standardization = impossible to compare programmatically!**

### Summary Statistics

```
USEFUL checkpoints (production-ready):
  - v5_5 (foundation, 100% coverage)
  - v5_11_homeostasis (best hierarchy)
  - v5_11_structural (best hierarchy)
  - homeostatic_rich (best balance)
  - final_rich_lr3e4 (best richness)
  Total: 5 checkpoints

DEAD/FAILED checkpoints:
  - crashed: 8
  - empty: 4
  - coverage collapsed: 4
  - inverted hierarchy: 1
  Total: 17 checkpoints

REDUNDANT/EXPERIMENTAL:
  - sweep tests: 45 (keep 1-2 best)
  - production duplicates: ~20 (nearly identical metrics)
  - test checkpoints: 12 (delete after validation)
  Total: ~77 checkpoints

RECOMMENDATION: Keep ~10, archive/delete 105
```

### Disk Space Analysis

Each checkpoint directory contains:
- `best.pt`: ~5-10 MB
- `latest.pt`: ~5-10 MB
- `epoch_*.pt`: ~5-10 MB each (10-30 files per directory)

**Estimated total:** 115 dirs × ~100 MB avg = **~11.5 GB** of checkpoints

**After cleanup:** ~10 dirs × ~100 MB = **~1 GB** needed

---

## Issue 8: No Checkpoint Metadata Standard

### Problem Description

Checkpoints lack standardized metadata making it impossible to:
1. Understand what training produced this checkpoint
2. Compare checkpoints programmatically
3. Track lineage (which checkpoint was used as base)

### Required Metadata (not present in most checkpoints)

```python
# What SHOULD be in every checkpoint:
{
    'version': '5.12',
    'created_at': '2025-12-29T10:00:00',
    'training_script': 'scripts/training/train_v5_12.py',
    'config_path': 'configs/v5_12.yaml',
    'config_hash': 'abc123...',  # For reproducibility
    'base_checkpoint': 'sandbox-training/checkpoints/v5_5/latest.pt',
    'epoch': 150,
    'metrics': {
        'coverage': 1.0,
        'hierarchy_A': -0.51,
        'hierarchy_B': -0.83,
        'richness': 0.00787,
        'dist_corr': 0.65,
        'Q': 1.8,
        'r_v0': 0.89,
        'r_v9': 0.12,
    },
    'architecture': {
        'model_class': 'TernaryVAEV5_11_PartialFreeze',
        'latent_dim': 16,
        'hidden_dim': 64,
        'dual_projection': True,
        'use_controller': True,
    },
    'training_summary': {
        'total_epochs': 200,
        'final_loss': 0.123,
        'training_time_hours': 2.5,
    },
}
```

### What's Actually Stored (varies wildly)

| Checkpoint | Has epoch | Has metrics | Has config | Has architecture |
|------------|-----------|-------------|------------|------------------|
| `v5_5` | Yes | **NO** | **NO** | **NO** |
| `v5_11_homeostasis` | Yes | Partial | **NO** | **NO** |
| `homeostatic_rich` | Yes | Yes | Partial | **NO** |
| `sweep3_*` | Yes | Yes | Partial | **NO** |
| Most others | Varies | **NO** | **NO** | **NO** |

---

## Recommended Checkpoint Structure (For V5.12+)

### Keep (5 checkpoints)
1. `v5_5/best.pt` - Foundation (100% coverage)
2. `v5_11_homeostasis/best.pt` - Best hierarchy (-0.8318)
3. `v5_11_structural/best.pt` - Best hierarchy alternative
4. `homeostatic_rich/best.pt` - Best balance (hierarchy + richness)
5. `final_rich_lr3e4/best.pt` - Best richness (0.008205)

### Archive (move to cold storage)
- All `v5_6` through `v5_10` (superseded)
- All `sweep*` directories (keep summary JSONs only)
- All `*_test` directories
- All duplicate v5_11 variants

### Delete (after verification)
- All `*_run_*` numbered directories
- Empty directories
- Crashed training directories
- Failed experiments (coverage < 90%)

---

## Version History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-12-29 | 1.0 | AI Whisperers | Initial critical issue documentation |
| 2025-12-29 | 1.1 | AI Whisperers | Added checkpoint chaos analysis (Issues 7-8) |
