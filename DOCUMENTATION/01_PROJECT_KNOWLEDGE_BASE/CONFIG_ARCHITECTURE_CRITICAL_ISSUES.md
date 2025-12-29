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

## Version History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-12-29 | 1.0 | AI Whisperers | Initial critical issue documentation |
