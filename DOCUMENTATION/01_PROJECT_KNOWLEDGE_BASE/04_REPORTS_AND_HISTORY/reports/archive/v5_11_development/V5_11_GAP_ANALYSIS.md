# V5.11 Gap Analysis: Definitive Fixes

**Doc-Type:** Technical Analysis · Version 2.0 · 2025-12-14

---

## Executive Summary

Deep analysis of `src/`, `configs/`, and `scripts/` reveals **13 gaps** (6 hyperparameter/loss gaps + 7 architectural gaps) preventing proper 3-adic manifold learning. Each gap is mapped to exact code locations with definitive fixes for V5.11.

**Part 1 (Gaps 1-6):** Loss computation and gradient flow
**Part 2 (Gaps 7-13):** Architectural patterns and data pipeline

---

## Gap 1: StateNet Gradient Flow (CRITICAL)

### Problem
StateNet outputs are converted to Python scalars via `.item()`, breaking ALL gradient flow. The 1,068+ parameters never receive gradients.

### Code Location
**File:** `src/models/ternary_vae_v5_10.py`, lines 870-892

```python
# Line 870: BREAKS GRADIENT
corrected_lr = lr * (1 + self.statenet_lr_scale * delta_lr.item())

# Line 874: BREAKS GRADIENT
self.lambda1 = max(0.5, min(0.95, self.lambda1 + self.statenet_lambda_scale * delta_lambda1.item()))

# Line 880: BREAKS GRADIENT
effective_ranking_weight = self.ranking_weight * (
    1 + self.statenet_ranking_scale * delta_ranking_weight.item()
)

# Lines 886-892: All stored as .item() scalars
self.statenet_corrections['delta_lr'].append(delta_lr.item())
self.statenet_corrections['delta_lambda1'].append(delta_lambda1.item())
# ... etc
```

### Root Cause
StateNet corrections modify **external hyperparameters** (Python floats), not tensors in the computation graph. The `.item()` call converts tensor → float, severing the gradient chain.

### Evidence of Problem
StateNet v5 has `attention_head_scales` (line 335-339) that are `nn.Parameter` but **never receive gradients** because their outputs go through `.item()`.

### Definitive Fix for V5.11

**Replace external hyperparameter modification with tensor-based loss weighting:**

```python
class DifferentiableController(nn.Module):
    """V5.11: All outputs are TENSORS participating in loss computation."""

    def __init__(self, input_dim=12, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6)
        )

    def forward(self, batch_stats: torch.Tensor) -> dict:
        """Returns TENSORS, not scalars. Gradients flow!"""
        raw = self.net(batch_stats)
        return {
            'rho': torch.sigmoid(raw[0]) * 0.5,           # Tensor [0, 0.5]
            'beta_A': F.softplus(raw[1]) + 0.5,           # Tensor [0.5, ∞)
            'beta_B': F.softplus(raw[2]) + 0.5,           # Tensor [0.5, ∞)
            'w_geodesic': F.softplus(raw[3]) + 0.1,       # Tensor [0.1, ∞)
            'w_opposition': F.softplus(raw[4]) + 0.1,     # Tensor [0.1, ∞)
            'temp_ratio': torch.sigmoid(raw[5]) + 0.5     # Tensor [0.5, 1.5]
        }

# In loss computation:
def compute_loss(outputs, controller_outputs):
    w = controller_outputs  # ALL TENSORS

    total_loss = (
        w['w_geodesic'] * geodesic_loss +    # Gradient: loss → w → controller
        w['beta_A'] * kl_A +                  # Gradient: loss → w → controller
        w['beta_B'] * kl_B +                  # Gradient: loss → w → controller
        w['w_opposition'] * opposition_loss   # Gradient: loss → w → controller
    )

    return total_loss  # Gradients flow through ALL weights
```

### Verification
```python
# Test gradient flow
loss = compute_loss(outputs, controller_outputs)
grads = torch.autograd.grad(loss, controller.parameters(), retain_graph=True)
assert all(g is not None and g.abs().sum() > 0 for g in grads), "Gradients must flow!"
```

---

## Gap 2: Competing Losses (Ranking vs Radial)

### Problem
`PAdicRankingLossHyperbolic` and `RadialStratificationLoss` optimize for **different objectives** that compete during training.

### Code Locations

**File 1:** `src/losses/padic_losses.py`, lines 504-759
```python
class PAdicRankingLossHyperbolic:
    # Line 738-742: Triplet loss on RELATIVE distances
    violations = d_anchor_pos - d_anchor_neg + hierarchical_margin
    ranking_loss = F.relu(violations).mean()

    # Line 742: Combined with radial (but radial has different target!)
    total_loss = ranking_loss + self.radial_weight * radial_loss
```

**File 2:** `src/losses/radial_stratification.py`, lines 36-159
```python
class RadialStratificationLoss:
    # Line 110-119: Targets ABSOLUTE radii
    target_radius = self.outer_radius - normalized_v * self.radius_range
    loss_per_sample = F.smooth_l1_loss(actual_radius, target_radius, ...)
```

### Root Cause
- **Ranking loss**: Cares about **relative** ordering (d_pos < d_neg)
- **Radial loss**: Cares about **absolute** positions (r = f(valuation))
- These can conflict: radial may push a point toward origin while ranking wants it further out to maintain relative order.

### Evidence of Problem
In `scripts/train/train_hyperbolic_structure.py`, we found:
- With equal weights: radial drifts from -0.25 to +0.05
- Ranking dominates radial because it has more gradient signal (triplet pairs vs single points)

### Definitive Fix for V5.11

**Unified PAdicGeodesicLoss that couples hierarchy and correlation geometrically:**

```python
class PAdicGeodesicLoss(nn.Module):
    """THE KEY INNOVATION: Single loss where hierarchy IS correlation.

    In proper hyperbolic geometry:
    - Two points near origin → small geodesic distance
    - Two points near boundary → large geodesic distance
    - High valuation pairs → must be near origin → small distance

    No competition: distance TARGET encodes both hierarchy and correlation.
    """

    def __init__(self, curvature=1.0, max_target_distance=3.0, valuation_scale=3.0):
        super().__init__()
        self.curvature = curvature
        self.max_target = max_target_distance
        self.valuation_scale = valuation_scale

    def target_distance(self, valuation: torch.Tensor) -> torch.Tensor:
        """Map 3-adic valuation to target hyperbolic distance.

        v_3 = 0 (not divisible by 3) → large target distance
        v_3 = 9 (divisible by 3^9)   → tiny target distance

        This AUTOMATICALLY enforces:
        - High-v pairs close together → both near origin
        - Low-v pairs far apart → both near boundary
        """
        return self.max_target * torch.exp(-valuation / self.valuation_scale)

    def forward(self, z_hyp: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        batch_size = z_hyp.size(0)
        n_pairs = min(1000, batch_size * (batch_size - 1) // 2)

        # Sample pairs
        i_idx = torch.randint(0, batch_size, (n_pairs,), device=z_hyp.device)
        j_idx = torch.randint(0, batch_size, (n_pairs,), device=z_hyp.device)

        # Avoid self-pairs
        same = i_idx == j_idx
        j_idx[same] = (j_idx[same] + 1) % batch_size

        # Actual Poincaré distance
        d_actual = self._poincare_distance(z_hyp[i_idx], z_hyp[j_idx])

        # Target distance from valuation
        diff = torch.abs(batch_indices[i_idx] - batch_indices[j_idx])
        valuation = TERNARY.valuation(diff).float()
        d_target = self.target_distance(valuation)

        # Single unified loss - NO COMPETITION
        return F.smooth_l1_loss(d_actual, d_target)
```

### Why This Works
In hyperbolic space, the **metric itself** couples radial and angular structure:
- `d_poincare(x, y)` depends on BOTH positions AND the geodesic between them
- Two points near origin have small `d_poincare` even if their Euclidean distance is similar to two boundary points
- Forcing `d_actual ≈ d_target` automatically places high-v points near origin

---

## Gap 3: Coverage Regression

### Problem
V5.10 trains both coverage and structure simultaneously, causing coverage to regress from v5.5's 100%.

### Code Location
**File:** `src/training/hyperbolic_trainer.py`, lines 476-662

```python
def train_epoch(self, train_loader, val_loader, epoch):
    # Line 489-505: Coverage evaluation (both VAEs training!)
    unique_A, cov_A = self.base_trainer.monitor.evaluate_coverage(...)

    # Line 527: Base training modifies encoders/decoders
    train_losses = self.base_trainer.train_epoch(...)  # TRAINS EVERYTHING

    # Line 571: Hyperbolic losses computed on z_A, z_B
    hyperbolic_metrics = self._compute_hyperbolic_losses(...)
```

### Root Cause
V5.5 achieved 100% coverage with reconstruction loss alone. V5.10 adds:
- Ranking loss (triplet margin)
- Radial loss (position targets)
- KL loss (hyperbolic prior)
- Centroid loss (cluster formation)

These additional losses **conflict** with reconstruction, causing coverage to drop.

### Evidence
From `scripts/visualization/analyze_v5_5_quality.py` output:
```
Perfect reconstructions: 19683/19683 (100.00%)  # v5.5
```

From v5.10 training logs:
```
Coverage: ~85-95%  # Lost 5-15% coverage
```

### Definitive Fix for V5.11

**Freeze v5.5 encoder as coverage base. Train only hyperbolic projection:**

```python
class FrozenV55Encoder(nn.Module):
    """Frozen coverage base - NEVER TRAINS."""

    def __init__(self, checkpoint_path: Path):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Build encoder architecture (matches v5.5)
        self.encoder = nn.Sequential(
            nn.Linear(9, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, 16)
        self.fc_logvar = nn.Linear(64, 16)

        # Load weights
        self._load_from_checkpoint(checkpoint)

        # FREEZE - This is the key!
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)  # Deterministic, no gradients


class TernaryVAEV5_11(nn.Module):
    """V5.11: Frozen encoder + trainable hyperbolic projection."""

    def __init__(self, v55_checkpoint_path: Path):
        super().__init__()

        # Frozen encoder (100% coverage locked)
        self.frozen_encoder = FrozenV55Encoder(v55_checkpoint_path)

        # Trainable projection (~3K params)
        self.hyperbolic_projection = HyperbolicProjection(latent_dim=16)

        # Trainable controller (~2K params)
        self.controller = DifferentiableController(input_dim=12)
```

### Coverage Guarantee
Since `FrozenV55Encoder.requires_grad = False`, no gradients flow through it. The 100% coverage is **mathematically locked**.

---

## Gap 4: Inverted Radial Hierarchy

### Problem
V5.5 has **positive** radial correlation (+0.24) when it should be **negative** (high valuation → small radius).

### Code Location
**File:** `scripts/visualization/analyze_v5_5_quality.py`, lines 173-181

```python
# Line 176-177: Computes correlation
corr_A_rad, p_A_rad = spearmanr(valuations, radii_A)
corr_B_rad, p_B_rad = spearmanr(valuations, radii_B)

# Output shows +0.24 - WRONG SIGN
# Expected: -0.24 or lower (high v → low radius)
```

### Root Cause
V5.5 uses **Euclidean** reconstruction loss with no radial constraint:
```python
# v5.5 loss (simplified)
loss = cross_entropy(logits, targets) + beta * kl_divergence(mu, logvar)
```

The standard Gaussian prior (`kl_divergence`) prefers points near origin in **Euclidean** sense, but this doesn't map to 3-adic valuation.

### Evidence from v5.5 Analysis
```
Mean radius by 3-adic valuation:
  Valuation | VAE-A radius
  v=0       |    2.24      # Low-v at high radius - OK
  v=4       |    2.35      # Mid-v ALSO at high radius - WRONG
  v=9       |    2.41      # High-v at HIGHEST radius - INVERTED!
```

### Definitive Fix for V5.11

**PAdicGeodesicLoss automatically enforces correct hierarchy:**

```python
def target_distance(self, valuation: torch.Tensor) -> torch.Tensor:
    """High valuation → small target distance → both points near origin."""
    return self.max_target * torch.exp(-valuation / self.valuation_scale)
```

When trained with this loss:
- High-v pairs get small target distance
- To achieve small `d_poincare`, both points must be near origin
- Radial hierarchy emerges from geodesic geometry, not explicit radial loss

**Proof from train_hyperbolic_structure.py:**
After training with radial emphasis:
```
Radial correlation: -0.25  # CORRECT SIGN
```

---

## Gap 5: Euclidean Contamination in Projection

### Problem
Multiple files use a simple division-based projection that doesn't respect hyperbolic geometry.

### Code Locations

**File 1:** `src/losses/hyperbolic_prior.py`, line 66-73
```python
def _project_to_poincare(self, z: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(z, dim=-1, keepdim=True)
    z_hyp = z / (1 + norm) * self.max_norm  # Simple scaling
    return z_hyp
```

**File 2:** `src/losses/padic_losses.py`, line 551-562
```python
def _project_to_poincare(self, z: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(z, dim=1, keepdim=True)
    z_hyp = z / (1 + norm) * self.max_norm  # Same simple scaling
    return z_hyp
```

### Root Cause
This projection:
1. Preserves **directions** (good)
2. Maps far points → boundary (good)
3. But doesn't use **exponential map** (hyperbolic geometry's natural projection)

The exponential map `exp_0(v) = tanh(√c||v||) * v/(√c||v||)` properly maps tangent vectors to the Poincaré ball.

### Evidence of Problem
When using simple projection, all points cluster near boundary (`norm ~ 0.94`):
```
Mean radius: 0.94 (all points near boundary)
Radial spread: 0.02 (no differentiation)
```

### Definitive Fix for V5.11

**Separate direction and radius networks in HyperbolicProjection:**

```python
class HyperbolicProjection(nn.Module):
    """Learn direction and radius INDEPENDENTLY."""

    def __init__(self, latent_dim=16, hidden_dim=64, max_radius=0.95):
        super().__init__()
        self.max_radius = max_radius

        # Direction network: learns ANGULAR structure
        self.direction_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Radius network: learns RADIAL hierarchy
        self.radius_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, z_euclidean: torch.Tensor) -> torch.Tensor:
        # Direction: unit vector (normalized)
        direction = F.normalize(
            z_euclidean + self.direction_net(z_euclidean),
            dim=-1
        )

        # Radius: scalar in [0, max_radius]
        radius = self.radius_net(z_euclidean) * self.max_radius

        # Combine: z_hyp = radius * direction
        return radius * direction
```

### Why This Works
- Direction and radius are **decoupled** - can learn each independently
- `radius_net` can learn: high-v input → low radius output
- `direction_net` can learn: similar inputs → similar directions
- No saturation issue (exp_map pushes everything to boundary)

---

## Gap 6: Direction/Radius Coupling

### Problem
Current projections treat direction and radius as coupled through the norm, preventing independent learning.

### Code Location
**File:** `src/losses/hyperbolic_prior.py`, lines 104-116

```python
def _exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
    """Both direction AND radius come from same vector v."""
    sqrt_c = math.sqrt(self.curvature)
    v_norm = torch.norm(v, dim=-1, keepdim=True)

    # Direction = v / ||v||  (from v)
    # Radius = tanh(√c ||v||)  (from ||v||)
    result = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    return result
```

### Root Cause
In `exp_map`, a single vector `v` determines both:
- **Direction**: `v / ||v||`
- **Radius**: `tanh(√c ||v||)`

This creates coupling: to change radius, you must change `||v||`, which affects direction stability during training.

### Evidence of Problem
From train_hyperbolic_structure.py experiments:
- Using pure exp_map: all points at boundary (radius ~0.99)
- Direction learning works but radius stuck at max

### Definitive Fix for V5.11

**HyperbolicProjection with separate networks** (see Gap 5 fix)

This decouples direction and radius learning:
```
z_hyp = radius_net(z) * normalize(direction_net(z))
         ↑ learns hierarchy   ↑ learns angular structure
```

---

## Gap Summary Table

| Gap | Severity | Location | Root Cause | V5.11 Fix |
|-----|----------|----------|------------|-----------|
| **1. StateNet Gradients** | CRITICAL | `ternary_vae_v5_10.py:870-892` | `.item()` breaks gradients | Tensor weights in loss |
| **2. Competing Losses** | HIGH | `padic_losses.py` + `radial_stratification.py` | Relative vs absolute objectives | Unified PAdicGeodesicLoss |
| **3. Coverage Regression** | HIGH | `hyperbolic_trainer.py:476-662` | Training everything simultaneously | Freeze v5.5 encoder |
| **4. Inverted Hierarchy** | HIGH | v5.5 checkpoint | No radial constraint | Geodesic loss encodes hierarchy |
| **5. Euclidean Projection** | MEDIUM | `hyperbolic_prior.py:66-73` | Simple scaling, not exp_map | Separate direction/radius nets |
| **6. Direction/Radius Coupling** | MEDIUM | `hyperbolic_prior.py:104-116` | Single vector for both | HyperbolicProjection decouples |

---

## V5.11 Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         V5.11 ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input x ──► [FROZEN v5.5 Encoder] ──► z_euclidean (16D)           │
│                    (45,792 params, requires_grad=False)             │
│                    GAP 3 FIX: Coverage locked at 100%               │
│                                                                     │
│              ──► [HyperbolicProjection] ──► z_hyp (Poincaré ball)  │
│                    (~3K params, trainable)                          │
│                    GAP 5+6 FIX: Separate direction_net + radius_net │
│                                                                     │
│              ──► [DifferentiableController]                         │
│                    (~2K params, trainable)                          │
│                    GAP 1 FIX: All outputs are TENSORS               │
│                                                                     │
│              ──► [Unified PAdicGeodesicLoss]                        │
│                    GAP 2+4 FIX: Single loss couples hierarchy       │
│                    + correlation via hyperbolic geometry            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Files to Create for V5.11

| File | Purpose | Fixes Gap |
|------|---------|-----------|
| `src/models/ternary_vae_v5_11.py` | Main model with frozen encoder | 3 |
| `src/losses/padic_geodesic.py` | Unified geodesic loss | 2, 4 |
| `src/models/hyperbolic_projection.py` | Decoupled direction/radius | 5, 6 |
| `src/models/differentiable_controller.py` | Tensor-output controller | 1 |
| `configs/ternary_v5_11.yaml` | Configuration | All |
| `scripts/train/train_v5_11.py` | Training script | All |

---

## Success Metrics

| Metric | V5.5 | V5.10 | V5.11 Target |
|--------|------|-------|--------------|
| Coverage | 100% | ~85-95% | **100%** (frozen) |
| Radial Hierarchy | +0.24 | ~0 | **< -0.3** |
| 3-adic Correlation | 0.62 | ~0.5 | **> 0.8** |
| StateNet Gradient Flow | N/A | **Broken** | **Full** |

---

---

# PART 2: ARCHITECTURAL GAPS (7-13)

---

## Gap 7: Train/Val Split Prevents 100% Coverage

### Problem
`GPUResidentTernaryDataset` uses 80/10/10 split, meaning training sees only 80% of operations.

### Code Location
**File:** `src/data/gpu_resident.py`, lines 43-88

```python
def __init__(self, device='cuda', train_split=0.8, val_split=0.1, seed=42):
    # Line 68-73: Split computation
    n_train = int(train_split * TERNARY.N_OPERATIONS)  # 15,746 operations
    n_val = int(val_split * TERNARY.N_OPERATIONS)      # 1,968 operations
    # test gets: 1,969 operations

    train_perm = perm[:n_train].to(self.device)
    # ...
    self.train_data = self.all_data[train_perm]  # Only 80% of data!
```

### Root Cause
Standard ML practice (train/val/test splits) conflicts with manifold learning goal. We need **all 19,683 operations** to achieve 100% coverage.

### Evidence
From v5.5 training:
- Training sees 15,746 operations
- Evaluation samples from these 15,746
- "100% coverage" means 100% of **training set**, not full manifold

### Definitive Fix for V5.11

**Use full manifold for training (no split):**

```python
class ManifoldDataset:
    """Full manifold dataset - ALL 19,683 operations."""

    def __init__(self, device='cuda'):
        self.all_data = TERNARY.all_ternary(device)
        self.all_indices = TERNARY.all_indices(device)

    def get_batches(self, batch_size=256, shuffle=True):
        n = TERNARY.N_OPERATIONS  # 19,683 - FULL manifold
        if shuffle:
            perm = torch.randperm(n, device=self.all_data.device)
            data = self.all_data[perm]
            indices = self.all_indices[perm]
        else:
            data, indices = self.all_data, self.all_indices

        for start in range(0, n, batch_size):
            yield data[start:start+batch_size], indices[start:start+batch_size]
```

**For V5.11**: Since encoder is frozen, there's no overfitting risk. Use full manifold.

---

## Gap 8: Cross-Injection detach() Breaks Inter-VAE Gradients

### Problem
The Three-Body system's cross-injection uses `.detach()`, preventing gradient flow between VAEs.

### Code Location
**File:** `src/models/ternary_vae_v5_10.py`, lines ~450-460

```python
# Cross-injection with stop-gradient
z_A_injected = (1 - rho) * z_A + rho * z_B.detach()
z_B_injected = (1 - rho) * z_B + rho * z_A.detach()
```

### Root Cause
The `.detach()` was added to prevent "gradient collapse" but it also prevents the Three-Body feedback loop:
- VAE-A receives gradients only from z_A
- VAE-B receives gradients only from z_B
- No inter-VAE learning signal

### Evidence
The Three-Body design intended:
> "Both VAEs actually learn, as well as the StateNet. StateNet is not only a relation, it actually influences and gets influenced back. So the three bodies get into feedback loops where they achieve exploitation of exploration."

But with `.detach()`, there's no feedback loop - just two isolated VAEs.

### Definitive Fix for V5.11

**For V5.11 specifically**: This gap is **mitigated** because:
1. We freeze the encoder (no VAE training at all)
2. We train only the HyperbolicProjection
3. The Three-Body dynamics become irrelevant

**For future versions**: If unfreezing VAEs, use gradient scaling instead of detach:
```python
# Allow gradients but scale them down
z_B_scaled = z_B * 0.1 + z_B.detach() * 0.9
z_A_injected = (1 - rho) * z_A + rho * z_B_scaled
```

---

## Gap 9: Curriculum tau is Buffer, Not Differentiable

### Problem
The curriculum module's `tau` is registered as a buffer and updated via `.item()`, breaking gradient flow.

### Code Location
**File:** `src/models/curriculum.py`, lines 52-99

```python
class ContinuousCurriculumModule(nn.Module):
    def __init__(self, ...):
        # Line 54: tau is a BUFFER, not a parameter
        self.register_buffer('tau', torch.tensor(initial_tau))

    def update_tau(self, delta_curriculum: float):
        # Line 84-88: Updated from SCALAR (no gradients)
        delta = self.tau_scale * delta_curriculum  # delta_curriculum is float!
        new_tau = self.tau + delta
        self.tau = torch.clamp(new_tau, ...)
```

**File:** `src/training/trainer.py`, lines 391-406

```python
# StateNet outputs delta_curriculum
corrections = self.model.apply_statenet_v5_corrections(...)

# Line 406: Extract as scalar - BREAKS GRADIENT
self.curriculum.update_tau(corrections['delta_curriculum'])
```

### Root Cause
Even though StateNet outputs a tensor, `corrections['delta_curriculum']` is extracted via `.item()` before being passed to `update_tau()`.

### Definitive Fix for V5.11

**For V5.11**: Curriculum module is **eliminated** entirely.

The unified PAdicGeodesicLoss handles both radial hierarchy and angular correlation in a single loss. No curriculum blending needed.

```python
# V5.11: No curriculum module
# Instead of: loss = (1-tau) * radial + tau * ranking
# V5.11 uses: loss = geodesic_loss  # Single loss handles both
```

---

## Gap 10: Phase-Scheduled Rho is Hardcoded

### Problem
Cross-injection strength (rho) follows hardcoded epoch-based phases, not adaptive training dynamics.

### Code Location
**File:** `src/models/ternary_vae_v5_10.py`, lines ~380-420

```python
def compute_phase_scheduled_rho(self, epoch: int, phase_4_start: int) -> float:
    # Hardcoded phase boundaries
    if epoch < 40:
        rho = self.rho_min  # Phase 1: 0.1
    elif epoch < 120:
        rho = ...  # Phase 2: ramp to 0.3
    elif epoch < phase_4_start:
        rho = ...  # Phase 3: ramp to 0.7
    else:
        rho = self.rho_max  # Phase 4: 0.7
```

### Root Cause
Phase transitions are epoch-based, not metric-based. Training dynamics are ignored.

### Evidence
If coverage drops at epoch 50, rho continues increasing toward 0.3 regardless. No feedback.

### Definitive Fix for V5.11

**For V5.11**: Rho becomes a **tensor output** from DifferentiableController:

```python
class DifferentiableController(nn.Module):
    def forward(self, batch_stats):
        raw = self.net(batch_stats)
        return {
            'rho': torch.sigmoid(raw[0]) * 0.5,  # TENSOR, adaptive
            # ...
        }

# In forward pass:
rho = controller_outputs['rho']  # Tensor!
z_A_cross = (1 - rho) * z_A + rho * z_B
```

Controller learns optimal rho from training dynamics via gradient descent.

---

## Gap 11: TERNARY Singleton Not Used Consistently

### Problem
Some code reimplements TERNARY operations instead of using the singleton.

### Code Locations

**File 1:** `src/training/trainer.py`, lines 210-227
```python
def _compute_batch_indices(self, batch_data: torch.Tensor) -> torch.Tensor:
    # Reimplements TERNARY.from_ternary()!
    digits = (batch_data + 1).long()
    weights = self._base3_weights.to(batch_data.device)
    indices = (digits * weights).sum(dim=1)
    return indices
```

**File 2:** Various visualization scripts have manual valuation loops

### Root Cause
Legacy code from before TERNARY singleton was created. Not updated during refactoring.

### Definitive Fix for V5.11

**Use TERNARY consistently:**

```python
# Instead of manual computation:
# indices = (digits * weights).sum(dim=1)

# Use singleton:
from src.core import TERNARY
indices = TERNARY.from_ternary(batch_data)
```

This ensures:
- O(1) lookups via LUT
- Single source of truth
- Device caching handled automatically

---

## Gap 12: Dual VAE Roles (Chaotic vs Frozen) Not Enforced

### Problem
VAE-A "chaotic" and VAE-B "frozen" roles are described in documentation but not structurally enforced.

### Code Location
**File:** `src/models/ternary_vae_v5_10.py`

Both encoders have nearly identical architectures:
```python
# VAE-A Encoder
self.encoder_A = nn.Sequential(
    nn.Linear(9, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU()
)

# VAE-B Encoder - SAME STRUCTURE
self.encoder_B = nn.Sequential(
    nn.Linear(9, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU()
)
```

### Root Cause
The "chaotic vs frozen" distinction exists only in:
- Different beta schedules (vae_a.beta_start=0.3 vs vae_b.beta_start=0.0)
- Different temperature schedules
- Documentation

Nothing structurally prevents both VAEs from converging to similar representations.

### Evidence
GAP 6 FIX in trainer.py computes `r_AB` (cross-VAE correlation):
```python
# Measures similarity between VAE-A and VAE-B representations
# High r_AB = redundancy (both VAEs learning same thing)
```

If roles were properly enforced, r_AB should be low. High r_AB indicates role collapse.

### Definitive Fix for V5.11

**For V5.11**: This gap is **mitigated** because:
1. We use only ONE frozen encoder (from v5.5)
2. No dual-VAE structure in V5.11 training
3. The "opposition" comes from the geodesic loss, not VAE competition

**For future versions**: Add structural enforcement:
```python
# VAE-A: High entropy prior, encourages boundary exploration
self.prior_A = HyperbolicPrior(sigma=1.5, target_radius=0.8)

# VAE-B: Low entropy prior, encourages origin anchoring
self.prior_B = HyperbolicPrior(sigma=0.5, target_radius=0.2)
```

---

## Gap 13: Three-Body Feedback Loop is Broken (SYNTHESIS)

### Problem
The Three-Body system (VAE-A, VAE-B, StateNet) was designed for emergent feedback loops, but multiple gaps break this:

1. **StateNet outputs**: `.item()` breaks gradients (Gap 1)
2. **Cross-injection**: `.detach()` breaks inter-VAE gradients (Gap 8)
3. **Curriculum tau**: buffer, not differentiable (Gap 9)
4. **Rho scheduling**: hardcoded, not adaptive (Gap 10)

### Root Cause
Each component was designed in isolation:
- StateNet: observes state, outputs corrections → but corrections are scalars
- Cross-injection: mixes latents → but detached for stability
- Curriculum: blends losses → but buffer, not learned

No end-to-end differentiable path from loss → controller → VAEs.

### Evidence (Intended vs Actual)

**Intended** (from THREE_BODY_REDESIGN.md):
> "The three bodies get into feedback loops where they achieve exploitation of exploration in high quality"

**Actual**:
```
Loss → backward() → VAE gradients
                 ↛ StateNet (broken by .item())
                 ↛ Cross-VAE (broken by .detach())
                 ↛ Curriculum (broken by buffer)
```

### Definitive Fix for V5.11

**Simplified Architecture** that achieves the same goal differently:

```
V5.11 Three-Body Alternative:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Frozen Encoder (v5.5)  ─────────────────────────────────┐  │
│       ↓                                                  │  │
│  HyperbolicProjection  ←─── gradient ←─── PAdicGeodesic  │  │
│       ↓                        ↑                         │  │
│  DifferentiableController ─────┘                         │  │
│       ↓                                                  │  │
│  Tensor weights ───────────────────────→ Loss            │  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Gradient flows:
  Loss → weights → Controller → projection → (frozen encoder blocked)
```

The feedback loop is now:
1. Controller outputs tensor weights
2. Weights multiply loss terms
3. Loss gradient flows through weights
4. Controller parameters update
5. Next forward: controller sees new batch stats
6. Outputs different weights based on learned behavior

This is simpler and **actually differentiable**.

---

## Complete Gap Summary Table (All 13 Gaps)

| Gap | Severity | Category | Location | Root Cause | V5.11 Fix |
|-----|----------|----------|----------|------------|-----------|
| **1. StateNet Gradients** | CRITICAL | Loss/Gradient | `ternary_vae_v5_10.py:870` | `.item()` breaks gradients | Tensor weights |
| **2. Competing Losses** | HIGH | Loss/Gradient | `padic_losses.py` | Relative vs absolute | Unified geodesic |
| **3. Coverage Regression** | HIGH | Loss/Gradient | `hyperbolic_trainer.py` | Train all simultaneously | Freeze encoder |
| **4. Inverted Hierarchy** | HIGH | Loss/Gradient | v5.5 checkpoint | No radial constraint | Geodesic hierarchy |
| **5. Euclidean Projection** | MEDIUM | Loss/Gradient | `hyperbolic_prior.py:66` | Simple scaling | Direction/radius nets |
| **6. Direction/Radius Coupling** | MEDIUM | Loss/Gradient | `hyperbolic_prior.py:104` | Single vector | HyperbolicProjection |
| **7. Train/Val Split** | MEDIUM | Architecture | `gpu_resident.py:68` | 80% split | Full manifold |
| **8. Cross-Injection detach** | HIGH | Architecture | `ternary_vae_v5_10.py:450` | No inter-VAE gradients | Mitigated (frozen) |
| **9. Curriculum Buffer** | MEDIUM | Architecture | `curriculum.py:54` | tau not differentiable | Eliminated |
| **10. Hardcoded Rho** | MEDIUM | Architecture | `ternary_vae_v5_10.py:380` | Epoch-based phases | Controller tensor |
| **11. TERNARY Inconsistent** | LOW | Architecture | `trainer.py:210` | Legacy code | Use singleton |
| **12. Dual VAE Roles** | MEDIUM | Architecture | Model structure | Only hyperparameter diff | Mitigated (single) |
| **13. Three-Body Broken** | HIGH | Architecture | Multiple | No differentiable path | Simplified loop |

---

## Updated V5.11 Architecture (Addressing All 13 Gaps)

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    V5.11 ARCHITECTURE (All Gaps Addressed)                 │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  FULL MANIFOLD (19,683 ops)  ────────────────────────────── [GAP 7 FIX]   │
│       ↓                                                                   │
│  Input x ──► [FROZEN v5.5 Encoder] ──► z_euclidean (16D)                 │
│                    (requires_grad=False)                                  │
│                    [GAP 3, 8, 12 FIX: No dual VAE, no cross-injection]   │
│                                                                           │
│              ──► [HyperbolicProjection] ──► z_hyp (Poincaré ball)        │
│                    direction_net (angular) + radius_net (radial)          │
│                    [GAP 5, 6 FIX: Decoupled learning]                     │
│                                                                           │
│              ──► [DifferentiableController]                               │
│                    Input: batch_stats (tensor)                            │
│                    Output: {rho, beta_A, beta_B, w_geodesic, ...}         │
│                    [GAP 1, 9, 10 FIX: All outputs are TENSORS]            │
│                                                                           │
│              ──► [PAdicGeodesicLoss]                                      │
│                    target = f(valuation) via TERNARY singleton            │
│                    [GAP 2, 4, 11 FIX: Unified loss, consistent API]       │
│                                                                           │
│              ──► Tensor-weighted loss computation                         │
│                    loss = w['w_geodesic'] * geodesic_loss + ...           │
│                    [GAP 13 FIX: End-to-end differentiable]                │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## References

- V5.11 Implementation Plan: `local-reports/V5_11_IMPLEMENTATION_PLAN.md`
- StateNet Redesign: `local-reports/STATENET_REDESIGN.md`
- Three-Body Redesign: `local-reports/THREE_BODY_REDESIGN.md`
- Proof-of-concept: `scripts/train/train_hyperbolic_structure.py`
- V5.5 Quality Analysis: `scripts/visualization/analyze_v5_5_quality.py`
- Core TERNARY Singleton: `src/core/ternary.py`
- GPU Resident Dataset: `src/data/gpu_resident.py`
- Curriculum Module: `src/models/curriculum.py`
