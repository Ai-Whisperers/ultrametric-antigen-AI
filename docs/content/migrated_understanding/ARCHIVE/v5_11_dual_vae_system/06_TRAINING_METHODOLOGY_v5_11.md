# Training Methodology: Phases, Curriculum, and Homeostasis

> [!CAUTION] > **HISTORICAL DOCUMENT (Superseded)**
> This document describes the "Homeostatic Control" and "Aggressive Beta Warmup" methodology (v5.10) which was later found to cause **posterior collapse** ("Anti-Grokking").
>
> For the **Correct Working Methodology** (Cyclical Beta, High LR) established in v5.12, please refer to:
>
> - [12_MASTER_TRAINING_SYNTHESIS.md](12_MASTER_TRAINING_SYNTHESIS.md) (The Pivot)
> - [16_EXPERIMENTAL_METHODOLOGY.md](16_EXPERIMENTAL_METHODOLOGY.md) (Current Best Practices)

**How to train a hyperbolic VAE without it collapsing**

---

## 1. The Challenge

Training VAEs on discrete spaces with geometric constraints is notoriously difficult:

- **Posterior collapse**: All latent codes become identical
- **Mode collapse**: Only a few outputs are generated
- **Geometric instability**: Points fly to the boundary
- **Conflicting objectives**: Reconstruction vs. regularization

Our solution: **Phase-scheduled training with homeostatic control**.

---

## 2. Phase-Scheduled Training

### The Four Phases

```
Epoch:    0        40       49   50       100+
          │        │        │    │        │
          ▼        ▼        ▼    ▼        ▼
     ┌────────┬─────────┬───────┬─────────────────┐
     │Phase 1 │ Phase 2 │ P3    │    Phase 4      │
     │Explore │Consolidate│Disrupt│   Converge    │
     │β-warmup│ Balance  │β-B up │   Stable      │
     └────────┴─────────┴───────┴─────────────────┘
```

### Phase 1: Exploration (Epochs 0-40)

**Goal**: Let VAE-A explore the space freely while building coverage.

```python
# Phase 1 configuration
beta_A: warmup from 0.3 to 0.8  # Gradually increase KL penalty
beta_B: 0.0                      # B is inactive
temperature: 1.0                 # High temperature = exploration
```

**What happens**:

- VAE-A learns to reconstruct all 19,683 operations
- Latent space expands to cover the data
- No geometric constraints yet

### Phase 2: Consolidation (Epochs 40-49)

**Goal**: Balance reconstruction and regularization.

```python
# Phase 2 configuration
beta_A: 0.8 (stable)
beta_B: 0.0 → 0.1 (gentle warmup)
temperature: 0.9 (slightly cooler)
```

**What happens**:

- VAE-A stabilizes
- VAE-B starts to activate
- System finds equilibrium

### Phase 3: Disruption (Epoch 50)

**Goal**: Brief disruption to escape local minima.

```python
# Phase 3 configuration
beta_B: sudden jump to 0.5
temperature: 0.8
```

**What happens**:

- Sudden change shakes the system
- Allows escape from suboptimal configurations
- Brief (1 epoch only)

### Phase 4: Convergence (Epochs 50+)

**Goal**: Stable training toward final solution.

```python
# Phase 4 configuration
beta_A: 0.8 (stable)
beta_B: warmup from 0.5 to 1.0
temperature: gradual decay to 0.7
geometric losses: activated
```

**What happens**:

- Both VAEs contribute
- Geometric losses shape the latent space
- System converges to final configuration

---

## 3. The β-Schedule

### Why β-Annealing?

If β starts high, the model immediately collapses to the prior.
If β stays low, the model ignores the prior entirely.

**Solution**: Start with low β, gradually increase.

```python
def compute_beta(epoch, phase):
    if phase == 1:
        # Linear warmup
        progress = epoch / 40
        return beta_start + progress * (beta_end - beta_start)
    elif phase == 4:
        # Continue warmup for β_B
        progress = (epoch - 50) / 50
        return min(1.0, beta_B_start + progress * 0.5)
```

### The Schedule Visualized

```
β_A:  0.3 ───────────────── 0.8 ═══════════════════════
β_B:  0.0 ════════════════ 0.0 ──── 0.5 ────────── 1.0

      │                     │        │              │
    epoch 0              epoch 40  epoch 50      epoch 100
```

---

## 4. Curriculum Learning

### Concept

Start with easy examples, gradually introduce harder ones.

### In Our Context

**Easy**: Operations with high p-adic valuation (near origin)
**Hard**: Operations with low valuation (complex, near boundary)

```python
class CurriculumSampler:
    def __init__(self, valuations, pace=0.1):
        self.valuations = valuations
        self.pace = pace
        self.max_difficulty = 0  # Start easy

    def sample(self, n, epoch):
        # Gradually increase difficulty
        self.max_difficulty = min(1.0, epoch * self.pace)

        # Filter to allowed difficulty
        allowed_mask = (self.valuations / 9.0) <= self.max_difficulty
        allowed_indices = torch.where(allowed_mask)[0]

        return torch.randint(len(allowed_indices), (n,))
```

### Schedule

```
Epoch 0-10:  Only high-valuation operations (easy)
Epoch 10-30: Medium-valuation operations added
Epoch 30+:   All operations (full difficulty)
```

---

## 5. Homeostatic Control

### What is Homeostasis?

Biological systems maintain stability through feedback loops:

- Too hot → sweat → cool down
- Too cold → shiver → warm up

We apply the same principle to training!

### The Controller

```python
class HomeostaticController:
    def __init__(self):
        self.target_kl = 5.0
        self.target_radius = 0.5
        self.ema_kl = 0.0
        self.ema_radius = 0.0

    def update(self, kl, radius):
        # Exponential moving average
        self.ema_kl = 0.9 * self.ema_kl + 0.1 * kl
        self.ema_radius = 0.9 * self.ema_radius + 0.1 * radius

    def get_adjustments(self):
        adjustments = {}

        # If KL too high, reduce β
        if self.ema_kl > self.target_kl * 1.5:
            adjustments["beta_scale"] = 0.95
        elif self.ema_kl < self.target_kl * 0.5:
            adjustments["beta_scale"] = 1.05

        # If radius too small, reduce radial loss
        if self.ema_radius < self.target_radius * 0.5:
            adjustments["radial_scale"] = 0.9

        return adjustments
```

### Adaptive Parameters

These parameters can be adjusted during training:

- `adaptive_sigma`: Prior spread
- `adaptive_curvature`: Hyperbolic curvature
- `adaptive_lr`: Learning rate

---

## 6. Gradient Balancing

### The Problem

Different losses have different gradient magnitudes:

- Reconstruction: Large gradients
- Geometric losses: Smaller gradients

Without balancing, geometric losses get ignored!

### The Solution

```python
def gradient_balance(grad_norm_A, grad_norm_B):
    """Scale gradients to have similar magnitude."""
    # EMA of gradient norms
    grad_norm_A_ema = 0.9 * grad_norm_A_ema + 0.1 * grad_norm_A
    grad_norm_B_ema = 0.9 * grad_norm_B_ema + 0.1 * grad_norm_B

    # Compute scaling factors
    scale_A = grad_norm_B_ema / (grad_norm_A_ema + 1e-8)
    scale_B = grad_norm_A_ema / (grad_norm_B_ema + 1e-8)

    # Clamp to prevent extreme scaling
    scale_A = torch.clamp(scale_A, 0.5, 2.0)
    scale_B = torch.clamp(scale_B, 0.5, 2.0)

    return scale_A, scale_B
```

---

## 7. The Training Loop

### Complete Training Step

```python
def train_step(batch, epoch):
    # 1. Compute phase
    phase = get_phase(epoch)

    # 2. Get scheduled parameters
    beta_A = compute_beta_A(epoch, phase)
    beta_B = compute_beta_B(epoch, phase)
    temperature = compute_temperature(epoch, phase)

    # 3. Forward pass
    outputs = model(batch["x"])

    # 4. Compute losses
    losses = loss_fn(
        batch["x"],
        outputs,
        batch["indices"],
        beta_A=beta_A,
        beta_B=beta_B,
    )

    # 5. Homeostatic adjustments
    adjustments = homeostatic_controller.get_adjustments()
    for key, scale in adjustments.items():
        losses["loss"] *= scale

    # 6. Backward pass
    optimizer.zero_grad()
    losses["loss"].backward()

    # 7. Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 8. Optimizer step
    optimizer.step()

    # 9. Update homeostatic state
    homeostatic_controller.update(losses["kl"], outputs["radius"])

    return losses
```

---

## 8. Stratified Batch Sampling

### The Problem

Random sampling might over-represent common operations (low valuation) and under-represent rare ones (high valuation).

### The Solution

```python
class StratifiedBatchSampler:
    def __init__(self, valuations, batch_size, n_strata=10):
        # Divide indices by valuation into strata
        self.strata = []
        for v in range(n_strata):
            mask = (valuations == v)
            self.strata.append(torch.where(mask)[0])

    def __iter__(self):
        while True:
            batch = []
            for stratum in self.strata:
                # Sample equally from each stratum
                n = self.batch_size // len(self.strata)
                indices = torch.randint(len(stratum), (n,))
                batch.extend(stratum[indices].tolist())

            yield batch[:self.batch_size]
```

---

## 9. Learning Rate Schedule

### Cosine Annealing with Warm Restarts

```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,      # Initial period
    T_mult=2,    # Period multiplier after each restart
    eta_min=1e-6 # Minimum learning rate
)
```

### Visualization

```
LR:   1e-3  ╲                 ╱╲            ╱╲
             ╲               ╱  ╲          ╱  ╲
              ╲             ╱    ╲        ╱    ╲
               ╲           ╱      ╲______╱      ╲___
                ╲_________╱
      │         │         │              │
    epoch 0   epoch 50  epoch 100     epoch 200

      Period 1    Period 2 (2x longer)   Period 3 (4x)
```

### Why This Works

- High LR at restart: Escape local minima
- Low LR at end of period: Fine-tune current solution
- Increasing periods: Allow longer exploration as training progresses

---

## 10. Checkpointing Strategy

### When to Save

```python
def should_checkpoint(metrics, best_metrics):
    # Save if coverage improved
    if metrics["coverage"] > best_metrics["coverage"]:
        return True, "coverage"

    # Save if geometric metrics improved (without losing coverage)
    if (metrics["coverage"] >= 0.99 * best_metrics["coverage"] and
        metrics["radial_correlation"] < best_metrics["radial_correlation"]):
        return True, "geometry"

    return False, None
```

### What to Save

```python
checkpoint = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_metrics": best_metrics,
    "config": config,
    "rng_state": torch.get_rng_state(),
}
torch.save(checkpoint, path)
```

---

## 11. Key Training Parameters

| Parameter       | Typical Value | Purpose                |
| --------------- | ------------- | ---------------------- |
| `batch_size`    | 256-512       | Larger = more stable   |
| `learning_rate` | 1e-3          | Initial learning rate  |
| `epochs`        | 100-300       | Total training epochs  |
| `beta_A_start`  | 0.3           | Initial β for VAE-A    |
| `beta_A_end`    | 0.8           | Final β for VAE-A      |
| `beta_B_start`  | 0.0           | Initial β for VAE-B    |
| `beta_B_end`    | 1.0           | Final β for VAE-B      |
| `free_bits`     | 0.1           | KL free bits           |
| `max_radius`    | 0.95          | Poincare ball boundary |
| `curvature`     | 1.0-2.0       | Hyperbolic curvature   |

---

## 12. Common Failure Modes

### Posterior Collapse

**Symptom**: All z codes identical
**Cause**: β too high too early
**Fix**: Slower β warmup, free bits

### Boundary Explosion

**Symptom**: All points at ||z|| → 1
**Cause**: Insufficient radial loss
**Fix**: Increase radial_weight, stricter max_radius

### Mode Collapse

**Symptom**: Only a few outputs
**Cause**: Entropy too low, repulsion too weak
**Fix**: Increase entropy_weight, repulsion_weight

### Geometric Loss Ignored

**Symptom**: Good reconstruction, random geometry
**Cause**: Reconstruction gradient dominates
**Fix**: Gradient balancing, increase geometric weights

---

## Summary

The training methodology combines:

1. **Phase scheduling**: Gradual introduction of objectives
2. **β-annealing**: Prevent premature collapse
3. **Curriculum learning**: Easy to hard examples
4. **Homeostatic control**: Automatic stability
5. **Gradient balancing**: Fair optimization
6. **Stratified sampling**: Balanced representation
7. **Cosine restarts**: Escape local minima

All working together to train a stable, well-structured hyperbolic VAE.

---

_Next, we'll see what this training regime discovered in real HIV data._
