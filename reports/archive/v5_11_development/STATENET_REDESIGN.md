# StateNet Redesign: Differentiable Control

**Doc-Type:** Architecture Proposal · Version 1.0 · 2025-12-14

---

## The Core Problem

Current StateNet outputs **scalars** that modify **external hyperparameters**:
```python
optimizer.lr = lr * (1 + delta_lr.item())  # External to graph
self.lambda1 = self.lambda1 + delta.item()  # Python float
```

No gradients flow because these operations are outside the computation graph.

---

## The Key Insight

**Move control signals INTO the forward pass as tensor operations.**

Instead of modifying external hyperparameters, StateNet outputs become **multipliers within the loss computation**:

```python
# OLD (broken):
self.lambda1 = self.lambda1 + delta_lambda1.item()  # No gradient
total_loss = lambda1 * loss_A + ...  # lambda1 is Python float

# NEW (differentiable):
control = self.controller(batch_features)  # Tensor output
weight_A = F.softplus(control[0])  # Tensor, bounded positive
total_loss = weight_A * loss_A + ...  # Gradient flows through weight_A!
```

---

## What Can Be Differentiably Controlled

| Parameter | Current | Differentiable Version |
|-----------|---------|----------------------|
| **Loss weights** (λ1, λ2, λ3) | Python floats | Tensor multipliers on loss terms |
| **Cross-injection ρ** | Python float | Tensor in `(1-ρ)*z_A + ρ*z_B` |
| **Temperature** | Division by scalar | Division by tensor: `logits / temp` |
| **Beta (KL weight)** | Scalar | Tensor multiplier: `beta * kl_loss` |
| **Learning rate** | Optimizer param | **Cannot** - truly external |

---

## Proposed Architecture: DifferentiableController

```python
class DifferentiableController(nn.Module):
    """Lightweight controller that learns to modulate training dynamics.

    Key difference from StateNet: ALL outputs participate in tensor
    operations, so gradients flow back and the controller learns.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        # Input: current batch statistics (computed as tensors)
        # - H_A, H_B: entropies
        # - kl_A, kl_B: KL divergences
        # - loss_ratio: recon_A / recon_B
        # - etc.
        input_dim = 8  # Keep it simple

        # Small network - we want this to be lightweight
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # 5 control signals
        )

        # Output indices:
        # 0: rho (cross-injection strength)
        # 1: weight_recon (reconstruction importance)
        # 2: weight_kl (KL importance)
        # 3: weight_ranking (ranking loss importance)
        # 4: weight_entropy_align (entropy alignment importance)

    def forward(self, batch_stats: torch.Tensor) -> dict:
        """
        Args:
            batch_stats: [H_A, H_B, kl_A, kl_B, recon_A, recon_B, ...]
                        All tensors from current batch (gradients intact)

        Returns:
            Dict of control signals (all tensors, gradients flow)
        """
        raw = self.net(batch_stats)

        return {
            'rho': torch.sigmoid(raw[0]) * 0.5,  # [0, 0.5]
            'weight_recon': F.softplus(raw[1]) + 0.5,  # [0.5, ∞)
            'weight_kl': F.softplus(raw[2]) * 0.5 + 0.1,  # [0.1, ∞)
            'weight_ranking': F.softplus(raw[3]),  # [0, ∞)
            'weight_entropy': F.softplus(raw[4]) * 0.3,  # [0, ∞)
        }
```

---

## Integration in Forward Pass

```python
def forward(self, x, compute_control=True):
    # Encode
    mu_A, logvar_A = self.encoder_A(x)
    mu_B, logvar_B = self.encoder_B(x)

    # Sample
    z_A = self.reparameterize(mu_A, logvar_A)
    z_B = self.reparameterize(mu_B, logvar_B)

    if compute_control and self.controller is not None:
        # Compute batch statistics (all tensors!)
        batch_stats = torch.stack([
            self.compute_entropy(logits_A).mean(),
            self.compute_entropy(logits_B).mean(),
            kl_A.mean(),
            kl_B.mean(),
            # ... other statistics
        ])

        # Get control signals (all tensors - gradients flow!)
        control = self.controller(batch_stats)
        rho = control['rho']  # Tensor!
    else:
        rho = self.rho_default  # Fallback

    # Cross-injection with TENSOR rho (gradients flow!)
    z_A_cross = (1 - rho) * z_A + rho * z_B.detach()
    z_B_cross = (1 - rho) * z_B + rho * z_A.detach()

    # Decode
    logits_A = self.decoder_A(z_A_cross)
    logits_B = self.decoder_B(z_B_cross)

    return {
        'logits_A': logits_A,
        'logits_B': logits_B,
        'control': control,  # For use in loss computation
        ...
    }
```

---

## Integration in Loss Computation

```python
def compute_loss(self, x, outputs):
    control = outputs['control']

    # Base losses (all tensors)
    recon_A = F.cross_entropy(outputs['logits_A'], x)
    recon_B = F.cross_entropy(outputs['logits_B'], x)
    kl_A = self.kl_divergence(outputs['mu_A'], outputs['logvar_A'])
    kl_B = self.kl_divergence(outputs['mu_B'], outputs['logvar_B'])

    # Ranking loss (if enabled)
    ranking = self.ranking_loss(outputs['z_A'], outputs['z_B'], batch_indices)

    # Entropy alignment
    entropy_diff = torch.abs(outputs['H_A'] - outputs['H_B'])

    # DIFFERENTIABLE WEIGHTING (gradients flow through control!)
    total_loss = (
        control['weight_recon'] * (recon_A + recon_B) +
        control['weight_kl'] * (kl_A + kl_B) +
        control['weight_ranking'] * ranking +
        control['weight_entropy'] * entropy_diff
    )

    # Controller learns: "what weights minimize total_loss over time?"
    # Gradients flow: loss → weights → controller → controller params

    return total_loss
```

---

## Why This Works

1. **All control signals are tensors** participating in tensor operations
2. **Gradients flow naturally**: `∂loss/∂weight → ∂weight/∂controller_params`
3. **Controller learns**: Weights that reduce loss get reinforced
4. **No RL needed**: Standard backprop works
5. **Lightweight**: ~2K params vs 8K+ in current StateNet

---

## What We Remove

1. **Complex 21D input vector** - replaced with 8 batch statistics
2. **Attention heads** - not needed for simple control
3. **Unused outputs** (delta_lr, delta_sigma, delta_curvature) - can't be differentiable
4. **EMA tracking** - batch statistics are computed fresh

---

## What We Keep (Conceptually)

1. **Adaptive control** - network learns to adjust based on training state
2. **Cross-injection modulation** - rho is now differentiable
3. **Loss weighting** - lambdas become learned tensor weights

---

## Implementation Plan

1. Create `DifferentiableController` class (simple, ~50 lines)
2. Modify forward pass to compute batch_stats and call controller
3. Modify loss computation to use tensor weights
4. Remove StateNet and all its complexity
5. Test: verify gradients flow with `torch.autograd.grad`

---

## Expected Benefits

- **Actually learns**: Gradients flow, weights update
- **Simpler**: 2K params vs 8K, no attention heads
- **Faster**: No complex attention expansion
- **Debuggable**: Can inspect what weights the controller learns

---

## Open Questions

1. Should rho be per-sample or per-batch?
2. What batch statistics are most informative?
3. Should we add regularization to prevent extreme weights?
4. How to handle warmup (controller starts random)?
