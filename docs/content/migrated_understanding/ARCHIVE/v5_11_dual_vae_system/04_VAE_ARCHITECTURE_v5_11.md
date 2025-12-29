# VAE Architecture: The Dual-Neural Design

**Why two VAEs are better than one, and how frozen components preserve knowledge**

---

## 1. What is a VAE?

### The Variational Autoencoder

A VAE learns to:
1. **Encode** data into a compressed latent representation
2. **Decode** the representation back to data
3. **Regularize** the latent space to be smooth and continuous

```
Input x → Encoder → μ, σ → Sample z → Decoder → Reconstruction x̂
                         ↓
              KL divergence to prior N(0,1)
```

### The Reparameterization Trick

To backpropagate through sampling:
```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # Random noise
    return mu + eps * std        # Differentiable!
```

---

## 2. Why TWO VAEs?

### The Exploration-Exploitation Dilemma

A single VAE must balance:
- **Exploration**: Finding diverse representations
- **Exploitation**: Accurate reconstruction

These goals conflict! High entropy → diverse but noisy. Low entropy → accurate but collapsed.

### The Dual-VAE Solution

We split the responsibilities:

```
                    Input x
                   /       \
                  /         \
            VAE-A            VAE-B
         (Explorer)       (Consolidator)
         High entropy     Low entropy
         High temperature Low temperature
         Discovers new    Refines known
              \              /
               \            /
                 StateNet
             (Meta-controller)
                    ↓
              Balanced output
```

### VAE-A: The Explorer

- **Role**: Discover novel functional regions
- **Behavior**: High temperature, high entropy
- **Character**: "Chaotic", exploratory
- **Analogy**: The creative artist trying new things

### VAE-B: The Consolidator

- **Role**: Ensure reconstruction accuracy
- **Behavior**: Low temperature, stable
- **Character**: "Anchor", reliable
- **Analogy**: The craftsman perfecting details

---

## 3. The V5.11 Architecture

### Design Philosophy

```
V5.11 Key Insight:
  - v5.5 achieved 100% coverage (all operations reconstructed)
  - BUT the radial hierarchy was inverted
  - Solution: FREEZE the encoder, TRAIN only the projection
```

### Architecture Diagram

```
Input: x (batch, 9) - Ternary operation {-1, 0, 1}

┌──────────────────────────────────────────────────────┐
│         FROZEN ENCODERS (from v5.5 checkpoint)       │
│  FrozenEncoder_A → mu_A, logvar_A (16D)             │
│  FrozenEncoder_B → mu_B, logvar_B (16D)             │
│  NO GRADIENTS - Preserves 100% coverage             │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│         TRAINABLE HYPERBOLIC PROJECTION              │
│  z_euclidean (16D) → MLP(64) → exp_map → z_poincare │
│  TRAINABLE - Learns Euclidean → Poincare mapping    │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│         TRAINABLE DIFFERENTIABLE CONTROLLER          │
│  z_hyp, stats → Control signals (rho, lambda)       │
│  TRAINABLE - Learns adaptive loss weighting         │
└──────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│                  FROZEN DECODER                      │
│  z_A (16D) → Linear(16→64→27) → logits (batch,9,3) │
│  NO GRADIENTS - Verification only                   │
└──────────────────────────────────────────────────────┘
```

### Key Insight: Partial Freezing

```
FROZEN (preserve coverage):     TRAINABLE (learn geometry):
├── encoder_A                   ├── HyperbolicProjection
├── encoder_B                   └── DifferentiableController
└── decoder_A
```

---

## 4. Component Details

### FrozenEncoder (src/models/frozen_components.py)

```python
class FrozenEncoder(nn.Module):
    """Encoder loaded from checkpoint, no gradients."""

    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
```

### HyperbolicProjection (src/models/hyperbolic_projection.py)

```python
class HyperbolicProjection(nn.Module):
    """Projects Euclidean latents to Poincare ball."""

    def __init__(self, latent_dim=16, hidden_dim=64, max_radius=0.95):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.max_radius = max_radius

    def forward(self, z_euclidean):
        z_transformed = self.mlp(z_euclidean)
        z_hyperbolic = exp_map_zero(z_transformed)
        z_hyperbolic = project_to_poincare(z_hyperbolic, self.max_radius)
        return z_hyperbolic
```

### DifferentiableController (src/models/differentiable_controller.py)

```python
class DifferentiableController(nn.Module):
    """Learns to adjust loss weights based on training state."""

    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 6 control signals
        )

    def forward(self, batch_stats):
        # batch_stats: [radius_A, radius_B, H_A, H_B, kl_A, kl_B, geo_loss, rad_loss]
        raw = self.fc(batch_stats)
        return {
            "rho": torch.sigmoid(raw[0]),           # Cross-injection
            "weight_geodesic": torch.sigmoid(raw[1]),
            "weight_radial": torch.sigmoid(raw[2]),
            "beta_A": F.softplus(raw[3]),
            "beta_B": F.softplus(raw[4]),
            "tau": torch.sigmoid(raw[5])            # Temperature
        }
```

---

## 5. Forward Pass

### Complete Forward Flow

```python
def forward(self, x, compute_control=True):
    # 1. Frozen encoding (no gradients)
    with torch.no_grad():
        mu_A, logvar_A = self.encoder_A(x)
        mu_B, logvar_B = self.encoder_B(x)
        z_A_euc = self.reparameterize(mu_A, logvar_A)
        z_B_euc = self.reparameterize(mu_B, logvar_B)

    # 2. Trainable projection to Poincaré ball
    z_A_hyp = self.projection(z_A_euc)
    z_B_hyp = self.projection(z_B_euc)

    # 3. Compute control signals
    if compute_control and self.controller:
        batch_stats = compute_batch_statistics(z_A_hyp, z_B_hyp, ...)
        control = self.controller(batch_stats)
    else:
        control = self.default_control

    # 4. Frozen decoder (verification only)
    with torch.no_grad():
        logits_A = self.decoder_A(z_A_euc)

    return {
        "z_A_euc": z_A_euc, "z_B_euc": z_B_euc,
        "z_A_hyp": z_A_hyp, "z_B_hyp": z_B_hyp,
        "logits_A": logits_A,
        "control": control,
        "mu_A": mu_A, "mu_B": mu_B,
        "logvar_A": logvar_A, "logvar_B": logvar_B
    }
```

---

## 6. Why This Design Works

### Principle 1: Freeze What Works

```
v5.5 achieved 100% coverage → DON'T BREAK IT
Freeze encoders and decoder → Coverage guaranteed
Train only projection → Learn geometry without losing coverage
```

### Principle 2: Separation of Concerns

```
Encoder: Compress to latent (frozen, proven)
Projection: Map to hyperbolic space (trainable, new)
Controller: Adapt training (trainable, meta)
Decoder: Reconstruct (frozen, proven)
```

### Principle 3: Differentiable End-to-End

```
Even though encoders are frozen:
- Projection receives frozen outputs (no gradient to encoder)
- Projection learns its own weights (gradient to projection)
- Controller learns from all statistics (gradient to controller)
```

---

## 7. Parameter Counts

### V5.11 Parameters

```python
def count_parameters(self):
    return {
        "frozen": 167,234,      # Encoder_A + Encoder_B + Decoder
        "projection": 1,280,    # Hyperbolic projection MLP
        "controller": 256,      # Differentiable controller
        "trainable": 1,536,     # projection + controller
        "total": 168,770
    }
```

**Key Insight**: Only 0.9% of parameters are trainable!
- We're fine-tuning geometry, not relearning everything
- This is 100x fewer parameters than EVE (~15M)

---

## 8. Loading Pretrained Weights

### Checkpoint Loading

```python
def load_v5_5_checkpoint(self, checkpoint_path, device="cpu"):
    """Load frozen components from v5.5 checkpoint."""
    checkpoint = load_checkpoint_compat(checkpoint_path)
    model_state = checkpoint["model"]

    # Load encoder_A
    enc_A_state = {
        k.replace("encoder_A.", ""): v
        for k, v in model_state.items()
        if k.startswith("encoder_A.")
    }
    self.encoder_A.load_state_dict(enc_A_state)

    # Similar for encoder_B and decoder_A...

    # Ensure frozen
    for param in self.encoder_A.parameters():
        param.requires_grad = False
```

---

## 9. Variants

### TernaryVAEV5_11_PartialFreeze

An alternative where encoder_B is trainable:
```python
class TernaryVAEV5_11_PartialFreeze(TernaryVAEV5_11):
    """
    Encoder_A: FROZEN (coverage anchor)
    Encoder_B: TRAINABLE (can adapt)
    Projection: TRAINABLE
    """
```

### DualHyperbolicProjection

Separate projections for A and B:
```python
class DualHyperbolicProjection(nn.Module):
    """Separate projections for explorer and consolidator."""

    def __init__(self, ...):
        self.projection_A = HyperbolicProjection(...)  # For explorer
        self.projection_B = HyperbolicProjection(...)  # For consolidator

    def forward(self, z_A, z_B):
        return self.projection_A(z_A), self.projection_B(z_B)
```

---

## 10. Output Dictionary

### All Forward Pass Outputs

```python
{
    # Euclidean latents (from frozen encoder)
    "z_A_euc": Tensor(batch, 16),
    "z_B_euc": Tensor(batch, 16),
    "mu_A": Tensor(batch, 16),
    "mu_B": Tensor(batch, 16),
    "logvar_A": Tensor(batch, 16),
    "logvar_B": Tensor(batch, 16),

    # Hyperbolic latents (from trainable projection)
    "z_A_hyp": Tensor(batch, 16),
    "z_B_hyp": Tensor(batch, 16),

    # Control signals (from trainable controller)
    "control": {
        "rho": Tensor(1),           # Cross-injection weight
        "weight_geodesic": Tensor(1),
        "weight_radial": Tensor(1),
        "beta_A": Tensor(1),
        "beta_B": Tensor(1),
        "tau": Tensor(1)            # Temperature
    },

    # Reconstruction (from frozen decoder)
    "logits_A": Tensor(batch, 9, 3)
}
```

---

## Summary

The Dual-VAE architecture works because:

1. **Division of Labor**: Explorer discovers, Consolidator refines
2. **Freeze-and-Extend**: Proven coverage preserved, geometry learned
3. **Geometric Projection**: Euclidean → Poincare maps hierarchy
4. **Adaptive Control**: Controller balances losses dynamically
5. **Minimal Training**: Only 0.9% of parameters trainable

---

*Next, we'll examine the loss functions that train this architecture.*
