# Loss Functions: The Training Objectives

**Every loss component explained with intuition and mathematics**

---

## 1. Overview: The Multi-Objective Loss

Training the Ternary VAE requires balancing multiple objectives:

```
Total Loss = λ₁·Loss_A + λ₂·Loss_B + λ₃·Alignment + Geometric Losses

Where:
  Loss_A = Reconstruction_A + β_A·KL_A
  Loss_B = Reconstruction_B + β_B·KL_B + Entropy_B + Repulsion_B
  Alignment = |H_A - H_B|  (entropy difference)
  Geometric Losses = Geodesic + Radial + P-adic
```

---

## 2. Reconstruction Loss

### Purpose
Ensure the decoder can reconstruct the input from the latent code.

### Implementation
```python
class ReconstructionLoss(nn.Module):
    def forward(self, logits, x):
        # logits: (batch, 9, 3) - probabilities for each trit position
        # x: (batch, 9) - input with values in {-1, 0, 1}

        x_classes = (x + 1).long()  # Convert {-1,0,1} to {0,1,2}
        loss = F.cross_entropy(
            logits.view(-1, 3),     # Flatten to (batch*9, 3)
            x_classes.view(-1),     # Flatten to (batch*9,)
            reduction="sum"
        ) / batch_size

        return loss
```

### Intuition
- Each of the 9 positions predicts one of 3 values
- Cross-entropy measures how far predictions are from true values
- Lower loss = better reconstruction

---

## 3. KL Divergence Loss

### Purpose
Regularize the latent space to be close to a standard normal distribution.

### The β-VAE Extension
```python
class KLDivergenceLoss(nn.Module):
    def __init__(self, free_bits=0.0):
        self.free_bits = free_bits

    def forward(self, mu, logvar):
        # KL per dimension: -0.5 * (1 + log(σ²) - μ² - σ²)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if self.free_bits > 0:
            # Allow some "free" information per dimension
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        return kl_per_dim.sum() / batch_size
```

### Intuition
- KL divergence = how different is q(z|x) from p(z) = N(0,1)?
- High KL = latent codes far from origin, high variance
- Low KL = latent codes clustered at origin (posterior collapse!)

### Free Bits
The "free bits" trick allows some information per dimension without penalty:
- Prevents posterior collapse (all z → 0)
- Typically free_bits = 0.1 to 0.5 nats

### The β Weight
```
β > 1: Stronger regularization, more disentangled but worse reconstruction
β < 1: Better reconstruction but less regularization
β = 1: Standard VAE
```

---

## 4. Entropy Regularization

### Purpose
Encourage diverse outputs from VAE-B (the consolidator).

### Implementation
```python
class EntropyRegularization(nn.Module):
    def forward(self, logits):
        probs = F.softmax(logits, dim=-1).mean(dim=0)  # Average over batch
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return -entropy  # Return NEGATIVE entropy as LOSS
```

### Intuition
- High entropy = diverse predictions across batch
- Low entropy = all samples predict same thing
- We MAXIMIZE entropy by MINIMIZING negative entropy

---

## 5. Repulsion Loss

### Purpose
Spread out latent codes to use the full latent space.

### Implementation
```python
class RepulsionLoss(nn.Module):
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def forward(self, z):
        dists = torch.cdist(z, z, p=2)  # Pairwise Euclidean distances
        mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)

        # RBF kernel: penalize points that are too close
        repulsion = torch.exp(-dists[mask]**2 / self.sigma**2).mean()

        return repulsion
```

### Intuition
- If two latent codes are very close, the kernel is ~1 (high loss)
- If they're far apart, the kernel is ~0 (low loss)
- This spreads out the latent space

---

## 6. P-adic Metric Loss

### Purpose
Align latent distances with 3-adic distances.

### Implementation
```python
class PAdicMetricLoss(nn.Module):
    def __init__(self, scale=1.0, n_pairs=1000):
        self.scale = scale
        self.n_pairs = n_pairs

    def forward(self, z, indices):
        # Sample random pairs
        i, j = sample_pairs(len(z), self.n_pairs)

        # Compute p-adic distances
        padic_dist = compute_padic_distance(indices[i], indices[j])

        # Compute latent distances
        latent_dist = torch.norm(z[i] - z[j], dim=-1)

        # MSE between scaled distances
        loss = F.mse_loss(self.scale * latent_dist, padic_dist)

        return loss
```

### Intuition
- Operations with similar p-adic valuation should be close in latent space
- Operations with different valuations should be far apart
- The model learns to respect the hierarchical structure

---

## 7. P-adic Ranking Loss (Triplet-Based)

### Purpose
A more robust way to enforce p-adic structure using ranking.

### Implementation
```python
class PAdicRankingLoss(nn.Module):
    def __init__(self, margin=0.1, n_triplets=500):
        self.margin = margin
        self.n_triplets = n_triplets

    def forward(self, z, indices):
        # Sample triplets (anchor, positive, negative)
        # where d_padic(anchor, positive) < d_padic(anchor, negative)
        a, p, n = sample_triplets(indices, self.n_triplets)

        # Compute latent distances
        d_ap = torch.norm(z[a] - z[p], dim=-1)
        d_an = torch.norm(z[a] - z[n], dim=-1)

        # Triplet loss: d(a,p) + margin < d(a,n)
        loss = F.relu(d_ap - d_an + self.margin).mean()

        return loss
```

### Intuition
- "If A is closer to B than to C in p-adic terms, then A should be closer to B than to C in latent space"
- More robust than MSE because it only enforces ORDERING, not exact distances
- Margin prevents trivial solutions (all points at same location)

---

## 8. Hyperbolic Prior Loss

### Purpose
Replace Gaussian prior with a hyperbolic distribution that respects the Poincare ball geometry.

### Implementation
```python
class HyperbolicPrior(nn.Module):
    def __init__(self, curvature=1.0, prior_sigma=1.0, max_norm=0.95):
        self.curvature = curvature
        self.prior_sigma = prior_sigma
        self.max_norm = max_norm

    def forward(self, mu, logvar):
        # Sample z in Euclidean space
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_euc = mu + eps * std

        # Project to Poincare ball
        z_hyp = project_to_poincare(z_euc, self.max_norm)

        # Compute hyperbolic KL divergence
        radius = torch.norm(z_hyp, dim=-1)
        kl = self._hyperbolic_kl(radius)

        return kl, z_hyp
```

### Intuition
- Standard Gaussian prior assumes flat space
- Hyperbolic prior knows about the curved geometry
- Points are distributed according to hyperbolic volume element

---

## 9. Geodesic Loss

### Purpose
Enforce that p-adic distance correlates with hyperbolic (geodesic) distance.

### Implementation
```python
class GeodesicLoss(nn.Module):
    def __init__(self, curvature=1.0, target_correlation=-0.7):
        self.curvature = curvature
        self.target_correlation = target_correlation

    def forward(self, z_hyp, indices):
        # Compute pairwise hyperbolic distances
        hyp_dist = poincare_distance_matrix(z_hyp, self.curvature)

        # Compute pairwise p-adic distances
        padic_dist = padic_distance_matrix(indices)

        # Spearman correlation (we want NEGATIVE correlation)
        # High p-adic valuation = close to origin = small hyp distance
        correlation = spearman_correlation(hyp_dist, padic_dist)

        # Loss: deviation from target correlation
        loss = (correlation - self.target_correlation).pow(2)

        return loss, correlation
```

### Intuition
- High p-adic valuation (divisible by powers of 3) = "fundamental"
- Fundamental operations should be near origin (small hyperbolic distance)
- Derived operations should be near boundary (large hyperbolic distance)
- Target correlation of -0.7 means strong inverse relationship

---

## 10. Radial Loss

### Purpose
Directly enforce the radial hierarchy (high valuation → small radius).

### Implementation
```python
class RadialLoss(nn.Module):
    def __init__(self, inner_radius=0.1, outer_radius=0.85):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def forward(self, z_hyp, valuations):
        radius = torch.norm(z_hyp, dim=-1)

        # Normalize valuation to [0, 1]
        v_norm = valuations / 9.0  # Max valuation is 9

        # Target radius: high valuation → small radius
        target_radius = self.outer_radius - v_norm * (self.outer_radius - self.inner_radius)

        # MSE to target
        loss = F.mse_loss(radius, target_radius)

        return loss
```

### Intuition
- Valuation 9 (index 0) → radius 0.1 (origin)
- Valuation 0 (most indices) → radius 0.85 (near boundary)
- Creates a "radial stratification" of the latent space

---

## 11. Centroid Loss

### Purpose
Enforce tree structure by placing subtree centroids correctly.

### Implementation
```python
class HyperbolicCentroidLoss(nn.Module):
    def __init__(self, max_level=4, curvature=1.0):
        self.max_level = max_level
        self.curvature = curvature

    def forward(self, z, indices):
        total_loss = 0

        for level in range(1, self.max_level + 1):
            # Compute prefixes at this level
            prefixes = compute_prefix(indices, level)

            # For each unique prefix, compute centroid
            for prefix in unique_prefixes:
                members = z[prefixes == prefix]
                centroid = hyperbolic_centroid(members)

                # Centroid should be closer to origin than members
                centroid_radius = torch.norm(centroid)
                member_radii = torch.norm(members, dim=-1)

                # Loss: centroid should have smaller radius
                violation = F.relu(centroid_radius - member_radii.min())
                total_loss += violation

        return total_loss
```

### Intuition
- In a tree, the parent should be "above" (closer to origin) than children
- Centroids of subtrees represent ancestors
- This enforces the tree structure explicitly

---

## 12. Homeostatic Losses

### Purpose
Adaptively adjust loss parameters based on training state.

### Implementation
```python
class HomeostaticHyperbolicPrior(HyperbolicPrior):
    def __init__(self, ...):
        super().__init__(...)
        self.adaptive_sigma = nn.Parameter(torch.tensor(1.0))
        self.adaptive_curvature = nn.Parameter(torch.tensor(1.0))

    def update_homeostatic_state(self, z, kl):
        # If KL too high, increase sigma (spread prior)
        # If KL too low, decrease sigma (concentrate prior)
        target_kl = 5.0
        adjustment = (kl - target_kl) / target_kl
        self.adaptive_sigma.data *= (1 + 0.01 * adjustment)
```

### Intuition
- "Homeostasis" = maintaining balance
- If training is unstable, adapt parameters automatically
- Prevents getting stuck in bad regions

---

## 13. Complete Loss Computation

### The Full Forward Pass

```python
def forward(self, x, outputs, batch_indices, ...):
    # Base losses
    ce_A = self.reconstruction_loss(outputs["logits_A"], x)
    ce_B = self.reconstruction_loss(outputs["logits_B"], x)
    kl_A = self.kl_loss(outputs["mu_A"], outputs["logvar_A"])
    kl_B = self.kl_loss(outputs["mu_B"], outputs["logvar_B"])

    # VAE losses
    loss_A = ce_A + beta_A * kl_A
    loss_B = ce_B + beta_B * kl_B + entropy_B + repulsion_B

    # Geometric losses (if indices provided)
    if batch_indices is not None:
        padic_loss = self.padic_ranking_loss(outputs["z_A_hyp"], batch_indices)
        geodesic_loss = self.geodesic_loss(outputs["z_A_hyp"], batch_indices)
        radial_loss = self.radial_loss(outputs["z_A_hyp"], valuations)

    # Combine with weights
    total = (lambda1 * loss_A +
             lambda2 * loss_B +
             lambda3 * entropy_align +
             geo_weight * geodesic_loss +
             rad_weight * radial_loss +
             padic_weight * padic_loss)

    return {"loss": total, "ce_A": ce_A, ...}
```

---

## 14. Loss Weight Schedule

Weights change during training:

```
Phase 1 (epoch 0-40):   β_A warmup, focus on reconstruction
Phase 2 (epoch 40-49):  Consolidation, balance all losses
Phase 3 (epoch 50):     β_B warmup (disruption)
Phase 4 (epoch 50+):    Convergence, stable weights
```

---

## Summary

| Loss | Purpose | Key Parameter |
|------|---------|---------------|
| Reconstruction | Accurate output | None |
| KL Divergence | Regularize latent | β, free_bits |
| Entropy | Diversity in B | weight |
| Repulsion | Spread latent | σ |
| P-adic Metric | MSE to p-adic | scale |
| P-adic Ranking | Order preservation | margin |
| Geodesic | Hyperbolic correlation | target_r |
| Radial | Valuation→radius | inner/outer |
| Centroid | Tree structure | max_level |

---

*These losses work together to create a well-structured hyperbolic latent space. Next, we'll see how training orchestrates them.*
