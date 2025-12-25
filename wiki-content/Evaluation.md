# Evaluation

Metrics and methods for evaluating Ternary VAE models.

---

## Core Metrics

### Reconstruction Accuracy

How well the model reconstructs input operations:

```python
def compute_reconstruction_accuracy(model, dataloader, device):
    """Compute reconstruction accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            preds = outputs["reconstruction"].argmax(dim=1)

            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

    return correct / total
```

**Interpretation**:
- Random guessing: ~0.005% (1/19683)
- Good model: >80%
- Excellent: >95%

### ELBO (Evidence Lower Bound)

The VAE objective:

```python
def compute_elbo(model, dataloader, device):
    """Compute ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))."""
    model.eval()
    total_recon = 0
    total_kl = 0
    n_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            # Reconstruction term
            recon_loss = F.cross_entropy(
                outputs["reconstruction"], batch_y, reduction="sum"
            )

            # KL term (assuming wrapped Gaussian prior)
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()

            total_recon += recon_loss.item()
            total_kl += kl.item()
            n_samples += len(batch_x)

    elbo = -(total_recon + total_kl) / n_samples
    return {
        "elbo": elbo,
        "reconstruction": -total_recon / n_samples,
        "kl": total_kl / n_samples,
    }
```

### Latent Space Quality

#### 1. Coverage

How much of the Poincare ball is used:

```python
def compute_coverage(model, dataloader, device, n_bins=10):
    """Compute latent space coverage."""
    model.eval()
    all_z = []

    with torch.no_grad():
        for batch_x, _ in dataloader:
            outputs = model(batch_x.to(device))
            all_z.append(outputs["z_hyperbolic"].cpu())

    z = torch.cat(all_z)

    # Radial coverage (bins by distance from origin)
    radii = z.norm(dim=1)
    bins = torch.linspace(0, 1, n_bins + 1)
    hist, _ = torch.histogram(radii, bins)
    radial_coverage = (hist > 0).float().mean().item()

    # Angular coverage (for 2D, bins by angle)
    if z.shape[1] == 2:
        angles = torch.atan2(z[:, 1], z[:, 0])
        angle_bins = torch.linspace(-torch.pi, torch.pi, n_bins + 1)
        angle_hist, _ = torch.histogram(angles, angle_bins)
        angular_coverage = (angle_hist > 0).float().mean().item()
    else:
        angular_coverage = None

    return {
        "radial_coverage": radial_coverage,
        "angular_coverage": angular_coverage,
        "mean_radius": radii.mean().item(),
        "max_radius": radii.max().item(),
    }
```

#### 2. Separation

How well-separated are different classes:

```python
def compute_class_separation(model, dataloader, device):
    """Compute inter/intra class distances."""
    from src.geometry import poincare_distance

    model.eval()
    all_z = []
    all_y = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x.to(device))
            all_z.append(outputs["z_hyperbolic"].cpu())
            all_y.append(batch_y)

    z = torch.cat(all_z)
    y = torch.cat(all_y)

    classes = y.unique()
    intra_dists = []
    inter_dists = []

    for c in classes:
        mask = y == c
        z_c = z[mask]

        # Intra-class: distances within class
        if len(z_c) > 1:
            for i in range(len(z_c)):
                for j in range(i + 1, len(z_c)):
                    d = poincare_distance(z_c[i], z_c[j], curvature=1.0)
                    intra_dists.append(d.item())

        # Inter-class: distances to other classes
        z_other = z[~mask]
        for i in range(min(100, len(z_c))):
            for j in range(min(100, len(z_other))):
                d = poincare_distance(z_c[i], z_other[j], curvature=1.0)
                inter_dists.append(d.item())

    return {
        "intra_class_mean": np.mean(intra_dists),
        "inter_class_mean": np.mean(inter_dists),
        "separation_ratio": np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8),
    }
```

---

## Hierarchical Metrics

### Radial-Valuation Correlation

Do high-valuation operations lie near the center?

```python
def compute_radial_correlation(model, dataloader, device):
    """Correlate radial position with 3-adic valuation."""
    model.eval()
    all_z = []
    all_ops = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x.to(device))
            all_z.append(outputs["z_hyperbolic"].cpu())
            all_ops.append(batch_y)

    z = torch.cat(all_z)
    ops = torch.cat(all_ops)

    # Compute 3-adic valuations
    def valuation_3(n):
        if n == 0:
            return 9  # Max valuation
        v = 0
        while n % 3 == 0:
            n //= 3
            v += 1
        return v

    valuations = torch.tensor([valuation_3(op.item()) for op in ops])
    radii = z.norm(dim=1)

    # Correlation (should be negative: high valuation = low radius)
    correlation = torch.corrcoef(torch.stack([radii, valuations.float()]))[0, 1]

    return {
        "radial_valuation_correlation": correlation.item(),
        # Negative is good: high valuation → center
    }
```

### Hierarchy Preservation

Do parent-child relationships in the ternary structure map to distance?

```python
def compute_hierarchy_preservation(model, dataloader, device):
    """Check if hierarchical structure is preserved."""
    # Sample pairs with known hierarchical relationship
    # (operations divisible by 3 are "parents" of their remainders)
    pass  # Implementation depends on specific hierarchy definition
```

---

## Generation Quality

### Sample Quality

```python
def generate_samples(model, n_samples, device):
    """Generate new samples from prior."""
    model.eval()

    with torch.no_grad():
        # Sample from prior (wrapped Gaussian in Poincare ball)
        z = torch.randn(n_samples, model.latent_dim, device=device)
        z = exp_map_zero(z, curvature=model.curvature)

        # Decode
        reconstruction = model.decode(z)
        samples = reconstruction.argmax(dim=1)

    return samples

# Evaluate samples
samples = generate_samples(model, 1000, device)
print(f"Unique samples: {len(samples.unique())}")
print(f"Sample distribution: {samples.bincount()[:10]}")
```

### Interpolation Quality

```python
def interpolate(model, z1, z2, n_steps=10):
    """Interpolate between two latent points via geodesic."""
    from src.geometry import mobius_add

    model.eval()
    points = []

    for t in torch.linspace(0, 1, n_steps):
        # Geodesic interpolation
        delta = mobius_add(-z1, z2, curvature=1.0)
        z_t = mobius_add(z1, t * delta, curvature=1.0)
        points.append(z_t)

    with torch.no_grad():
        reconstructions = [model.decode(z.unsqueeze(0)).argmax(1).item() for z in points]

    return reconstructions
```

---

## Biological Metrics

### Codon Usage Bias

```python
def compute_codon_bias_score(sequences, target_organism="human"):
    """Score sequences by codon usage bias."""
    from src.losses.autoimmunity import HUMAN_CODON_RSCU

    if target_organism == "human":
        rscu = HUMAN_CODON_RSCU
    else:
        raise ValueError(f"Unknown organism: {target_organism}")

    scores = []
    for seq in sequences:
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        score = sum(rscu.get(c, 0) for c in codons) / len(codons)
        scores.append(score)

    return {
        "mean_rscu": np.mean(scores),
        "std_rscu": np.std(scores),
        "min_rscu": np.min(scores),
        "max_rscu": np.max(scores),
    }
```

### Conservation Score

```python
def compute_conservation_alignment(embeddings, alignment):
    """Correlate embedding distance with sequence conservation."""
    # Higher conservation should correlate with smaller embedding norm
    pass
```

---

## Benchmark Suite

### Quick Evaluation

```python
def quick_evaluate(model, val_loader, device):
    """Run quick evaluation suite."""
    results = {}

    # Accuracy
    results["accuracy"] = compute_reconstruction_accuracy(model, val_loader, device)

    # ELBO components
    elbo = compute_elbo(model, val_loader, device)
    results.update(elbo)

    # Coverage
    coverage = compute_coverage(model, val_loader, device)
    results.update(coverage)

    return results
```

### Full Evaluation

```python
def full_evaluate(model, val_loader, test_loader, device):
    """Run comprehensive evaluation."""
    results = quick_evaluate(model, val_loader, device)

    # Separation
    sep = compute_class_separation(model, test_loader, device)
    results.update(sep)

    # Hierarchy
    hier = compute_radial_correlation(model, test_loader, device)
    results.update(hier)

    return results
```

---

## Visualization

### Latent Space Plot

```python
def plot_latent_space(model, dataloader, device, save_path=None):
    """Visualize 2D latent space."""
    model.eval()
    all_z = []
    all_y = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x.to(device))
            all_z.append(outputs["z_hyperbolic"].cpu())
            all_y.append(batch_y)

    z = torch.cat(all_z).numpy()
    y = torch.cat(all_y).numpy()

    # Reduce to 2D if needed
    if z.shape[1] > 2:
        from sklearn.manifold import TSNE
        z = TSNE(n_components=2).fit_transform(z)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw Poincare ball boundary
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)

    # Plot points
    scatter = ax.scatter(z[:, 0], z[:, 1], c=y % 20, cmap='tab20', alpha=0.6, s=10)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('Latent Space')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
```

---

## Comparison Table

| Metric | Random | Poor | Good | Excellent |
|--------|--------|------|------|-----------|
| Accuracy | <1% | 1-50% | 50-90% | >90% |
| ELBO | - | <-10 | -10 to -5 | >-5 |
| Coverage | <20% | 20-50% | 50-80% | >80% |
| Separation Ratio | <1 | 1-2 | 2-5 | >5 |
| Radial Correlation | ~0 | 0 to -0.3 | -0.3 to -0.6 | <-0.6 |

---

*See also: [[Training]], [[Troubleshooting]], [[Tutorials]]*
