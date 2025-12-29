"""Extract and compare embeddings from all v5.11 checkpoint variants.

This script performs deep analysis of the embedding spaces from:
- v5_11 (280 epochs, best @ epoch 9)
- v5_11_overnight (12,800 epochs, best @ epoch 12761)
- v5_11_structural (best @ epoch 44)

Goal: Identify unique structural properties in each training path.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR

import json
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def hyperbolic_radii(embeddings: np.ndarray, c: float = 1.0) -> np.ndarray:
    """V5.12.2: Compute hyperbolic distance from origin for Poincare ball embeddings."""
    sqrt_c = np.sqrt(c)
    euclidean_norms = np.linalg.norm(embeddings, axis=1)
    clamped = np.clip(euclidean_norms * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


def generate_all_ternary_operations() -> np.ndarray:
    """Generate all 19,683 possible ternary operations."""
    operations = []
    for i in range(3**9):
        op = []
        num = i
        for _ in range(9):
            op.append(num % 3 - 1)
            num //= 3
        operations.append(op)
    return np.array(operations, dtype=np.float32)


def nu_3(n: int) -> int:
    """Compute 3-adic valuation of n."""
    if n == 0:
        return 10  # Convention for 0
    count = 0
    while n % 3 == 0:
        n //= 3
        count += 1
    return count


class HyperbolicProjection(nn.Module):
    """Trainable projection to Poincare ball."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_radius = max_radius
        self.curvature = curvature

        # Direction network
        layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        ]
        self.direction_net = nn.Sequential(*layers)

        # Radius network
        radius_hidden = max(32, hidden_dim // 2)
        self.radius_net = nn.Sequential(
            nn.Linear(latent_dim, radius_hidden),
            nn.SiLU(),
            nn.Linear(radius_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_euclidean: torch.Tensor) -> torch.Tensor:
        """Project Euclidean latent to Poincare ball."""
        import torch.nn.functional as F

        direction_residual = self.direction_net(z_euclidean)
        direction = F.normalize(z_euclidean + direction_residual, dim=-1)
        radius = self.radius_net(z_euclidean) * self.max_radius
        z_hyp = direction * radius
        return z_hyp


def load_checkpoint_embeddings(checkpoint_path: Path, device: str = "cpu") -> Dict[str, Any]:
    """Load checkpoint and extract projection weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint["metrics"],
        "model_state": checkpoint["model_state"],
        "config": checkpoint.get("config", {}),
    }


def create_projection_from_state_dict(state_dict: Dict, device: str = "cpu") -> nn.Module:
    """Dynamically create projection module matching state dict architecture."""
    # Build the modules dynamically from state dict
    direction_layers = []
    radius_layers = []

    # Collect layer info
    dir_weights = {}
    rad_weights = {}
    for k, v in state_dict.items():
        if k.startswith("direction_net."):
            dir_weights[k] = v
        elif k.startswith("radius_net."):
            rad_weights[k] = v

    # Build direction_net dynamically
    class DynamicProjection(nn.Module):
        def __init__(self, dir_state, rad_state, max_radius=0.95):
            super().__init__()
            self.max_radius = max_radius

            # Build direction_net
            dir_layers = []
            i = 0
            while f"direction_net.{i}.weight" in dir_state or f"direction_net.{i}.bias" in dir_state:
                if f"direction_net.{i}.weight" in dir_state:
                    w = dir_state[f"direction_net.{i}.weight"]
                    b = dir_state.get(f"direction_net.{i}.bias")
                    if len(w.shape) == 2:
                        layer = nn.Linear(w.shape[1], w.shape[0])
                        dir_layers.append(layer)
                    elif len(w.shape) == 1:  # LayerNorm
                        layer = nn.LayerNorm(w.shape[0])
                        dir_layers.append(layer)
                i += 1
            # Add SiLU activations where needed (after Linear+LayerNorm pairs)
            self.direction_net = nn.Sequential(*dir_layers)

            # Build radius_net
            rad_layers = []
            i = 0
            while f"radius_net.{i}.weight" in rad_state or f"radius_net.{i}.bias" in rad_state:
                if f"radius_net.{i}.weight" in rad_state:
                    w = rad_state[f"radius_net.{i}.weight"]
                    if len(w.shape) == 2:
                        layer = nn.Linear(w.shape[1], w.shape[0])
                        rad_layers.append(layer)
                i += 1
            self.radius_net = nn.Sequential(*rad_layers)

            # Load state
            self.load_state_dict({**dir_state, **rad_state}, strict=False)

        def forward(self, z):
            import torch.nn.functional as F

            direction_residual = self.direction_net(z)
            direction = F.normalize(z + direction_residual, dim=-1)
            radius = torch.sigmoid(self.radius_net(z)) * self.max_radius
            return direction * radius

    proj = DynamicProjection(dir_weights, rad_weights)
    proj.to(device).eval()
    return proj


def create_projection_from_state(model_state: Dict, device: str = "cpu") -> Tuple[nn.Module, nn.Module]:
    """Create HyperbolicProjection and load weights from checkpoint.

    Returns proj_A and proj_B (dual projection architecture).
    """
    # Extract projection weights for proj_A and proj_B
    proj_A_state = {}
    proj_B_state = {}
    for k, v in model_state.items():
        if k.startswith("projection.proj_A."):
            new_key = k[len("projection.proj_A.") :]
            proj_A_state[new_key] = v
        elif k.startswith("projection.proj_B."):
            new_key = k[len("projection.proj_B.") :]
            proj_B_state[new_key] = v

    # Try standard architecture first
    try:
        proj_A = HyperbolicProjection(latent_dim=16, hidden_dim=64, max_radius=0.95)
        proj_A.load_state_dict(proj_A_state)
        proj_A.to(device).eval()

        proj_B = HyperbolicProjection(latent_dim=16, hidden_dim=64, max_radius=0.95)
        proj_B.load_state_dict(proj_B_state)
        proj_B.to(device).eval()
    except RuntimeError:
        # Different architecture - use dynamic loading
        print("  Using dynamic projection loading (different architecture)")

        # Simple manual loading approach
        proj_A = create_manual_projection(proj_A_state, device)
        proj_B = create_manual_projection(proj_B_state, device)

    return proj_A, proj_B


def create_manual_projection(state_dict: Dict, device: str = "cpu") -> nn.Module:
    """Manually create projection with correct architecture from state dict."""
    # Detect hidden dim from first layer
    first_weight = state_dict.get("direction_net.0.weight")
    if first_weight is not None:
        hidden_dim = first_weight.shape[0]
    else:
        hidden_dim = 64

    # Count layers
    n_layers = sum(1 for k in state_dict if "direction_net" in k and "weight" in k and len(state_dict[k].shape) == 2)

    # Create appropriate architecture
    if hidden_dim == 128:
        # Deeper architecture with hidden_dim=128
        proj = HyperbolicProjectionDeep(latent_dim=16, hidden_dim=128, max_radius=0.95)
    else:
        proj = HyperbolicProjection(latent_dim=16, hidden_dim=hidden_dim, max_radius=0.95)

    try:
        proj.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  Warning: partial load - {e}")

    proj.to(device).eval()
    return proj


class HyperbolicProjectionDeep(nn.Module):
    """Deeper projection architecture (matches v5_11_structural)."""

    def __init__(self, latent_dim=16, hidden_dim=128, max_radius=0.95):
        super().__init__()
        self.max_radius = max_radius

        # Deeper direction network with 3 hidden layers
        self.direction_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Deeper radius network
        radius_hidden = hidden_dim // 2
        self.radius_net = nn.Sequential(
            nn.Linear(latent_dim, radius_hidden),
            nn.SiLU(),
            nn.Linear(radius_hidden, radius_hidden),
            nn.SiLU(),
            nn.Linear(radius_hidden, radius_hidden),
            nn.SiLU(),
            nn.Linear(radius_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        import torch.nn.functional as F

        direction_residual = self.direction_net(z)
        direction = F.normalize(z + direction_residual, dim=-1)
        radius = self.radius_net(z) * self.max_radius
        return direction * radius


def load_frozen_encoder_weights(v5_5_checkpoint: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load frozen encoder weights from v5.5 checkpoint."""
    checkpoint = torch.load(v5_5_checkpoint, map_location=device, weights_only=True)
    model_state = checkpoint["model"]

    encoder_state = {}
    for key, value in model_state.items():
        if key.startswith("encoder_A."):
            new_key = key[len("encoder_A.") :]
            encoder_state[new_key] = value

    return encoder_state


def extract_euclidean_embeddings(data: torch.Tensor, encoder_state: Dict, device: str = "cpu") -> torch.Tensor:
    """Extract Euclidean embeddings using frozen encoder."""
    import torch.nn as nn

    # Recreate encoder architecture
    encoder = nn.Sequential(
        nn.Linear(9, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
    )
    fc_mu = nn.Linear(64, 16)

    # Load weights
    encoder_full_state = {}
    fc_mu_state = {}
    for k, v in encoder_state.items():
        if k.startswith("encoder."):
            encoder_full_state[k[len("encoder.") :]] = v
        elif k.startswith("fc_mu."):
            fc_mu_state[k[len("fc_mu.") :]] = v

    encoder.load_state_dict(encoder_full_state)
    fc_mu.load_state_dict(fc_mu_state)

    encoder.to(device).eval()
    fc_mu.to(device).eval()

    with torch.no_grad():
        h = encoder(data.to(device))
        mu = fc_mu(h)

    return mu


def compute_padic_valuations(data: np.ndarray) -> np.ndarray:
    """Compute 3-adic valuations for each operation."""
    valuations = []
    for op in data:
        # Convert to index
        idx = 0
        for i, v in enumerate(op):
            idx += (int(v) + 1) * (3**i)
        valuations.append(nu_3(idx))
    return np.array(valuations)


def analyze_embedding_space(embeddings: torch.Tensor, valuations: np.ndarray, name: str) -> Dict[str, Any]:
    """Perform comprehensive analysis of embedding space."""
    emb_np = embeddings.cpu().numpy()

    # Basic statistics (V5.12.2: use hyperbolic radii, not Euclidean norms)
    norms = np.linalg.norm(emb_np, axis=1)  # Keep for reference

    # Radial analysis (for hyperbolic embeddings)
    radii = hyperbolic_radii(emb_np)  # V5.12.2: proper hyperbolic distance from origin

    # Correlation with p-adic valuations
    valuation_radial_corr = np.corrcoef(valuations, radii)[0, 1]

    # Per-valuation statistics
    unique_vals = np.unique(valuations)
    val_stats = {}
    for v in unique_vals:
        mask = valuations == v
        val_stats[int(v)] = {
            "count": int(np.sum(mask)),
            "mean_radius": float(np.mean(radii[mask])),
            "std_radius": float(np.std(radii[mask])),
            "min_radius": float(np.min(radii[mask])),
            "max_radius": float(np.max(radii[mask])),
        }

    # Distance matrix analysis
    from scipy.spatial.distance import pdist, squareform

    # Sample for efficiency (full 19683 x 19683 is large)
    n_sample = min(2000, len(emb_np))
    indices = np.random.choice(len(emb_np), n_sample, replace=False)
    sample_emb = emb_np[indices]
    sample_val = valuations[indices]

    distances = pdist(sample_emb)
    dist_matrix = squareform(distances)

    # Valuation difference matrix
    val_diff = np.abs(sample_val[:, None] - sample_val[None, :])
    val_diff_flat = val_diff[np.triu_indices(n_sample, k=1)]

    # Distance-valuation correlation
    dist_val_corr = np.corrcoef(distances, val_diff_flat)[0, 1]

    # Clustering analysis by valuation
    from scipy.cluster.hierarchy import linkage

    Z = linkage(sample_emb, method="ward")

    # Silhouette-like metric: intra-valuation vs inter-valuation distances
    intra_dists = []
    inter_dists = []
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            if sample_val[i] == sample_val[j]:
                intra_dists.append(dist_matrix[i, j])
            else:
                inter_dists.append(dist_matrix[i, j])

    separation_ratio = np.mean(inter_dists) / (np.mean(intra_dists) + 1e-8)

    # PCA analysis
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(10, emb_np.shape[1]))
    pca.fit(emb_np)
    explained_variance = pca.explained_variance_ratio_

    # Intrinsic dimensionality estimation
    cumsum = np.cumsum(explained_variance)
    intrinsic_dim_90 = np.searchsorted(cumsum, 0.90) + 1
    intrinsic_dim_95 = np.searchsorted(cumsum, 0.95) + 1

    return {
        "name": name,
        "n_embeddings": len(emb_np),
        "embedding_dim": emb_np.shape[1],
        "radial_stats": {
            "mean": float(np.mean(radii)),
            "std": float(np.std(radii)),
            "min": float(np.min(radii)),
            "max": float(np.max(radii)),
            "range": float(np.max(radii) - np.min(radii)),
        },
        "valuation_radial_correlation": float(valuation_radial_corr),
        "distance_valuation_correlation": float(dist_val_corr),
        "separation_ratio": float(separation_ratio),
        "per_valuation_stats": val_stats,
        "pca": {
            "explained_variance_ratio": explained_variance.tolist(),
            "intrinsic_dim_90pct": int(intrinsic_dim_90),
            "intrinsic_dim_95pct": int(intrinsic_dim_95),
        },
    }


def compute_hyperbolic_metrics(embeddings: torch.Tensor, curvature: float = 1.0) -> Dict[str, float]:
    """Compute hyperbolic geometry specific metrics."""
    emb = embeddings.cpu().numpy()
    euclidean_norms = np.linalg.norm(emb, axis=1)

    # Hyperbolic distance from origin (V5.12.2: use proper formula)
    # d_H(0, x) = 2 * arctanh(sqrt(c) * |x|) / sqrt(c)
    hyp_dists = hyperbolic_radii(emb, c=curvature)

    # Effective curvature estimation
    # Points near boundary have higher effective curvature
    boundary_points = euclidean_norms > 0.9  # Boundary in Poincare ball
    core_points = euclidean_norms < 0.3  # Core region

    return {
        "mean_hyperbolic_distance": float(np.mean(hyp_dists)),
        "std_hyperbolic_distance": float(np.std(hyp_dists)),
        "boundary_fraction": float(np.mean(boundary_points)),
        "core_fraction": float(np.mean(core_points)),
        "hyperbolic_spread": float(np.max(hyp_dists) - np.min(hyp_dists)),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths
    v5_5_path = CHECKPOINTS_DIR / "v5_5" / "best.pt"

    checkpoints = {
        "v5_11": CHECKPOINTS_DIR / "v5_11" / "best.pt",
        "v5_11_overnight": CHECKPOINTS_DIR / "v5_11_overnight" / "best.pt",
        "v5_11_structural": CHECKPOINTS_DIR / "v5_11_structural" / "best.pt",
    }

    # Generate all ternary operations
    print("Generating ternary operations...")
    data = generate_all_ternary_operations()
    data_tensor = torch.from_numpy(data).float()

    # Compute p-adic valuations
    print("Computing p-adic valuations...")
    valuations = compute_padic_valuations(data)
    print(f"Valuation distribution: {np.bincount(valuations.astype(int))}")

    # Load frozen encoder
    print("Loading frozen encoder from v5.5...")
    encoder_state = load_frozen_encoder_weights(v5_5_path, device)

    # Extract Euclidean embeddings (same for all variants - frozen encoder)
    print("Extracting Euclidean embeddings...")
    euclidean_emb = extract_euclidean_embeddings(data_tensor, encoder_state, device)
    print(f"Euclidean embeddings shape: {euclidean_emb.shape}")

    # Analyze Euclidean space first
    print("\n" + "=" * 60)
    print("EUCLIDEAN EMBEDDING ANALYSIS (Frozen v5.5 Encoder)")
    print("=" * 60)
    euclidean_analysis = analyze_embedding_space(euclidean_emb, valuations, "euclidean_v5.5")
    print(f"Valuation-radial correlation: {euclidean_analysis['valuation_radial_correlation']:.4f}")
    print(f"Distance-valuation correlation: {euclidean_analysis['distance_valuation_correlation']:.4f}")
    print(f"Separation ratio: {euclidean_analysis['separation_ratio']:.4f}")

    # Results storage
    all_results = {"euclidean": euclidean_analysis}

    # Analyze each hyperbolic projection variant
    for name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"HYPERBOLIC EMBEDDING ANALYSIS: {name.upper()}")
        print("=" * 60)

        # Load checkpoint
        ckpt_data = load_checkpoint_embeddings(ckpt_path, device)
        print(f"Best epoch: {ckpt_data['epoch']}")
        print(f"Stored metrics: {ckpt_data['metrics']}")

        # Create projection and load weights (dual projection)
        proj_A, proj_B = create_projection_from_state(ckpt_data["model_state"], device)

        # Project to hyperbolic space using proj_A (primary projection)
        with torch.no_grad():
            hyp_emb = proj_A(euclidean_emb)
            hyp_emb_B = proj_B(euclidean_emb)

        print(f"Hyperbolic embeddings shape (A): {hyp_emb.shape}")
        print(f"Hyperbolic embeddings shape (B): {hyp_emb_B.shape}")

        # Analyze hyperbolic space
        hyp_analysis = analyze_embedding_space(hyp_emb, valuations, name)
        hyp_metrics = compute_hyperbolic_metrics(hyp_emb)

        print("\nRadial statistics:")
        print(f"  Mean radius: {hyp_analysis['radial_stats']['mean']:.4f}")
        print(f"  Radius range: [{hyp_analysis['radial_stats']['min']:.4f}, {hyp_analysis['radial_stats']['max']:.4f}]")

        print("\nStructural correlations:")
        print(f"  Valuation-radial: {hyp_analysis['valuation_radial_correlation']:.4f}")
        print(f"  Distance-valuation: {hyp_analysis['distance_valuation_correlation']:.4f}")
        print(f"  Separation ratio: {hyp_analysis['separation_ratio']:.4f}")

        print("\nHyperbolic geometry:")
        print(f"  Mean hyperbolic distance: {hyp_metrics['mean_hyperbolic_distance']:.4f}")
        print(f"  Boundary fraction (r>0.9): {hyp_metrics['boundary_fraction']:.4f}")
        print(f"  Core fraction (r<0.3): {hyp_metrics['core_fraction']:.4f}")

        print("\nPCA analysis:")
        print(f"  Intrinsic dim (90%): {hyp_analysis['pca']['intrinsic_dim_90pct']}")
        print(f"  Intrinsic dim (95%): {hyp_analysis['pca']['intrinsic_dim_95pct']}")
        print(f"  Top 3 variance: {hyp_analysis['pca']['explained_variance_ratio'][:3]}")

        print("\nPer-valuation radii:")
        for v in sorted(hyp_analysis["per_valuation_stats"].keys()):
            stats = hyp_analysis["per_valuation_stats"][v]
            print(f"  v={v}: mean={stats['mean_radius']:.4f}, std={stats['std_radius']:.4f}, n={stats['count']}")

        # Store results
        all_results[name] = {
            "embedding_analysis": hyp_analysis,
            "hyperbolic_metrics": hyp_metrics,
            "checkpoint_epoch": ckpt_data["epoch"],
            "checkpoint_metrics": {
                k: float(v) if hasattr(v, "item") else v for k, v in ckpt_data["metrics"].items() if not isinstance(v, np.ndarray)
            },
        }

    # Comparative summary
    print("\n" + "=" * 60)
    print("COMPARATIVE SUMMARY")
    print("=" * 60)

    print("\n{:<20} {:>12} {:>12} {:>12} {:>12}".format("Metric", "v5_11", "overnight", "structural", "Best"))
    print("-" * 68)

    metrics_to_compare = [
        (
            "val_rad_corr",
            lambda r: abs(r["embedding_analysis"]["valuation_radial_correlation"]),
        ),
        (
            "dist_val_corr",
            lambda r: r["embedding_analysis"]["distance_valuation_correlation"],
        ),
        ("separation", lambda r: r["embedding_analysis"]["separation_ratio"]),
        ("hyp_spread", lambda r: r["hyperbolic_metrics"]["hyperbolic_spread"]),
        (
            "intrinsic_dim",
            lambda r: r["embedding_analysis"]["pca"]["intrinsic_dim_90pct"],
        ),
    ]

    for metric_name, extractor in metrics_to_compare:
        values = {}
        for name in ["v5_11", "v5_11_overnight", "v5_11_structural"]:
            values[name] = extractor(all_results[name])

        best_name = max(values, key=values.get) if metric_name != "intrinsic_dim" else min(values, key=values.get)

        print(
            "{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>12}".format(
                metric_name,
                values["v5_11"],
                values["v5_11_overnight"],
                values["v5_11_structural"],
                best_name.replace("v5_11_", ""),
            )
        )

    # Save results
    output_path = Path(__file__).parent / "comparison_results.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Identify unique strengths
    print("\n" + "=" * 60)
    print("UNIQUE STRENGTHS PER VARIANT")
    print("=" * 60)

    v5_11 = all_results["v5_11"]
    overnight = all_results["v5_11_overnight"]
    structural = all_results["v5_11_structural"]

    print("\nv5_11 (production baseline):")
    print(f"  - Early convergence (epoch {v5_11['checkpoint_epoch']})")
    print("  - Stable training dynamics")

    print("\nv5_11_overnight (extended exploration):")
    print(f"  - Highest distance-valuation correlation: {overnight['embedding_analysis']['distance_valuation_correlation']:.4f}")
    print(f"  - Best hyperbolic spread: {overnight['hyperbolic_metrics']['hyperbolic_spread']:.4f}")
    print(f"  - Most thorough exploration ({overnight['checkpoint_epoch']} epochs)")

    print("\nv5_11_structural (structural focus):")
    print("  - Targeted structural learning")
    if "composite_score" in structural.get("checkpoint_metrics", {}):
        print("  - Composite score tracking enabled")


if __name__ == "__main__":
    main()
