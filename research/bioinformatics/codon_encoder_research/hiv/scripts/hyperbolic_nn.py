"""
Hyperbolic Neural Networks for HIV Sequence Analysis.

This module implements neural network layers that operate in hyperbolic space,
which naturally captures hierarchical and tree-like structures found in
viral evolution and phylogenetics.

Key features:
1. Poincaré ball operations (Möbius addition, exp/log maps)
2. Hyperbolic linear layers and MLPs
3. Hyperbolic attention mechanisms
4. Hyperbolic VAE components
5. Integration with p-adic codon encoding

Based on papers:
- Nickel & Kiela 2017: Poincaré embeddings
- Ganea et al. 2018: Hyperbolic neural networks
- Chami et al. 2019: HGCN
- Mathieu et al. 2019: Hyperbolic VAEs

Requirements:
    pip install torch numpy

Author: Research Team
Date: December 2025
"""

from typing import Optional, Tuple

import numpy as np

# Lazy imports
_TORCH_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass


# Constants
EPS = 1e-15
MAX_NORM = 1 - 1e-5


# ============================================================================
# Poincaré Ball Operations
# ============================================================================

def poincare_distance(u: "torch.Tensor", v: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """
    Compute hyperbolic distance in the Poincaré ball.

    d(u, v) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-u) ⊕_c v||)

    Args:
        u: Point in Poincaré ball, shape (..., dim)
        v: Point in Poincaré ball, shape (..., dim)
        c: Curvature parameter (default 1.0)

    Returns:
        Hyperbolic distance, shape (...)
    """
    sqrt_c = np.sqrt(c)
    diff = mobius_add(-u, v, c=c)
    norm = torch.clamp(torch.norm(diff, dim=-1), max=MAX_NORM)
    return (2.0 / sqrt_c) * torch.arctanh(sqrt_c * norm)


def mobius_add(u: "torch.Tensor", v: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """
    Möbius addition in the Poincaré ball.

    u ⊕_c v = ((1 + 2c<u,v> + c||v||²)u + (1 - c||u||²)v) /
              (1 + 2c<u,v> + c²||u||²||v||²)

    Args:
        u: First point, shape (..., dim)
        v: Second point, shape (..., dim)
        c: Curvature parameter

    Returns:
        Result of Möbius addition, shape (..., dim)
    """
    u_sq = torch.sum(u * u, dim=-1, keepdim=True)
    v_sq = torch.sum(v * v, dim=-1, keepdim=True)
    uv = torch.sum(u * v, dim=-1, keepdim=True)

    num = (1 + 2 * c * uv + c * v_sq) * u + (1 - c * u_sq) * v
    denom = 1 + 2 * c * uv + c ** 2 * u_sq * v_sq

    return num / torch.clamp(denom, min=EPS)


def mobius_scalar_mul(r: float, v: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """
    Möbius scalar multiplication.

    r ⊗_c v = (1/sqrt(c)) * tanh(r * arctanh(sqrt(c) * ||v||)) * v / ||v||

    Args:
        r: Scalar multiplier
        v: Point in Poincaré ball, shape (..., dim)
        c: Curvature parameter

    Returns:
        Scaled point, shape (..., dim)
    """
    sqrt_c = np.sqrt(c)
    norm_v = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=EPS)
    scale = torch.tanh(r * torch.arctanh(sqrt_c * norm_v)) / (sqrt_c * norm_v)
    return scale * v


def expmap0(v: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """
    Exponential map from origin in Poincaré ball.

    exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)

    Args:
        v: Tangent vector at origin, shape (..., dim)
        c: Curvature parameter

    Returns:
        Point in Poincaré ball, shape (..., dim)
    """
    sqrt_c = np.sqrt(c)
    norm_v = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=EPS)
    scale = torch.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
    return scale * v


def logmap0(v: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """
    Logarithmic map to origin in Poincaré ball.

    log_0(v) = arctanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)

    Args:
        v: Point in Poincaré ball, shape (..., dim)
        c: Curvature parameter

    Returns:
        Tangent vector at origin, shape (..., dim)
    """
    sqrt_c = np.sqrt(c)
    norm_v = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=EPS, max=MAX_NORM)
    scale = torch.arctanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
    return scale * v


def project_to_ball(x: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """
    Project points to be inside the Poincaré ball.

    Args:
        x: Points that may be outside ball, shape (..., dim)
        c: Curvature parameter

    Returns:
        Projected points, shape (..., dim)
    """
    max_radius = MAX_NORM / np.sqrt(c)
    norm = torch.norm(x, dim=-1, keepdim=True)
    clamped = torch.clamp(norm, max=max_radius)
    return x * (clamped / torch.clamp(norm, min=EPS))


# ============================================================================
# Hyperbolic Neural Network Layers
# ============================================================================

if _TORCH_AVAILABLE:

    class HyperbolicLinear(nn.Module):
        """
        Hyperbolic linear layer using Möbius operations.

        Applies a linear transformation in tangent space, then projects back.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            c: float = 1.0,
            bias: bool = True,
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.c = c

            # Euclidean weight matrix
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter("bias", None)

            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # Map to tangent space at origin
            x_tangent = logmap0(x, c=self.c)

            # Apply Euclidean linear transformation
            x_transformed = F.linear(x_tangent, self.weight, self.bias)

            # Map back to Poincaré ball
            x_hyp = expmap0(x_transformed, c=self.c)

            return project_to_ball(x_hyp, c=self.c)

    class HyperbolicMLP(nn.Module):
        """
        Multi-layer perceptron in hyperbolic space.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            output_dim: int,
            c: float = 1.0,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.c = c

            dims = [input_dim] + hidden_dims + [output_dim]
            layers = []

            for i in range(len(dims) - 1):
                layers.append(HyperbolicLinear(dims[i], dims[i + 1], c=c))
                if i < len(dims) - 2:  # No activation after last layer
                    layers.append(nn.Dropout(dropout))

            self.layers = nn.ModuleList(layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            for i, layer in enumerate(self.layers):
                if isinstance(layer, HyperbolicLinear):
                    x = layer(x)
                    if i < len(self.layers) - 1:
                        # Apply non-linearity in tangent space
                        x_tangent = logmap0(x, c=self.c)
                        x_tangent = torch.relu(x_tangent)
                        x = expmap0(x_tangent, c=self.c)
                        x = project_to_ball(x, c=self.c)
                else:
                    # Apply Euclidean layers (dropout) to tangent space
                    x_tangent = logmap0(x, c=self.c)
                    x_tangent = layer(x_tangent)
                    x = expmap0(x_tangent, c=self.c)
                    x = project_to_ball(x, c=self.c)

            return x

    class HyperbolicAttention(nn.Module):
        """
        Attention mechanism in hyperbolic space.

        Uses hyperbolic distance for attention scores.
        """

        def __init__(
            self,
            dim: int,
            num_heads: int = 4,
            c: float = 1.0,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.c = c

            self.query = HyperbolicLinear(dim, dim, c=c)
            self.key = HyperbolicLinear(dim, dim, c=c)
            self.value = HyperbolicLinear(dim, dim, c=c)
            self.out = HyperbolicLinear(dim, dim, c=c)

            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            query: "torch.Tensor",
            key: "torch.Tensor",
            value: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            batch_size = query.size(0)
            seq_len = query.size(1)

            # Transform to Q, K, V
            q = self.query(query)
            k = self.key(key)
            v = self.value(value)

            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

            # Compute attention scores using hyperbolic distance
            # scores[i,j] = -d(q_i, k_j)  (negative distance = higher attention)
            scores = torch.zeros(batch_size, seq_len, seq_len, self.num_heads)

            for i in range(seq_len):
                for j in range(seq_len):
                    dist = poincare_distance(q[:, i, :, :], k[:, j, :, :], c=self.c)
                    scores[:, i, j, :] = -dist

            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))

            # Softmax attention
            attn = F.softmax(scores, dim=2)
            attn = self.dropout(attn)

            # Apply attention to values (in tangent space)
            v_tangent = logmap0(v, c=self.c)
            out_tangent = torch.einsum("bqkh,bkhd->bqhd", attn, v_tangent)
            out = expmap0(out_tangent.view(batch_size, seq_len, self.dim), c=self.c)

            return self.out(project_to_ball(out, c=self.c))

    class HyperbolicEncoder(nn.Module):
        """
        Hyperbolic encoder for sequence data.

        Maps Euclidean inputs to hyperbolic representations.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2,
            c: float = 1.0,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.c = c

            # Initial projection to hyperbolic space
            self.input_proj = nn.Linear(input_dim, hidden_dim)

            # Hyperbolic layers
            self.layers = nn.ModuleList([
                HyperbolicLinear(hidden_dim, hidden_dim, c=c)
                for _ in range(num_layers)
            ])

            # Output projection
            self.output_proj = HyperbolicLinear(hidden_dim, output_dim, c=c)

            self.dropout = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # Project to Euclidean hidden space
            x = self.input_proj(x)

            # Map to Poincaré ball
            x = expmap0(x, c=self.c)
            x = project_to_ball(x, c=self.c)

            # Apply hyperbolic layers
            for layer in self.layers:
                x = layer(x)
                # Non-linearity in tangent space
                x_tangent = logmap0(x, c=self.c)
                x_tangent = torch.relu(x_tangent)
                x_tangent = self.dropout(x_tangent)
                x = expmap0(x_tangent, c=self.c)
                x = project_to_ball(x, c=self.c)

            return self.output_proj(x)

    class HyperbolicVAE(nn.Module):
        """
        Variational Autoencoder in hyperbolic space.

        Uses wrapped normal distribution in the Poincaré ball.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            c: float = 1.0,
        ):
            super().__init__()
            self.c = c
            self.latent_dim = latent_dim

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Mean and log-variance in tangent space
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            # Decoder
            self.decoder = nn.Sequential(
                HyperbolicLinear(latent_dim, hidden_dim, c=c),
            )

            self.output = nn.Linear(hidden_dim, input_dim)

        def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Encode to mean and log-variance in tangent space."""
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def reparameterize(
            self, mu: "torch.Tensor", logvar: "torch.Tensor"
        ) -> "torch.Tensor":
            """Sample from wrapped normal distribution."""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            # Sample in tangent space
            z_tangent = mu + eps * std

            # Map to Poincaré ball
            z = expmap0(z_tangent, c=self.c)
            return project_to_ball(z, c=self.c)

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            """Decode from hyperbolic latent space."""
            h = self.decoder(z)

            # Map to tangent space for output
            h_tangent = logmap0(h, c=self.c)
            return self.output(h_tangent)

        def forward(
            self, x: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar

        def loss(
            self,
            x: "torch.Tensor",
            recon: "torch.Tensor",
            mu: "torch.Tensor",
            logvar: "torch.Tensor",
            beta: float = 1.0,
        ) -> "torch.Tensor":
            """VAE loss with KL divergence in tangent space."""
            # Reconstruction loss
            recon_loss = F.mse_loss(recon, x, reduction="sum")

            # KL divergence (approximation for wrapped normal)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            return recon_loss + beta * kl_loss


# ============================================================================
# Utility Functions
# ============================================================================

def hyperbolic_midpoint(points: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """
    Compute the Fréchet mean (centroid) in hyperbolic space.

    Uses gradient descent optimization.

    Args:
        points: Points in Poincaré ball, shape (n_points, dim)
        c: Curvature parameter

    Returns:
        Midpoint in Poincaré ball, shape (dim,)
    """
    # Initialize at Euclidean mean
    centroid = torch.mean(points, dim=0)
    centroid = project_to_ball(centroid.unsqueeze(0), c=c).squeeze(0)

    # Gradient descent
    lr = 0.1
    for _ in range(100):
        # Compute gradient: sum of log maps
        gradients = logmap0(mobius_add(-centroid, points, c=c), c=c)
        mean_grad = torch.mean(gradients, dim=0)

        # Update centroid
        centroid = mobius_add(centroid, expmap0(lr * mean_grad, c=c), c=c)
        centroid = project_to_ball(centroid.unsqueeze(0), c=c).squeeze(0)

    return centroid


def hyperbolic_kmeans(
    points: "torch.Tensor",
    n_clusters: int,
    max_iter: int = 100,
    c: float = 1.0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    K-means clustering in hyperbolic space.

    Args:
        points: Points in Poincaré ball, shape (n_points, dim)
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        c: Curvature parameter

    Returns:
        Tuple of (cluster assignments, centroids)
    """
    n_points = points.size(0)

    # Initialize centroids randomly
    indices = torch.randperm(n_points)[:n_clusters]
    centroids = points[indices].clone()

    for _ in range(max_iter):
        # Assign points to nearest centroid
        distances = torch.zeros(n_points, n_clusters)
        for i, centroid in enumerate(centroids):
            distances[:, i] = poincare_distance(points, centroid.unsqueeze(0), c=c)

        assignments = torch.argmin(distances, dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_clusters):
            cluster_points = points[assignments == i]
            if len(cluster_points) > 0:
                new_centroids[i] = hyperbolic_midpoint(cluster_points, c=c)
            else:
                new_centroids[i] = centroids[i]

        # Check convergence
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return assignments, centroids


# ============================================================================
# HIV-Specific Applications
# ============================================================================

class HIVSequenceEmbedder(nn.Module if _TORCH_AVAILABLE else object):
    """
    Hyperbolic embedding model for HIV sequences.

    Combines codon-level p-adic encoding with hyperbolic neural networks.
    """

    def __init__(
        self,
        codon_dim: int = 3,  # p-adic codon embedding dimension
        hidden_dim: int = 64,
        output_dim: int = 16,
        c: float = 1.0,
    ):
        if _TORCH_AVAILABLE:
            super().__init__()
            self.c = c

            # Process individual codons
            self.codon_encoder = HyperbolicEncoder(
                input_dim=codon_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                c=c,
            )

            # Aggregate codons
            self.aggregator = HyperbolicLinear(hidden_dim, output_dim, c=c)

    def forward(self, codon_embeddings: "torch.Tensor") -> "torch.Tensor":
        """
        Embed a sequence of codon embeddings.

        Args:
            codon_embeddings: Shape (batch, n_codons, codon_dim)

        Returns:
            Sequence embedding, shape (batch, output_dim)
        """
        batch_size, n_codons, _ = codon_embeddings.shape

        # Encode each codon
        codon_embs = self.codon_encoder(
            codon_embeddings.view(-1, codon_embeddings.size(-1))
        )
        codon_embs = codon_embs.view(batch_size, n_codons, -1)

        # Aggregate using hyperbolic mean
        seq_embs = []
        for b in range(batch_size):
            centroid = hyperbolic_midpoint(codon_embs[b], c=self.c)
            seq_embs.append(centroid)

        seq_emb = torch.stack(seq_embs)

        return self.aggregator(seq_emb)


class MutationEffectPredictor(nn.Module if _TORCH_AVAILABLE else object):
    """
    Predict mutation effects using hyperbolic distance.

    Uses the principle that mutations to central (conserved) positions
    are more deleterious.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        c: float = 1.0,
    ):
        if _TORCH_AVAILABLE:
            super().__init__()
            self.c = c

            self.predictor = nn.Sequential(
                nn.Linear(embedding_dim * 2 + 1, 64),  # +1 for distance
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),  # Predicted fitness effect
            )

    def forward(
        self,
        wt_embedding: "torch.Tensor",
        mut_embedding: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Predict mutation effect from embeddings.

        Args:
            wt_embedding: Wild-type embedding, shape (batch, dim)
            mut_embedding: Mutant embedding, shape (batch, dim)

        Returns:
            Predicted fitness effect, shape (batch, 1)
        """
        # Hyperbolic distance
        dist = poincare_distance(wt_embedding, mut_embedding, c=self.c)

        # Map embeddings to tangent space for concatenation
        wt_tangent = logmap0(wt_embedding, c=self.c)
        mut_tangent = logmap0(mut_embedding, c=self.c)

        # Concatenate features
        features = torch.cat([
            wt_tangent,
            mut_tangent,
            dist.unsqueeze(-1),
        ], dim=-1)

        return self.predictor(features)


# Example usage
if __name__ == "__main__":
    print("Testing Hyperbolic Neural Networks Module")
    print("=" * 50)

    if not _TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        exit(0)

    # Test Poincaré ball operations
    print("\nTesting Poincaré ball operations...")

    u = torch.randn(5, 3)
    u = project_to_ball(u)
    v = torch.randn(5, 3)
    v = project_to_ball(v)

    print(f"Points u shape: {u.shape}")
    print(f"Points v shape: {v.shape}")

    # Möbius addition
    w = mobius_add(u, v)
    print(f"Möbius addition result shape: {w.shape}")

    # Distance
    d = poincare_distance(u, v)
    print(f"Distances: {d}")

    # Exp/Log maps
    tangent = logmap0(u)
    back = expmap0(tangent)
    print(f"Log-Exp roundtrip error: {torch.max(torch.abs(u - back)):.6f}")

    # Test HyperbolicLinear
    print("\nTesting HyperbolicLinear...")
    layer = HyperbolicLinear(3, 8)
    x = project_to_ball(torch.randn(10, 3))
    y = layer(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Test HyperbolicMLP
    print("\nTesting HyperbolicMLP...")
    mlp = HyperbolicMLP(3, [16, 16], 8)
    y = mlp(x)
    print(f"MLP output shape: {y.shape}")

    # Test HyperbolicEncoder
    print("\nTesting HyperbolicEncoder...")
    encoder = HyperbolicEncoder(10, 32, 16)
    x = torch.randn(5, 10)
    z = encoder(x)
    print(f"Encoder input shape: {x.shape}, output shape: {z.shape}")

    # Test HyperbolicVAE
    print("\nTesting HyperbolicVAE...")
    vae = HyperbolicVAE(20, 32, 8)
    x = torch.randn(5, 20)
    recon, mu, logvar = vae(x)
    loss = vae.loss(x, recon, mu, logvar)
    print(f"VAE reconstruction shape: {recon.shape}")
    print(f"VAE loss: {loss.item():.4f}")

    # Test hyperbolic clustering
    print("\nTesting hyperbolic K-means...")
    points = project_to_ball(torch.randn(100, 3))
    assignments, centroids = hyperbolic_kmeans(points, n_clusters=5)
    print(f"Cluster assignments: {torch.bincount(assignments)}")

    # Test HIV-specific models
    print("\nTesting HIVSequenceEmbedder...")
    embedder = HIVSequenceEmbedder(codon_dim=3, hidden_dim=32, output_dim=16)
    codon_embs = torch.randn(4, 10, 3)  # 4 sequences, 10 codons each
    seq_embs = embedder(codon_embs)
    print(f"Sequence embeddings shape: {seq_embs.shape}")

    # Test MutationEffectPredictor
    print("\nTesting MutationEffectPredictor...")
    predictor = MutationEffectPredictor(embedding_dim=16)
    wt_emb = project_to_ball(torch.randn(4, 16))
    mut_emb = project_to_ball(torch.randn(4, 16))
    effects = predictor(wt_emb, mut_emb)
    print(f"Predicted effects shape: {effects.shape}")
    print(f"Predicted effects: {effects.squeeze().tolist()}")

    print("\n" + "=" * 50)
    print("Module testing complete!")
