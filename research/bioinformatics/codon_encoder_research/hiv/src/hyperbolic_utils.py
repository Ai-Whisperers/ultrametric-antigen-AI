"""
Hyperbolic Utilities for RA Bioinformatics Analysis

Provides Poincaré ball geometry functions for computing distances
and projecting embeddings in hyperbolic space.

The Poincaré ball model naturally captures hierarchical/tree-like
structures, which aligns with the p-adic ultrametric hypothesis
for codon organization.

Version History:
- v1.0: Original with post-hoc Euclidean→Poincaré projection
- v2.0: Native hyperbolic via codon-encoder-3-adic (trained on V5.11.3)
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# POINCARÉ BALL DISTANCE
# ============================================================================


def poincare_distance(
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    c: float = 1.0,
    eps: float = 1e-10,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute Poincaré ball distance between points.

    Formula:
        d(x, y) = (1/√c) * arcosh(1 + 2c * ||x-y||² / ((1 - c||x||²)(1 - c||y||²)))

    Args:
        x: First point(s), shape (..., dim)
        y: Second point(s), shape (..., dim)
        c: Curvature parameter (default 1.0)
        eps: Small value for numerical stability

    Returns:
        Distance(s), shape (...)
    """
    is_numpy = isinstance(x, np.ndarray)

    if is_numpy:
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

    # Compute norms squared
    x_norm_sq = torch.sum(x**2, dim=-1)
    y_norm_sq = torch.sum(y**2, dim=-1)
    diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)

    # Compute denominator with stability
    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    denom = torch.clamp(denom, min=eps)

    # Compute arcosh argument
    arg = 1 + 2 * c * diff_norm_sq / denom
    arg = torch.clamp(arg, min=1.0 + eps)  # arcosh requires arg >= 1

    # Final distance
    dist = (1 / np.sqrt(c)) * torch.acosh(arg)

    if is_numpy:
        return dist.numpy()
    return dist


def poincare_distance_matrix(
    X: Union[np.ndarray, torch.Tensor],
    Y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    c: float = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute pairwise Poincaré distances.

    Args:
        X: Points, shape (n, dim)
        Y: Points, shape (m, dim). If None, computes X vs X.
        c: Curvature parameter

    Returns:
        Distance matrix, shape (n, m)
    """
    is_numpy = isinstance(X, np.ndarray)

    if Y is None:
        Y = X

    if is_numpy:
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()

    n, m = X.shape[0], Y.shape[0]

    # Expand for broadcasting
    X_exp = X.unsqueeze(1).expand(n, m, -1)  # (n, m, dim)
    Y_exp = Y.unsqueeze(0).expand(n, m, -1)  # (n, m, dim)

    dist = poincare_distance(X_exp, Y_exp, c=c)

    if is_numpy:
        return dist.numpy()
    return dist


# ============================================================================
# PROJECTION TO POINCARÉ BALL
# ============================================================================


def project_to_poincare(
    z: Union[np.ndarray, torch.Tensor],
    max_radius: float = 0.95,
    method: str = "normalize",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Project Euclidean vectors to Poincaré ball.

    Args:
        z: Euclidean vectors, shape (..., dim)
        max_radius: Maximum radius in ball (must be < 1)
        method: Projection method
            - 'normalize': Scale to fit within ball
            - 'tanh': Smooth mapping via tanh

    Returns:
        Projected vectors on Poincaré ball
    """
    is_numpy = isinstance(z, np.ndarray)

    if is_numpy:
        z = torch.from_numpy(z).float()

    if method == "normalize":
        norm = torch.norm(z, dim=-1, keepdim=True)
        # Scale so max norm is max_radius
        scale = max_radius / (1 + norm)
        z_hyp = z * scale
    elif method == "tanh":
        norm = torch.norm(z, dim=-1, keepdim=True)
        direction = z / (norm + 1e-10)
        radius = torch.tanh(norm) * max_radius
        z_hyp = direction * radius
    else:
        raise ValueError(f"Unknown projection method: {method}")

    if is_numpy:
        return z_hyp.numpy()
    return z_hyp


# ============================================================================
# CODON ENCODER WITH HYPERBOLIC PROJECTION
# ============================================================================


class CodonEncoder(nn.Module):
    """Codon encoder network (same architecture as training)."""

    def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.cluster_head = nn.Linear(embed_dim, n_clusters)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.1)

    def encode(self, x):
        return self.encoder(x)

    def get_cluster(self, x):
        emb = self.encode(x)
        logits = self.cluster_head(emb)
        return torch.argmax(logits, dim=-1), emb

    def get_cluster_probs(self, x):
        emb = self.encode(x)
        logits = self.cluster_head(emb)
        probs = torch.softmax(logits, dim=-1)
        return probs, emb


class HyperbolicCodonEncoder:
    """
    Wrapper that combines CodonEncoder with hyperbolic projection.

    Provides a unified interface for encoding codons to hyperbolic space.

    For legacy encoders (v5.5 Euclidean), applies post-hoc projection.
    For 3-adic encoders (V5.11.3), embeddings are already hyperbolic.
    """

    def __init__(
        self,
        encoder: CodonEncoder,
        max_radius: float = 0.95,
        projection_method: str = "normalize",
        native_hyperbolic: bool = False,
    ):
        self.encoder = encoder
        self.max_radius = max_radius
        self.projection_method = projection_method
        self.native_hyperbolic = native_hyperbolic  # True for 3-adic encoder
        self.device = next(encoder.parameters()).device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to hyperbolic space."""
        with torch.no_grad():
            emb = self.encoder.encode(x)
            if self.native_hyperbolic:
                # 3-adic encoder already produces hyperbolic embeddings
                return emb
            return project_to_poincare(emb, self.max_radius, self.projection_method)

    def encode_numpy(self, x: np.ndarray) -> np.ndarray:
        """Encode numpy array to hyperbolic space."""
        x_tensor = torch.from_numpy(x).float().to(self.device)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)
        return self.encode(x_tensor).cpu().numpy()

    def get_cluster(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cluster assignment and hyperbolic embedding."""
        with torch.no_grad():
            cluster_id, emb = self.encoder.get_cluster(x)
            if self.native_hyperbolic:
                return cluster_id, emb
            emb_hyp = project_to_poincare(emb, self.max_radius, self.projection_method)
            return cluster_id, emb_hyp


# ============================================================================
# LOADING UTILITIES
# ============================================================================


def load_codon_encoder(device: str = "cpu", version: str = "3adic") -> Tuple[CodonEncoder, dict, bool]:
    """
    Load the trained codon encoder and mapping.

    Args:
        device: Device to load model on
        version: Encoder version
            - '3adic': Native hyperbolic (trained on V5.11.3) - DEFAULT
            - 'legacy': Original encoder (trained on v5.5 Euclidean)

    Returns:
        Tuple of (encoder, codon_to_position_mapping, native_hyperbolic)
    """
    script_dir = Path(__file__).parent
    research_dir = script_dir.parent.parent.parent

    # Additional paths for different project structures
    # script_dir = .../hiv/scripts
    # parents: [0]=scripts, [1]=hiv, [2]=codon_encoder_research, [3]=bioinformatics,
    #          [4]=03_EXPERIMENTS_AND_LABS, [5]=01_PROJECT_KNOWLEDGE_BASE, [6]=DOCUMENTATION, [7]=ternary-vaes
    project_root = script_dir.parents[6]  # Navigate to ternary-vaes root

    if version == "3adic":
        # New native hyperbolic encoder (trained on V5.11.3 embeddings)
        encoder_paths = [
            research_dir / "genetic_code" / "data" / "codon_encoder_3adic.pt",
            script_dir.parent / "data" / "codon_encoder_3adic.pt",
            # Project-level paths
            project_root / "research" / "bioinformatics" / "genetic_code" / "data" / "codon_encoder_3adic.pt",
            project_root
            / "DOCUMENTATION"
            / "01_PROJECT_KNOWLEDGE_BASE"
            / "03_EXPERIMENTS_AND_LABS"
            / "bioinformatics"
            / "codon_encoder_research"
            / "rheumatoid_arthritis"
            / "data"
            / "codon_encoder_3adic.pt",
        ]
        native_hyperbolic = True
    else:
        # Legacy encoder (trained on v5.5 Euclidean, needs projection)
        encoder_paths = [
            research_dir / "genetic_code" / "data" / "legacy" / "codon_encoder_legacy.pt",
            script_dir.parent / "data" / "legacy" / "codon_encoder_legacy.pt",
        ]
        native_hyperbolic = False

    encoder_path = None
    for path in encoder_paths:
        if path.exists():
            encoder_path = path
            break

    if encoder_path is None:
        raise FileNotFoundError(f"Codon encoder not found (version={version})")

    encoder = CodonEncoder(input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21)
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["model_state"])
    encoder.eval()
    encoder.to(device)

    # Get mapping
    mapping = checkpoint.get("codon_to_position", {})

    return encoder, mapping, native_hyperbolic


def load_hyperbolic_encoder(device: str = "cpu", max_radius: float = 0.95, version: str = "3adic") -> Tuple[HyperbolicCodonEncoder, dict]:
    """
    Load encoder with hyperbolic projection wrapper.

    Args:
        device: Device to load model on
        max_radius: Maximum radius for post-hoc projection (legacy only)
        version: Encoder version ('3adic' or 'legacy')

    Returns:
        Tuple of (hyperbolic_encoder, codon_to_position_mapping)
    """
    encoder, mapping, native_hyperbolic = load_codon_encoder(device, version=version)
    hyp_encoder = HyperbolicCodonEncoder(encoder, max_radius=max_radius, native_hyperbolic=native_hyperbolic)
    return hyp_encoder, mapping


# ============================================================================
# CODON ENCODING UTILITIES
# ============================================================================


def codon_to_onehot(codon: str) -> np.ndarray:
    """Convert codon string to one-hot encoding (12-dim)."""
    nucleotides = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    onehot = np.zeros(12)
    for i, nuc in enumerate(codon.upper()):
        if nuc in nucleotides:
            onehot[i * 4 + nucleotides[nuc]] = 1
    return onehot


def encode_codon_hyperbolic(codon: str, encoder: HyperbolicCodonEncoder) -> np.ndarray:
    """
    Encode a single codon to hyperbolic space.

    Args:
        codon: 3-letter codon string (e.g., 'ATG')
        encoder: Hyperbolic encoder wrapper

    Returns:
        16-dim hyperbolic embedding
    """
    onehot = codon_to_onehot(codon)
    return encoder.encode_numpy(onehot).squeeze()


def encode_sequence_hyperbolic(aa_sequence: str, encoder: HyperbolicCodonEncoder, aa_to_codon: dict) -> np.ndarray:
    """
    Encode amino acid sequence to hyperbolic embeddings.

    Args:
        aa_sequence: Amino acid sequence string
        encoder: Hyperbolic encoder wrapper
        aa_to_codon: Mapping from amino acids to codons

    Returns:
        Array of embeddings, shape (len(sequence), 16)
    """
    embeddings = []
    for aa in aa_sequence.upper():
        codon = aa_to_codon.get(aa, "NNN")
        if codon != "NNN":
            emb = encode_codon_hyperbolic(codon, encoder)
            embeddings.append(emb)
    return np.array(embeddings)


# ============================================================================
# HYPERBOLIC GEOMETRY UTILITIES
# ============================================================================


def hyperbolic_centroid(
    points: Union[np.ndarray, torch.Tensor],
    c: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute Fréchet mean (centroid) in Poincaré ball.

    Uses iterative algorithm since hyperbolic mean has no closed form.

    Args:
        points: Points on Poincaré ball, shape (n, dim)
        c: Curvature parameter
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Centroid point, shape (dim,)
    """
    is_numpy = isinstance(points, np.ndarray)

    if is_numpy:
        points = torch.from_numpy(points).float()

    # Initialize with Euclidean mean (projected to ball)
    centroid = project_to_poincare(points.mean(dim=0, keepdim=True), max_radius=0.9).squeeze()

    for _ in range(max_iter):
        # Compute distances to current centroid
        dists = poincare_distance(points, centroid.unsqueeze(0).expand(points.shape[0], -1), c=c)

        # Weighted update (simplified gradient descent in tangent space)
        weights = 1 / (dists + 1e-10)
        weights = weights / weights.sum()

        # Move toward weighted mean in tangent space
        new_centroid = (points * weights.unsqueeze(-1)).sum(dim=0)
        new_centroid = project_to_poincare(new_centroid.unsqueeze(0), max_radius=0.9).squeeze()

        # Check convergence
        shift = torch.norm(new_centroid - centroid)
        centroid = new_centroid

        if shift < tol:
            break

    if is_numpy:
        return centroid.numpy()
    return centroid


def hyperbolic_variance(
    points: Union[np.ndarray, torch.Tensor],
    centroid: Optional[Union[np.ndarray, torch.Tensor]] = None,
    c: float = 1.0,
) -> float:
    """
    Compute variance in Poincaré ball (mean squared geodesic distance).

    Args:
        points: Points on Poincaré ball, shape (n, dim)
        centroid: Pre-computed centroid (computed if None)
        c: Curvature parameter

    Returns:
        Variance (scalar)
    """
    is_numpy = isinstance(points, np.ndarray)

    if is_numpy:
        points = torch.from_numpy(points).float()
        if centroid is not None:
            centroid = torch.from_numpy(centroid).float()

    if centroid is None:
        centroid = hyperbolic_centroid(points, c=c)

    dists = poincare_distance(points, centroid.unsqueeze(0).expand(points.shape[0], -1), c=c)
    variance = (dists**2).mean().item()

    return variance


# ============================================================================
# AMINO ACID CODON TABLE
# ============================================================================

AA_TO_CODON = {
    "A": "GCT",
    "R": "CGG",
    "N": "AAC",
    "D": "GAC",
    "C": "TGC",
    "E": "GAG",
    "Q": "CAG",
    "G": "GGC",
    "H": "CAC",
    "I": "ATC",
    "L": "CTG",
    "K": "AAG",
    "M": "ATG",
    "F": "TTC",
    "P": "CCG",
    "S": "TCG",
    "T": "ACC",
    "W": "TGG",
    "Y": "TAC",
    "V": "GTG",
    "*": "TGA",
}

# All arginine codons
ARGININE_CODONS = ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"]


# ============================================================================
# RESULTS DIRECTORY UTILITIES
# ============================================================================


def get_results_dir(hyperbolic: bool = True) -> Path:
    """Get the appropriate results directory."""
    script_dir = Path(__file__).parent
    results_base = script_dir.parent / "results"

    if hyperbolic:
        results_dir = results_base / "hyperbolic"
    else:
        results_dir = results_base / "euclidean"

    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
