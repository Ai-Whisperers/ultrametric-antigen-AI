# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""
Diffusion Map Encoder for Manifold Learning.

Implementation based on Coifman & Lafon (2006) diffusion maps framework,
adapted for biological sequence analysis with p-adic distance integration.
Enables discovery of intrinsic geometric structure in high-dimensional
biological data (e.g., protein conformations, gene expression landscapes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionMapResult:
    """Result of diffusion map embedding."""

    coordinates: torch.Tensor  # (batch, n_components)
    eigenvalues: torch.Tensor  # (batch, n_components)
    diffusion_distances: torch.Tensor  # (batch, n_points, n_points)
    diffusion_time: float


class KernelBuilder(nn.Module):
    """
    Build adaptive kernels for diffusion maps.

    Supports multiple kernel types with automatic bandwidth selection.
    """

    def __init__(
        self,
        kernel_type: str = "gaussian",
        adaptive_bandwidth: bool = True,
        n_neighbors: int = 10,
        alpha: float = 0.5,
    ):
        """
        Initialize kernel builder.

        Args:
            kernel_type: Type of kernel ("gaussian", "cosine", "padic")
            adaptive_bandwidth: Use local bandwidth estimation
            n_neighbors: Number of neighbors for bandwidth estimation
            alpha: Normalization parameter (0=graph Laplacian, 1=Laplace-Beltrami)
        """
        super().__init__()
        self.kernel_type = kernel_type
        self.adaptive_bandwidth = adaptive_bandwidth
        self.n_neighbors = n_neighbors
        self.alpha = alpha

    def compute_pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances."""
        # (batch, n, d)
        x_norm = (x**2).sum(dim=-1, keepdim=True)
        dist_sq = x_norm + x_norm.transpose(-2, -1) - 2 * torch.bmm(x, x.transpose(-2, -1))
        return torch.sqrt(torch.clamp(dist_sq, min=1e-10))

    def estimate_bandwidth(self, distances: torch.Tensor) -> torch.Tensor:
        """Estimate local bandwidth using k-nearest neighbors."""
        # Sort distances to find k-th nearest neighbor
        k = min(self.n_neighbors, distances.shape[-1] - 1)
        sorted_dists, _ = torch.sort(distances, dim=-1)
        # Use k-th neighbor distance as bandwidth
        epsilon = sorted_dists[..., k : k + 1]  # (batch, n, 1)
        return torch.clamp(epsilon, min=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build kernel matrix from input features.

        Args:
            x: Input features (batch, n_points, d)

        Returns:
            Kernel matrix (batch, n_points, n_points)
        """
        distances = self.compute_pairwise_distances(x)

        if self.adaptive_bandwidth:
            epsilon = self.estimate_bandwidth(distances)
            # Use geometric mean of bandwidths for symmetry
            epsilon_ij = torch.sqrt(epsilon * epsilon.transpose(-2, -1))
        else:
            # Global bandwidth: median of all distances
            epsilon_ij = distances.median(dim=-1, keepdim=True)[0].median(dim=-2, keepdim=True)[0]
            epsilon_ij = epsilon_ij.expand_as(distances)

        if self.kernel_type == "gaussian":
            kernel = torch.exp(-(distances**2) / (2 * epsilon_ij**2 + 1e-10))
        elif self.kernel_type == "cosine":
            # Cosine similarity kernel
            x_norm = F.normalize(x, p=2, dim=-1)
            kernel = torch.bmm(x_norm, x_norm.transpose(-2, -1))
            kernel = (kernel + 1) / 2  # Map to [0, 1]
        elif self.kernel_type == "padic":
            # P-adic inspired kernel: use 3-adic valuation
            # Treat features as ternary encodings
            kernel = self._padic_kernel(x, distances)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Apply alpha-normalization (density correction)
        if self.alpha > 0:
            q = kernel.sum(dim=-1, keepdim=True)  # Row sums (density estimate)
            q_alpha = q**self.alpha
            kernel = kernel / (q_alpha * q_alpha.transpose(-2, -1) + 1e-10)

        return kernel

    def _padic_kernel(self, x: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """Compute p-adic inspired kernel based on 3-adic distances."""
        # Discretize to ternary representation
        x_int = (x * 100).long() % 81  # Map to 0-80 (ternary digits)

        batch_size, n_points, dim = x.shape
        padic_dist = torch.zeros(batch_size, n_points, n_points, device=x.device)

        for d in range(dim):
            diff = (x_int[..., d : d + 1] - x_int[..., d : d + 1].transpose(-2, -1)).abs()
            # 3-adic valuation approximation
            v3 = torch.zeros_like(diff, dtype=torch.float32)
            for power in range(4):
                divisible = (diff % (3 ** (power + 1))) == 0
                v3 = torch.where(divisible & (diff > 0), torch.tensor(power + 1.0, device=x.device), v3)
            padic_dist += 3.0 ** (-v3)

        # Normalize and convert to kernel
        padic_dist = padic_dist / dim
        return torch.exp(-padic_dist)


class DiffusionMapEncoder(nn.Module):
    """
    Diffusion Map Encoder for manifold learning.

    Uses random walk diffusion on a graph to discover intrinsic
    geometric structure. Particularly suited for:
    - Protein conformational landscapes
    - Gene expression trajectories
    - Evolutionary sequence relationships
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_components: int = 16,
        diffusion_time: float = 1.0,
        kernel_type: str = "gaussian",
        alpha: float = 0.5,
        n_eigenvectors: int = 32,
    ):
        """
        Initialize diffusion map encoder.

        Args:
            input_dim: Dimension of input features
            n_components: Number of diffusion coordinates
            diffusion_time: Diffusion time parameter
            kernel_type: Kernel type for affinity computation
            alpha: Normalization parameter
            n_eigenvectors: Number of eigenvectors to compute
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components
        self.diffusion_time = diffusion_time
        self.n_eigenvectors = n_eigenvectors

        # Feature preprocessor
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )

        # Kernel builder
        self.kernel = KernelBuilder(
            kernel_type=kernel_type,
            adaptive_bandwidth=True,
            alpha=alpha,
        )

        # Learnable output projection
        self.output_proj = nn.Linear(n_eigenvectors, n_components)

    def compute_transition_matrix(self, kernel: torch.Tensor) -> torch.Tensor:
        """Compute row-stochastic transition matrix from kernel."""
        # Normalize rows to get transition probabilities
        row_sums = kernel.sum(dim=-1, keepdim=True)
        P = kernel / (row_sums + 1e-10)
        return P

    def diffusion_power(self, P: torch.Tensor, t: float) -> torch.Tensor:
        """Compute P^t using eigendecomposition for non-integer t."""
        if t == int(t):
            # Integer power: use matrix power
            result = P
            for _ in range(int(t) - 1):
                result = torch.bmm(result, P)
            return result

        # Non-integer: use eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(P)
        # P^t = V * diag(lambda^t) * V^T
        lambda_t = torch.pow(torch.abs(eigenvalues) + 1e-10, t)
        result = torch.bmm(eigenvectors * lambda_t.unsqueeze(-2), eigenvectors.transpose(-2, -1))
        return result

    def compute_diffusion_coordinates(self, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute diffusion coordinates from transition matrix.

        Returns eigenvectors scaled by eigenvalues^t (diffusion coordinates).
        """
        # Symmetrize for stable eigendecomposition
        # D^(-1/2) * K * D^(-1/2) is symmetric
        row_sums = P.sum(dim=-1)
        D_inv_sqrt = 1.0 / torch.sqrt(row_sums + 1e-10)
        P_sym = D_inv_sqrt.unsqueeze(-1) * P * D_inv_sqrt.unsqueeze(-2)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(P_sym)

        # Sort by eigenvalue (descending)
        idx = torch.argsort(eigenvalues, dim=-1, descending=True)
        eigenvalues = torch.gather(eigenvalues, -1, idx)
        eigenvectors = torch.gather(eigenvectors, -1, idx.unsqueeze(-2).expand_as(eigenvectors))

        # Take top k eigenvectors (skip first trivial one)
        available = eigenvalues.shape[-1] - 1
        k = min(self.n_eigenvectors, available)
        if k <= 0:
            k = 1  # Need at least one
        eigenvalues = eigenvalues[..., 1 : k + 1]
        eigenvectors = eigenvectors[..., 1 : k + 1]

        # Scale by eigenvalue^t for diffusion coordinates
        lambda_t = torch.pow(torch.abs(eigenvalues) + 1e-10, self.diffusion_time)
        coordinates = eigenvectors * lambda_t.unsqueeze(-2)

        # Undo symmetrization
        coordinates = coordinates * D_inv_sqrt.unsqueeze(-1)

        # Pad if needed to match n_eigenvectors
        if coordinates.shape[-1] < self.n_eigenvectors:
            padding = torch.zeros(
                *coordinates.shape[:-1], self.n_eigenvectors - coordinates.shape[-1], device=coordinates.device
            )
            coordinates = torch.cat([coordinates, padding], dim=-1)
            eigenvalue_padding = torch.zeros(
                *eigenvalues.shape[:-1], self.n_eigenvectors - eigenvalues.shape[-1], device=eigenvalues.device
            )
            eigenvalues = torch.cat([eigenvalues, eigenvalue_padding], dim=-1)

        return coordinates, eigenvalues

    def compute_diffusion_distance(self, P: torch.Tensor) -> torch.Tensor:
        """Compute diffusion distance matrix."""
        P_t = self.diffusion_power(P, self.diffusion_time)

        # Diffusion distance: ||P_t(x, .) - P_t(y, .)||
        # Expand for pairwise computation
        P_t_x = P_t.unsqueeze(-2)  # (batch, n, 1, n)
        P_t_y = P_t.unsqueeze(-3)  # (batch, 1, n, n)

        diff = P_t_x - P_t_y
        dist = torch.sqrt((diff**2).sum(dim=-1) + 1e-10)

        return dist

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Compute diffusion map embedding.

        Args:
            x: Input features (batch, n_points, input_dim)

        Returns:
            Dictionary with diffusion coordinates and metadata
        """
        # Project features
        x_proj = self.feature_proj(x)

        # Build kernel
        K = self.kernel(x_proj)

        # Compute transition matrix
        P = self.compute_transition_matrix(K)

        # Get diffusion coordinates
        coordinates, eigenvalues = self.compute_diffusion_coordinates(P)

        # Project to output dimension
        output = self.output_proj(coordinates)

        # Compute diffusion distances
        diff_dist = self.compute_diffusion_distance(P)

        return {
            "z": output,
            "coordinates": coordinates,
            "eigenvalues": eigenvalues,
            "diffusion_distances": diff_dist,
            "kernel_matrix": K,
            "transition_matrix": P,
        }

    def embed_new_points(self, x_new: torch.Tensor, x_train: torch.Tensor, train_coords: torch.Tensor) -> torch.Tensor:
        """
        Embed new points using Nystrom extension.

        Args:
            x_new: New points to embed (batch, n_new, d)
            x_train: Training points (batch, n_train, d)
            train_coords: Training diffusion coordinates (batch, n_train, k)

        Returns:
            Embedded new points (batch, n_new, k)
        """
        x_new_proj = self.feature_proj(x_new)
        x_train_proj = self.feature_proj(x_train)

        # Compute kernel between new and training points
        x_all = torch.cat([x_train_proj, x_new_proj], dim=1)
        K_all = self.kernel(x_all)

        n_train = x_train.shape[1]
        K_new_train = K_all[:, n_train:, :n_train]

        # Normalize
        row_sums = K_new_train.sum(dim=-1, keepdim=True)
        K_normalized = K_new_train / (row_sums + 1e-10)

        # Nystrom extension: new_coords = K_normalized @ train_coords
        new_coords = torch.bmm(K_normalized, train_coords)

        return new_coords


class MultiscaleDiffusion(nn.Module):
    """
    Multi-scale diffusion for capturing structure at different resolutions.

    Uses multiple diffusion times to capture both local and global geometry.
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 32,
        n_scales: int = 4,
        time_range: tuple[float, float] = (0.5, 8.0),
    ):
        """
        Initialize multi-scale diffusion encoder.

        Args:
            input_dim: Input feature dimension
            output_dim: Output embedding dimension
            n_scales: Number of diffusion scales
            time_range: (min_time, max_time) for diffusion
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_scales = n_scales

        # Per-scale output dimension
        self.per_scale_dim = output_dim // n_scales
        self.concat_dim = self.per_scale_dim * n_scales

        # Create encoders at different scales
        times = torch.logspace(
            torch.log10(torch.tensor(time_range[0])),
            torch.log10(torch.tensor(time_range[1])),
            n_scales,
        )
        self.diffusion_times = nn.Parameter(times, requires_grad=False)

        self.encoders = nn.ModuleList(
            [
                DiffusionMapEncoder(
                    input_dim=input_dim,
                    n_components=self.per_scale_dim,
                    diffusion_time=float(times[i]),
                )
                for i in range(n_scales)
            ]
        )

        # Fusion layer - input is concatenated scale outputs
        self.fusion = nn.Sequential(
            nn.Linear(self.concat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Compute multi-scale diffusion embedding.

        Args:
            x: Input features (batch, n_points, input_dim)

        Returns:
            Dictionary with multi-scale embeddings
        """
        scale_outputs = []
        scale_eigenvalues = []

        for encoder in self.encoders:
            result = encoder(x)
            scale_outputs.append(result["z"])
            scale_eigenvalues.append(result["eigenvalues"])

        # Concatenate scale outputs
        z_concat = torch.cat(scale_outputs, dim=-1)

        # Fuse
        z_fused = self.fusion(z_concat)

        return {
            "z": z_fused,
            "scale_embeddings": scale_outputs,
            "scale_eigenvalues": scale_eigenvalues,
            "diffusion_times": self.diffusion_times,
        }


class DiffusionPseudotime(nn.Module):
    """
    Compute pseudotime ordering using diffusion maps.

    Useful for trajectory inference in biological processes
    (e.g., cell differentiation, evolutionary paths).
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_waypoints: int = 10,
    ):
        """
        Initialize pseudotime computation module.

        Args:
            input_dim: Input feature dimension
            n_waypoints: Number of waypoints for trajectory
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_waypoints = n_waypoints

        self.diffusion = DiffusionMapEncoder(
            input_dim=input_dim,
            n_components=3,  # Use first 3 components
            diffusion_time=1.0,
        )

        # Waypoint selector
        self.waypoint_proj = nn.Linear(3, n_waypoints)

    def forward(self, x: torch.Tensor, root_idx: int | None = None) -> dict[str, Any]:
        """
        Compute pseudotime from diffusion coordinates.

        Args:
            x: Input features (batch, n_points, input_dim)
            root_idx: Optional root point index for ordering

        Returns:
            Dictionary with pseudotime values and trajectory
        """
        result = self.diffusion(x)
        coords = result["z"]

        # First diffusion coordinate approximates pseudotime
        # when there's a clear trajectory
        pseudotime = coords[..., 0]

        if root_idx is not None:
            # Orient pseudotime from root
            root_time = pseudotime[:, root_idx : root_idx + 1]
            pseudotime = pseudotime - root_time
            pseudotime = torch.abs(pseudotime)

        # Normalize to [0, 1]
        pt_min = pseudotime.min(dim=-1, keepdim=True)[0]
        pt_max = pseudotime.max(dim=-1, keepdim=True)[0]
        pseudotime = (pseudotime - pt_min) / (pt_max - pt_min + 1e-10)

        # Compute waypoints along trajectory
        waypoint_scores = self.waypoint_proj(coords)
        waypoint_probs = F.softmax(waypoint_scores, dim=-2)

        return {
            "pseudotime": pseudotime,
            "diffusion_coordinates": coords,
            "waypoint_probabilities": waypoint_probs,
            "eigenvalues": result["eigenvalues"],
        }

    def order_by_pseudotime(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Order points by pseudotime.

        Returns:
            (ordered_indices, pseudotime_values)
        """
        result = self.forward(x)
        pseudotime = result["pseudotime"]

        indices = torch.argsort(pseudotime, dim=-1)

        return indices, pseudotime
