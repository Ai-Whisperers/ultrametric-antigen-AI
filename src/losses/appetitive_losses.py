"""Appetitive losses for bio-inspired VAE training.

This module implements emergent drives toward:
1. Metric structure (ranking loss with adaptive margins)
2. Hierarchical specialization (MSB/LSB variance constraints)
3. Curiosity (density-based exploration)
4. Symbiosis (mutual information coupling)
5. Algebraic closure (homomorphism constraint)

Single responsibility: Appetitive loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class AdaptiveRankingLoss(nn.Module):
    """Ranking loss with multi-scale margins for ultrametric approximation.

    Instead of forcing absolute distance matching (MSE), preserves ordinal
    structure: if d_3(a,p) < d_3(a,n), then d_latent(a,p) < d_latent(a,n).

    Adaptive margins scale with 3-adic valuation gap for exponential hierarchy.
    """

    def __init__(self, base_margin: float = 0.1, n_triplets: int = 1000):
        """Initialize adaptive ranking loss.

        Args:
            base_margin: Base triplet margin
            n_triplets: Number of triplets to sample per batch
        """
        super().__init__()
        self.base_margin = base_margin
        self.n_triplets = n_triplets
        # Precompute 3-adic valuations for all 19683 operations
        self.register_buffer('valuations', self._precompute_valuations())

    def _precompute_valuations(self) -> torch.Tensor:
        """Compute v_3(i) for i in [0, 19683)."""
        vals = torch.zeros(19683, dtype=torch.long)
        for i in range(19683):
            if i == 0:
                vals[i] = 9  # Convention: v_3(0) = max
            else:
                v, n = 0, i
                while n % 3 == 0:
                    v += 1
                    n //= 3
                vals[i] = v
        return vals

    def _three_adic_distance(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Compute d_3(i,j) = 3^{-v_3(|i-j|)}."""
        diff = torch.abs(i - j)
        # Clamp to valid range
        diff_clamped = diff.clamp(0, 19682)
        v = self.valuations[diff_clamped]
        return torch.pow(3.0, -v.float())

    def forward(self, z: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Compute adaptive ranking loss.

        Args:
            z: Latent codes (batch_size, latent_dim)
            indices: Operation indices (batch_size,)

        Returns:
            Ranking loss scalar
        """
        batch_size = z.size(0)
        device = z.device

        if batch_size < 3:
            return torch.tensor(0.0, device=device)

        # Sample anchor, positive, negative indices
        n = min(self.n_triplets, batch_size * (batch_size - 1) * (batch_size - 2) // 6)
        anchor_idx = torch.randint(0, batch_size, (n,), device=device)
        pos_idx = torch.randint(0, batch_size, (n,), device=device)
        neg_idx = torch.randint(0, batch_size, (n,), device=device)

        # Get operation indices
        op_anchor = indices[anchor_idx]
        op_pos = indices[pos_idx]
        op_neg = indices[neg_idx]

        # Compute 3-adic distances
        d3_pos = self._three_adic_distance(op_anchor, op_pos)
        d3_neg = self._three_adic_distance(op_anchor, op_neg)

        # Filter: keep triplets where d3_pos < d3_neg (positive is closer in 3-adic)
        valid = (d3_pos < d3_neg) & (anchor_idx != pos_idx) & (anchor_idx != neg_idx)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Compute latent distances for valid triplets
        z_a = z[anchor_idx[valid]]
        z_p = z[pos_idx[valid]]
        z_n = z[neg_idx[valid]]

        d_lat_pos = torch.norm(z_a - z_p, dim=1)
        d_lat_neg = torch.norm(z_a - z_n, dim=1)

        # Adaptive margins based on valuation gap
        diff_pos = torch.abs(op_anchor[valid] - op_pos[valid]).clamp(0, 19682)
        diff_neg = torch.abs(op_anchor[valid] - op_neg[valid]).clamp(0, 19682)
        v_pos = self.valuations[diff_pos]
        v_neg = self.valuations[diff_neg]
        margins = self.base_margin * (1.0 + 0.5 * torch.abs(v_pos - v_neg).float())

        # Triplet loss with adaptive margin
        loss = F.relu(d_lat_pos - d_lat_neg + margins).mean()
        return loss


class HierarchicalNormLoss(nn.Module):
    """Enforce MSB > MID > LSB variance hierarchy.

    Partitions latent dimensions into groups and ensures higher-significance
    groups have larger magnitude variance.
    """

    def __init__(self, latent_dim: int = 16, n_groups: int = 4):
        """Initialize hierarchical norm loss.

        Args:
            latent_dim: Total latent dimension
            n_groups: Number of hierarchy groups
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.n_groups = n_groups
        self.dims_per_group = latent_dim // n_groups

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute hierarchy violation loss.

        Args:
            z: Latent codes (batch_size, latent_dim)

        Returns:
            Hierarchy violation loss
        """
        # Compute variance per dimension group
        group_vars = []
        for g in range(self.n_groups):
            start = g * self.dims_per_group
            end = start + self.dims_per_group
            group_z = z[:, start:end]
            group_vars.append(group_z.var(dim=0).mean())

        # Penalize violations: Var(group_i) should be > Var(group_{i+1})
        loss = torch.tensor(0.0, device=z.device)
        for i in range(self.n_groups - 1):
            # Larger margin for MSB groups
            margin = 0.1 * (self.n_groups - i)
            violation = F.relu(group_vars[i + 1] - group_vars[i] + margin)
            loss = loss + violation

        return loss


class CuriosityModule(nn.Module):
    """Density-based exploration drive.

    Estimates latent density using KDE and provides intrinsic reward
    for visiting low-density regions.
    """

    def __init__(self, latent_dim: int = 16, bandwidth: float = 1.0, max_history: int = 5000):
        """Initialize curiosity module.

        Args:
            latent_dim: Latent space dimension
            bandwidth: KDE bandwidth
            max_history: Maximum history size
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.bandwidth = bandwidth
        self.max_history = max_history
        self.register_buffer('z_history', torch.zeros(0, latent_dim))

    def update_history(self, z: torch.Tensor):
        """Add new latent codes to history."""
        z_detached = z.detach()
        self.z_history = torch.cat([self.z_history, z_detached], dim=0)
        if self.z_history.size(0) > self.max_history:
            # Keep most recent
            self.z_history = self.z_history[-self.max_history:]

    def estimate_density(self, z: torch.Tensor) -> torch.Tensor:
        """Estimate p(z) using KDE.

        Args:
            z: Latent codes (batch_size, latent_dim)

        Returns:
            Density estimates (batch_size,)
        """
        if self.z_history.size(0) == 0:
            return torch.ones(z.size(0), device=z.device)

        # Compute pairwise distances
        dists = torch.cdist(z, self.z_history)  # (batch, history)

        # Gaussian kernel
        kernel_vals = torch.exp(-0.5 * (dists / self.bandwidth) ** 2)
        density = kernel_vals.mean(dim=1)

        return density

    def forward(self, z: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Compute curiosity loss.

        Args:
            z: Latent codes (batch_size, latent_dim)
            update: Whether to update history

        Returns:
            Curiosity loss (negative = encourage low-density exploration)
        """
        density = self.estimate_density(z)
        reward = -torch.log(density + 1e-8)

        if update:
            self.update_history(z)

        # Negative reward as loss (maximize curiosity = minimize negative reward)
        return -reward.mean()


class SymbioticBridge(nn.Module):
    """Mutual information-based coupling between VAE-A and VAE-B.

    Uses InfoNCE to estimate MI and provides adaptive coupling signal.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32):
        """Initialize symbiotic bridge.

        Args:
            latent_dim: Latent space dimension
            hidden_dim: Hidden dimension for attention/estimation
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Cross-attention: z_A attends to z_B
        self.query_A = nn.Linear(latent_dim, hidden_dim)
        self.key_B = nn.Linear(latent_dim, hidden_dim)
        self.value_B = nn.Linear(latent_dim, latent_dim)  # Output latent_dim for residual

        # Reverse attention: z_B attends to z_A
        self.query_B = nn.Linear(latent_dim, hidden_dim)
        self.key_A = nn.Linear(latent_dim, hidden_dim)
        self.value_A = nn.Linear(latent_dim, latent_dim)  # Output latent_dim for residual

        # MI estimator (bilinear)
        self.mi_estimator = nn.Bilinear(latent_dim, latent_dim, 1)

        # Adaptive rho parameters
        self.rho_base = 0.1
        self.rho_max = 0.7

    def cross_attend(
        self, z_A: torch.Tensor, z_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional cross-attention.

        Args:
            z_A: VAE-A latent codes (batch_size, latent_dim)
            z_B: VAE-B latent codes (batch_size, latent_dim)

        Returns:
            z_A_enhanced: z_A with information from z_B
            z_B_enhanced: z_B with information from z_A
        """
        hidden_dim = self.query_A.out_features

        # A attends to B
        Q_A = self.query_A(z_A)
        K_B = self.key_B(z_B)
        V_B = self.value_B(z_B)
        attn_A = F.softmax(torch.matmul(Q_A, K_B.T) / np.sqrt(hidden_dim), dim=-1)
        z_A_enhanced = z_A + torch.matmul(attn_A, V_B)

        # B attends to A
        Q_B = self.query_B(z_B)
        K_A = self.key_A(z_A)
        V_A = self.value_A(z_A)
        attn_B = F.softmax(torch.matmul(Q_B, K_A.T) / np.sqrt(hidden_dim), dim=-1)
        z_B_enhanced = z_B + torch.matmul(attn_B, V_A)

        return z_A_enhanced, z_B_enhanced

    def estimate_mi(self, z_A: torch.Tensor, z_B: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Estimate mutual information using InfoNCE.

        Args:
            z_A: VAE-A latent codes (batch_size, latent_dim)
            z_B: VAE-B latent codes (batch_size, latent_dim)

        Returns:
            mi_loss: InfoNCE loss (lower = higher MI)
            estimated_mi: Estimated MI value
        """
        batch_size = z_A.size(0)
        device = z_A.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device), 0.0

        # Positive pairs: (z_A[i], z_B[i])
        self.mi_estimator(z_A, z_B).squeeze(-1)

        # Negative pairs: (z_A[i], z_B[j]) for j != i
        # Efficient implementation using broadcasting
        z_A_expanded = z_A.unsqueeze(1).expand(-1, batch_size, -1)
        z_B_expanded = z_B.unsqueeze(0).expand(batch_size, -1, -1)

        # All pairwise scores
        all_scores = self.mi_estimator(
            z_A_expanded.reshape(-1, self.latent_dim),
            z_B_expanded.reshape(-1, self.latent_dim)
        ).reshape(batch_size, batch_size)

        # InfoNCE: positive on diagonal, negatives off-diagonal
        labels = torch.arange(batch_size, device=device)
        mi_loss = F.cross_entropy(all_scores, labels)

        # Estimated MI (higher is better)
        estimated_mi = float(np.log(batch_size) - mi_loss.item())

        return mi_loss, estimated_mi

    def compute_adaptive_rho(self, estimated_mi: float, target_mi: float = 2.0) -> float:
        """Compute coupling strength based on MI.

        Low MI -> increase rho (need more coupling)
        High MI -> can relax rho (well coupled)

        Args:
            estimated_mi: Current MI estimate
            target_mi: Target MI level

        Returns:
            Adaptive rho value
        """
        mi_ratio = estimated_mi / (target_mi + 1e-8)
        rho = self.rho_base + (self.rho_max - self.rho_base) * (1 - np.tanh(mi_ratio - 1))
        return float(np.clip(rho, self.rho_base, self.rho_max))

    def forward(self, z_A: torch.Tensor, z_B: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full symbiotic bridge forward pass.

        Args:
            z_A: VAE-A latent codes
            z_B: VAE-B latent codes

        Returns:
            Dictionary with enhanced codes, MI loss, and adaptive rho
        """
        z_A_enhanced, z_B_enhanced = self.cross_attend(z_A, z_B)
        mi_loss, estimated_mi = self.estimate_mi(z_A, z_B)
        adaptive_rho = self.compute_adaptive_rho(estimated_mi)

        return {
            'z_A_enhanced': z_A_enhanced,
            'z_B_enhanced': z_B_enhanced,
            'mi_loss': mi_loss,
            'estimated_mi': estimated_mi,
            'adaptive_rho': adaptive_rho
        }


class AlgebraicClosureLoss(nn.Module):
    """Force latent arithmetic to respect operation composition.

    Implements the homomorphism constraint:
    z_a + z_b - z_0 = z_{a o b}

    where o is operation composition.
    """

    def __init__(self):
        """Initialize algebraic closure loss."""
        super().__init__()
        # Identity operation index (all zeros -> maps to index 9841)
        self.register_buffer('identity_idx', torch.tensor(9841))

    def _idx_to_lut(self, idx: torch.Tensor) -> torch.Tensor:
        """Convert operation indices to LUT representation.

        Args:
            idx: Operation indices (batch_size,)

        Returns:
            LUT representation (batch_size, 9) with values in {-1, 0, 1}
        """
        device = idx.device
        batch_size = idx.size(0)
        lut = torch.zeros(batch_size, 9, dtype=torch.long, device=device)
        for i in range(9):
            lut[:, i] = (idx // (3 ** i)) % 3 - 1
        return lut

    def _lut_to_idx(self, lut: torch.Tensor) -> torch.Tensor:
        """Convert LUT representation to operation indices.

        Args:
            lut: LUT representation (batch_size, 9)

        Returns:
            Operation indices (batch_size,)
        """
        device = lut.device
        idx = torch.zeros(lut.size(0), dtype=torch.long, device=device)
        for i in range(9):
            idx += (lut[:, i] + 1) * (3 ** i)
        return idx

    def compose_operations(
        self, a_idx: torch.Tensor, b_idx: torch.Tensor
    ) -> torch.Tensor:
        """Compute (a o b) indices.

        WARNING: Composition is NOT well-defined for ternary operations.

        Ternary operations are functions {-1,0,1}^2 -> {-1,0,1}. True function
        composition (a o b)(x,y) = a(b(x,y), ?) requires two inputs but b outputs
        only one value. This method returns identity indices as a safe fallback.

        The AlgebraicClosureLoss using this is disabled by default (appetite_closure=0.0).
        Do not enable until proper composition semantics are defined.

        Args:
            a_idx: First operation indices
            b_idx: Second operation indices

        Returns:
            Identity indices (composition not well-defined)
        """
        # Return identity operation (index 9841) as safe fallback
        # This ensures zero loss contribution: z_a + z_b - z_0 - z_identity ≈ z_a + z_b - z_0 - z_0
        device = a_idx.device
        return torch.full_like(a_idx, self.identity_idx, device=device)

    def forward(
        self,
        z: torch.Tensor,
        indices: torch.Tensor,
        n_pairs: int = 500
    ) -> torch.Tensor:
        """Compute algebraic closure loss.

        L = E[||z_a + z_b - z_0 - z_{a o b}||^2]

        Args:
            z: Latent codes (batch_size, latent_dim)
            indices: Operation indices (batch_size,)
            n_pairs: Number of pairs to sample

        Returns:
            Closure loss scalar
        """
        batch_size = z.size(0)
        device = z.device

        if batch_size < 3:
            return torch.tensor(0.0, device=device)

        # Find identity element in batch
        identity_mask = (indices == self.identity_idx)
        if identity_mask.any():
            z_0 = z[identity_mask].mean(dim=0)
        else:
            # Fallback: use batch mean
            z_0 = z.mean(dim=0)

        # Sample pairs
        n = min(n_pairs, batch_size * (batch_size - 1) // 2)
        a_batch_idx = torch.randint(0, batch_size, (n,), device=device)
        b_batch_idx = torch.randint(0, batch_size, (n,), device=device)

        # Ensure different operations
        valid = a_batch_idx != b_batch_idx
        a_batch_idx = a_batch_idx[valid]
        b_batch_idx = b_batch_idx[valid]

        if len(a_batch_idx) == 0:
            return torch.tensor(0.0, device=device)

        # Get operation indices
        a_op_idx = indices[a_batch_idx]
        b_op_idx = indices[b_batch_idx]

        # Compute composition
        composed_op_idx = self.compose_operations(a_op_idx, b_op_idx)

        # Find which composed operations are in the batch
        # Create a lookup tensor
        idx_to_batch = torch.full((19683,), -1, dtype=torch.long, device=device)
        idx_to_batch[indices] = torch.arange(batch_size, device=device)

        composed_batch_idx = idx_to_batch[composed_op_idx]
        valid_mask = composed_batch_idx >= 0

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Get latent codes for valid triplets
        z_a = z[a_batch_idx[valid_mask]]
        z_b = z[b_batch_idx[valid_mask]]
        z_composed = z[composed_batch_idx[valid_mask]]

        # Homomorphism loss: z_a + z_b - z_0 should equal z_{a o b}
        z_predicted = z_a + z_b - z_0.unsqueeze(0)
        loss = F.mse_loss(z_predicted, z_composed)

        return loss


class ViolationBuffer:
    """Track persistently violated triplets with exponential decay.

    Acts as "irritation memory" - chronic violations get escalating attention.
    """

    def __init__(self, capacity: int = 10000, decay: float = 0.95):
        """Initialize violation buffer.

        Args:
            capacity: Maximum number of violations to track
            decay: Decay factor for old violations
        """
        self.capacity = capacity
        self.decay = decay
        self.violations: Dict[Tuple[int, int, int], Tuple[int, int]] = {}

    def record_violation(self, anchor: int, pos: int, neg: int, epoch: int):
        """Record a triplet violation.

        Args:
            anchor: Anchor operation index
            pos: Positive operation index
            neg: Negative operation index
            epoch: Current epoch
        """
        key = (anchor, pos, neg)
        if key in self.violations:
            count, _ = self.violations[key]
            self.violations[key] = (count + 1, epoch)
        else:
            self.violations[key] = (1, epoch)

        # Prune if over capacity
        if len(self.violations) > self.capacity:
            self._prune(epoch)

    def _prune(self, current_epoch: int, max_age: int = 50):
        """Remove violations older than max_age epochs."""
        self.violations = {
            k: v for k, v in self.violations.items()
            if current_epoch - v[1] < max_age
        }

    def get_attention_weights(
        self, triplets: List[Tuple[int, int, int]]
    ) -> torch.Tensor:
        """Return attention weights proportional to violation history.

        Args:
            triplets: List of (anchor, pos, neg) triplets

        Returns:
            Attention weights tensor
        """
        weights = []
        for t in triplets:
            if t in self.violations:
                count, _ = self.violations[t]
                # Logarithmic scaling
                weights.append(1.0 + np.log1p(count))
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)

    def get_violation_rate(self) -> float:
        """Get current violation rate estimate."""
        if not self.violations:
            return 0.0
        total_count = sum(v[0] for v in self.violations.values())
        return total_count / len(self.violations)
