"""Train Hyperbolic Structure on Frozen v5.5 Manifold.

Strategy:
- v5.5 encoder is FROZEN (100% coverage already achieved)
- Train ONLY hyperbolic structure using true Poincaré geometry
- No reconstruction loss - just hyperbolic ranking + radial hierarchy
- Focus on what v5.5 didn't achieve: proper 3-adic → hyperbolic mapping

Usage:
    python scripts/train/train_hyperbolic_structure.py
"""

import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import TernaryOperationDataset
from src.data.generation import generate_all_ternary_operations
from src.losses.padic_losses import PAdicRankingLossHyperbolic
from src.losses.hyperbolic_prior import HyperbolicPrior
from src.metrics.hyperbolic import poincare_distance, compute_3adic_valuation


def compute_hyperbolic_correlation(model, device, n_samples=5000, curvature=1.0):
    """Compute 3-adic vs hyperbolic distance correlation for our model."""
    model.eval()

    with torch.no_grad():
        # Generate random operation indices
        indices = torch.randint(0, 19683, (n_samples,), device=device)

        # Convert to ternary representation
        ternary_data = torch.zeros(n_samples, 9, device=device)
        for i in range(9):
            ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

        # Forward pass
        outputs = model(ternary_data.float())
        z_hyp = outputs['z_hyp']

        # Mean radius
        mean_radius = torch.norm(z_hyp, dim=1).mean().item()

        # Sample triplets for correlation
        n_triplets = 1000
        i_idx = torch.randint(n_samples, (n_triplets,), device=device)
        j_idx = torch.randint(n_samples, (n_triplets,), device=device)
        k_idx = torch.randint(n_samples, (n_triplets,), device=device)

        valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        i_idx, j_idx, k_idx = i_idx[valid], j_idx[valid], k_idx[valid]

        if len(i_idx) < 100:
            return 0.5, mean_radius

        # 3-adic valuations
        diff_ij = torch.abs(indices[i_idx] - indices[j_idx])
        diff_ik = torch.abs(indices[i_idx] - indices[k_idx])
        v_ij = compute_3adic_valuation(diff_ij)
        v_ik = compute_3adic_valuation(diff_ik)

        # 3-adic ordering: larger valuation = closer
        padic_closer_ij = (v_ij > v_ik).float()

        # Poincaré distances
        d_ij = poincare_distance(z_hyp[i_idx], z_hyp[j_idx], curvature)
        d_ik = poincare_distance(z_hyp[i_idx], z_hyp[k_idx], curvature)

        # Hyperbolic ordering: smaller distance = closer
        hyp_closer_ij = (d_ij < d_ik).float()

        # Concordance
        corr = (padic_closer_ij == hyp_closer_ij).float().mean().item()

    return corr, mean_radius


class FrozenV55Encoder(nn.Module):
    """Frozen v5.5 encoder - provides coverage, no training."""

    def __init__(self, checkpoint_path: Path, device: str = 'cuda'):
        super().__init__()

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint['model']

        # Build encoder architecture (matches v5.5)
        self.encoder = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, 16)
        self.fc_logvar = nn.Linear(64, 16)

        # Load weights from v5.5 encoder_A
        enc_state = {}
        for k, v in model_state.items():
            if k.startswith('encoder_A.'):
                new_key = k.replace('encoder_A.', '')
                enc_state[new_key] = v

        self.encoder.load_state_dict({
            k.replace('encoder.', ''): v
            for k, v in enc_state.items() if k.startswith('encoder.')
        })
        self.fc_mu.load_state_dict({
            k.replace('fc_mu.', ''): v
            for k, v in enc_state.items() if k.startswith('fc_mu.')
        })
        self.fc_logvar.load_state_dict({
            k.replace('fc_logvar.', ''): v
            for k, v in enc_state.items() if k.startswith('fc_logvar.')
        })

        # FREEZE - no gradients
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class HyperbolicProjection(nn.Module):
    """Learnable projection from Euclidean latent to Poincaré ball.

    This is the TRAINABLE component that learns proper hyperbolic structure.
    Key: Learn BOTH direction AND radius independently.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64, curvature: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.curvature = curvature
        self.max_radius = 0.95

        # Network to predict DIRECTION (unit vector)
        self.direction_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Network to predict RADIUS (scalar per point)
        self.radius_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1], will scale to [0, max_radius]
        )

    def forward(self, z_euclidean: torch.Tensor) -> torch.Tensor:
        """Project Euclidean latent to Poincaré ball with learned direction and radius."""
        # Predict direction (normalized)
        direction = z_euclidean + self.direction_net(z_euclidean)
        direction = F.normalize(direction, dim=-1)

        # Predict radius [0, max_radius]
        radius = self.radius_net(z_euclidean) * self.max_radius

        # Combine: z_hyp = radius * direction
        z_hyp = radius * direction

        return z_hyp


class RadialStratificationLoss(nn.Module):
    """Enforce proper radial hierarchy: high valuation → near origin.

    This is THE key loss that v5.5 was missing.
    """

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.9,
        max_valuation: int = 9,
        curvature: float = 1.0
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.max_valuation = max_valuation
        self.curvature = curvature

    def forward(self, z_hyp: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Compute radial stratification loss in hyperbolic space."""
        # Compute 3-adic valuation for each index
        valuations = self._compute_valuation(batch_indices)

        # Normalized valuation [0, 1]
        norm_v = valuations / self.max_valuation

        # Target radius: high valuation → small radius
        # v=0 → outer_radius, v=max → inner_radius
        target_radius = self.outer_radius - norm_v * (self.outer_radius - self.inner_radius)

        # Actual radius in Poincaré ball
        actual_radius = torch.norm(z_hyp, dim=1)

        # Weight by valuation (high-v points are rarer, weight them more)
        weights = 1.0 + norm_v * 2.0  # High valuation gets 3x weight

        # Smooth L1 loss
        loss = F.smooth_l1_loss(actual_radius, target_radius, reduction='none')
        return (loss * weights).mean()

    def _compute_valuation(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute 3-adic valuation v_3(n)."""
        from src.core import TERNARY
        return TERNARY.valuation(indices).float()


class HyperbolicStructureModel(nn.Module):
    """Full model: Frozen v5.5 encoder + Trainable hyperbolic projection."""

    def __init__(self, frozen_encoder: FrozenV55Encoder, curvature: float = 1.0):
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.projection = HyperbolicProjection(latent_dim=16, curvature=curvature)
        self.curvature = curvature

    def forward(self, x: torch.Tensor) -> dict:
        # Get frozen embeddings (no gradients)
        with torch.no_grad():
            mu, logvar = self.frozen_encoder(x)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_euclidean = mu + eps * std

        # Project to hyperbolic (THIS IS TRAINABLE)
        z_hyp = self.projection(z_euclidean)

        return {
            'z_euclidean': z_euclidean,
            'z_hyp': z_hyp,
            'mu': mu,
            'logvar': logvar
        }


def train_hyperbolic_structure(
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    curvature: float = 1.0,
    device: str = 'cuda'
):
    """Train hyperbolic structure on frozen v5.5 manifold."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hyperbolic_structure_{timestamp}"

    print("=" * 70)
    print("HYPERBOLIC STRUCTURE TRAINING ON FROZEN V5.5 MANIFOLD")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Curvature: {curvature}")
    print(f"Epochs: {epochs}")

    # Load frozen v5.5 encoder
    checkpoint_path = PROJECT_ROOT / 'sandbox-training' / 'checkpoints' / 'v5_5' / 'latest.pt'
    print(f"\nLoading frozen encoder from: {checkpoint_path}")

    frozen_encoder = FrozenV55Encoder(checkpoint_path, device).to(device)
    model = HyperbolicStructureModel(frozen_encoder, curvature).to(device)

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {frozen_params:,}")

    # Data
    operations = generate_all_ternary_operations()
    dataset = TernaryOperationDataset(operations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Losses - ONLY hyperbolic, NO reconstruction
    ranking_loss_fn = PAdicRankingLossHyperbolic(
        base_margin=0.05,
        margin_scale=0.15,
        n_triplets=500,
        hard_negative_ratio=0.5,
        curvature=curvature,
        radial_weight=0.3,  # Emphasize radial hierarchy
        max_norm=0.95
    )
    radial_loss_fn = RadialStratificationLoss(
        inner_radius=0.1,
        outer_radius=0.9,
        curvature=curvature
    )
    hyperbolic_prior = HyperbolicPrior(
        latent_dim=16,
        curvature=curvature,
        prior_sigma=1.0,
        max_norm=0.95
    ).to(device)

    # Optimizer - only for trainable projection
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # TensorBoard
    writer = SummaryWriter(log_dir=PROJECT_ROOT / 'runs' / run_name)

    best_corr = 0.0
    best_radial_corr = 1.0  # Want negative, so start positive

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ranking = 0.0
        epoch_radial = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for batch_idx, batch_data in enumerate(loader):
            x = batch_data.to(device)
            batch_indices = torch.arange(batch_idx * batch_size,
                                        min((batch_idx + 1) * batch_size, len(dataset)),
                                        device=device)

            # Ensure batch_indices matches batch size
            if len(batch_indices) != x.size(0):
                batch_indices = torch.randint(0, 19683, (x.size(0),), device=device)

            optimizer.zero_grad()

            # Forward
            outputs = model(x)
            z_hyp = outputs['z_hyp']

            # Losses - ALL HYPERBOLIC
            ranking_loss, ranking_metrics = ranking_loss_fn(z_hyp, batch_indices)
            radial_loss = radial_loss_fn(z_hyp, batch_indices)
            kl_loss = hyperbolic_prior.kl_divergence(outputs['mu'], outputs['logvar'])

            # Total loss - EMPHASIZE RADIAL (the key missing component in v5.5)
            total_loss = 0.3 * ranking_loss + 2.0 * radial_loss + 0.1 * kl_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.projection.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ranking += ranking_loss.item()
            epoch_radial += radial_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        scheduler.step()

        # Epoch averages
        avg_loss = epoch_loss / n_batches
        avg_ranking = epoch_ranking / n_batches
        avg_radial = epoch_radial / n_batches
        avg_kl = epoch_kl / n_batches

        # Evaluation
        model.eval()
        with torch.no_grad():
            # Compute hyperbolic correlation
            corr_A_hyp, mean_radius_A = compute_hyperbolic_correlation(
                model, device, n_samples=5000, curvature=curvature
            )

            # Compute radial hierarchy correlation
            all_ops = torch.tensor(operations, dtype=torch.float32, device=device)
            all_outputs = model(all_ops)
            all_z_hyp = all_outputs['z_hyp']
            all_radii = torch.norm(all_z_hyp, dim=1).cpu().numpy()

            from src.core import TERNARY
            all_valuations = TERNARY.valuation(torch.arange(19683)).float().numpy()

            from scipy.stats import spearmanr
            radial_corr, _ = spearmanr(all_valuations, all_radii)

        # Logging
        writer.add_scalar('Loss/total', avg_loss, epoch)
        writer.add_scalar('Loss/ranking', avg_ranking, epoch)
        writer.add_scalar('Loss/radial', avg_radial, epoch)
        writer.add_scalar('Loss/kl', avg_kl, epoch)
        writer.add_scalar('Metrics/corr_hyperbolic', corr_A_hyp, epoch)
        writer.add_scalar('Metrics/radial_hierarchy_corr', radial_corr, epoch)
        writer.add_scalar('Metrics/mean_radius', mean_radius_A, epoch)

        # Track best
        if corr_A_hyp > best_corr:
            best_corr = corr_A_hyp
        if radial_corr < best_radial_corr:  # Want negative
            best_radial_corr = radial_corr

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
              f"Rank: {avg_ranking:.4f} | Radial: {avg_radial:.4f} | "
              f"Corr(hyp): {corr_A_hyp:.4f} | RadialCorr: {radial_corr:+.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = PROJECT_ROOT / 'sandbox-training' / 'checkpoints' / 'hyperbolic_structure'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state': model.projection.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'corr_hyperbolic': corr_A_hyp,
                'radial_corr': radial_corr,
                'curvature': curvature
            }, checkpoint_dir / f'epoch_{epoch}.pt')

    writer.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best hyperbolic correlation: {best_corr:.4f}")
    print(f"Best radial hierarchy correlation: {best_radial_corr:+.4f} (want negative)")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_hyperbolic_structure(
        epochs=100,
        batch_size=512,
        lr=1e-3,
        curvature=1.0,
        device=device
    )
