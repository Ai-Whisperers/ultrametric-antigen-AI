#!/usr/bin/env python3
"""
Train Codon Encoder with Fused Hyperbolic Embeddings

This script trains a codon encoder using the dual-layer fused embeddings:
- Base layer: v5_5 frozen encoder (100% coverage)
- Hierarchy layer: homeostatic_rich (best hierarchy + richness)

Key improvements over 09_train_codon_encoder_3adic.py:
1. Uses fused embeddings combining coverage and hierarchy
2. Multi-loss training with Poincare geodesic distance
3. Radial targeting for 3-adic structure preservation
4. Cluster separation optimization

Output: research/genetic_code/data/codon_encoder_fused.pt
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# GENETIC CODE DATA
# =============================================================================

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

CLUSTER_SIZES = [6, 6, 6, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
N_CLUSTERS = 21


# =============================================================================
# POINCARE GEOMETRY
# =============================================================================


def poincare_distance(x, y, c=1.0, eps=1e-10):
    """Compute Poincare ball geodesic distance."""
    norm_x_sq = torch.sum(x**2, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y**2, dim=-1, keepdim=True)
    diff_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)

    denom = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)
    denom = torch.clamp(denom, min=eps)

    arg = 1 + 2 * c * diff_sq / denom
    arg = torch.clamp(arg, min=1.0 + eps)

    dist = (1 / np.sqrt(c)) * torch.acosh(arg)
    return dist.squeeze(-1)


def hyperbolic_radius(embeddings: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """V5.12.2: Compute hyperbolic distance from origin in Poincare ball."""
    sqrt_c = np.sqrt(c)
    euclidean_norms = torch.norm(embeddings, dim=-1)
    clamped = torch.clamp(euclidean_norms * sqrt_c, max=0.999)
    return 2.0 * torch.arctanh(clamped) / sqrt_c


def exp_map_0(v, c=1.0, eps=1e-10):
    """Exponential map from origin in Poincare ball."""
    norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    sqrt_c = np.sqrt(c)
    return torch.tanh(sqrt_c * norm_v / 2) * v / (sqrt_c * norm_v)


def project_to_ball(x, max_radius=0.95, eps=1e-10):
    """Project points to stay within Poincare ball."""
    norms = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.where(norms > max_radius, max_radius / norms, torch.ones_like(norms))
    return x * scale


# =============================================================================
# DATA PREPARATION
# =============================================================================


def prepare_codon_data():
    """Prepare codon training data."""
    codons = list(GENETIC_CODE.keys())
    amino_acids = [GENETIC_CODE[c] for c in codons]

    # Sort AAs by degeneracy (descending)
    aa_counts = defaultdict(int)
    for aa in amino_acids:
        aa_counts[aa] += 1

    aa_sorted = sorted(aa_counts.keys(), key=lambda x: (-aa_counts[x], x))
    aa_to_cluster = {aa: i for i, aa in enumerate(aa_sorted)}

    codon_clusters = [aa_to_cluster[GENETIC_CODE[c]] for c in codons]

    # One-hot features (3 positions x 4 nucleotides = 12)
    nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    codon_features = []
    for codon in codons:
        features = []
        for nuc in codon:
            one_hot = [0, 0, 0, 0]
            one_hot[nuc_to_idx[nuc]] = 1
            features.extend(one_hot)
        codon_features.append(features)

    codon_features = torch.tensor(codon_features, dtype=torch.float32)
    codon_clusters = torch.tensor(codon_clusters, dtype=torch.long)

    # Synonymous/non-synonymous pairs
    positive_pairs = []
    negative_pairs = []
    for i in range(64):
        for j in range(i + 1, 64):
            if amino_acids[i] == amino_acids[j]:
                positive_pairs.append((i, j))
            else:
                negative_pairs.append((i, j))

    print(f"  Codons: {len(codons)}")
    print(f"  Clusters: {N_CLUSTERS}")
    print(f"  Positive pairs: {len(positive_pairs)}")
    print(f"  Negative pairs: {len(negative_pairs)}")

    return {
        "codons": codons,
        "amino_acids": amino_acids,
        "features": codon_features,
        "clusters": codon_clusters,
        "aa_to_cluster": aa_to_cluster,
        "aa_sorted": aa_sorted,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
    }


def discover_natural_positions(embeddings, n_clusters=21, cluster_sizes=CLUSTER_SIZES):
    """Discover natural cluster positions in embedding space.

    Uses K-means-like clustering on radii and angles to find
    positions that naturally form 21 amino acid clusters.
    """
    from sklearn.cluster import KMeans

    # V5.12.2: Use hyperbolic radius for Poincare ball embeddings
    radii = hyperbolic_radius(embeddings).numpy()
    n_ops = len(embeddings)

    # Sort by radius (hierarchical ordering)
    sorted_indices = np.argsort(radii)

    # Assign to clusters based on sizes
    cluster_to_positions = defaultdict(list)
    position_to_cluster = {}

    idx = 0
    for cluster_id, size in enumerate(cluster_sizes):
        # Find positions for this cluster
        for _ in range(size):
            if idx < n_ops:
                pos = int(sorted_indices[idx])
                cluster_to_positions[cluster_id].append(pos)
                position_to_cluster[pos] = cluster_id
                idx += 1

    return dict(cluster_to_positions), position_to_cluster


# =============================================================================
# MODEL
# =============================================================================


class CodonEncoderFused(nn.Module):
    """Codon encoder for fused hyperbolic embeddings."""

    def __init__(self, input_dim=12, hidden_dim=64, embed_dim=16, n_clusters=21, max_radius=0.95):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_clusters = n_clusters
        self.max_radius = max_radius

        # Deeper encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Hyperbolic projection (Euclidean -> Poincare)
        self.hyp_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Cluster classification head
        self.cluster_head = nn.Linear(embed_dim, n_clusters)

        # Learnable cluster centers (in Poincare ball)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.3)

        # Learnable target radii per cluster (for 3-adic structure)
        self.target_radii = nn.Parameter(torch.linspace(0.2, 0.9, n_clusters))

    def forward(self, x):
        """Forward pass returning both Euclidean and hyperbolic embeddings."""
        z_euc = self.encoder(x)

        # Project to hyperbolic space
        z_hyp_raw = self.hyp_proj(z_euc)
        z_hyp = project_to_ball(exp_map_0(z_hyp_raw), self.max_radius)

        cluster_logits = self.cluster_head(z_euc)

        return {
            "z_euc": z_euc,
            "z_hyp": z_hyp,
            "cluster_logits": cluster_logits,
        }

    def init_centers_from_embeddings(self, vae_embeddings, cluster_to_positions):
        """Initialize cluster centers from VAE embeddings."""
        with torch.no_grad():
            for cluster_id in range(self.n_clusters):
                positions = cluster_to_positions.get(cluster_id, [])
                if positions:
                    center = vae_embeddings[positions].mean(dim=0)
                    self.cluster_centers[cluster_id] = project_to_ball(
                        center.unsqueeze(0), self.max_radius
                    ).squeeze(0)


# =============================================================================
# LOSSES
# =============================================================================


def contrastive_loss_poincare(embeddings, positive_pairs, negative_pairs, margin=2.0):
    """Contrastive loss using Poincare geodesic distance."""
    loss = torch.tensor(0.0, device=embeddings.device)
    n_pairs = 0

    # Positive: minimize distance
    for i, j in positive_pairs:
        dist = poincare_distance(embeddings[i:i+1], embeddings[j:j+1])
        loss = loss + dist.pow(2)
        n_pairs += 1

    # Negative: maximize distance with margin
    n_neg = min(len(negative_pairs), len(positive_pairs) * 2)
    np.random.shuffle(negative_pairs)
    for i, j in negative_pairs[:n_neg]:
        dist = poincare_distance(embeddings[i:i+1], embeddings[j:j+1])
        loss = loss + F.relu(margin - dist).pow(2)
        n_pairs += 1

    return loss / max(n_pairs, 1)


def center_alignment_loss_poincare(embeddings, clusters, cluster_centers):
    """Align embeddings with cluster centers using Poincare distance."""
    loss = torch.tensor(0.0, device=embeddings.device)

    for i, cluster_id in enumerate(clusters):
        center = cluster_centers[cluster_id]
        dist = poincare_distance(embeddings[i:i+1], center.unsqueeze(0))
        loss = loss + dist.pow(2)

    return loss / len(clusters)


def radial_hierarchy_loss(embeddings, clusters, target_radii):
    """Encourage radii to follow cluster ordering (3-adic structure)."""
    # V5.12.2: Use hyperbolic radius for Poincare ball embeddings
    radii = hyperbolic_radius(embeddings)
    loss = torch.tensor(0.0, device=embeddings.device)

    for i, cluster_id in enumerate(clusters):
        target = target_radii[cluster_id]
        diff = (radii[i] - target).pow(2)
        loss = loss + diff

    return loss / len(clusters)


def separation_loss(embeddings, clusters, n_clusters):
    """Encourage separation between clusters."""
    loss = torch.tensor(0.0, device=embeddings.device)
    n_pairs = 0

    for c1 in range(n_clusters):
        for c2 in range(c1 + 1, n_clusters):
            mask1 = clusters == c1
            mask2 = clusters == c2

            if mask1.sum() > 0 and mask2.sum() > 0:
                center1 = embeddings[mask1].mean(dim=0)
                center2 = embeddings[mask2].mean(dim=0)
                dist = poincare_distance(center1.unsqueeze(0), center2.unsqueeze(0))
                loss = loss + F.relu(1.0 - dist)
                n_pairs += 1

    return loss / max(n_pairs, 1)


# =============================================================================
# TRAINING
# =============================================================================


def train_model(model, data, vae_embeddings, cluster_to_positions, config):
    """Train the codon encoder."""
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2
    )

    features = data["features"]
    clusters = data["clusters"]
    positive_pairs = data["positive_pairs"]
    negative_pairs = list(data["negative_pairs"])

    # Initialize centers
    model.init_centers_from_embeddings(vae_embeddings, cluster_to_positions)

    history = {
        "loss": [], "cluster_acc": [], "contrastive": [],
        "hierarchy": [], "separation": []
    }

    best_acc = 0.0
    best_state = None

    for epoch in range(config["n_epochs"]):
        model.train()
        optimizer.zero_grad()

        outputs = model(features)
        z_hyp = outputs["z_hyp"]
        cluster_logits = outputs["cluster_logits"]

        # Multi-loss training
        loss_cluster = F.cross_entropy(cluster_logits, clusters)
        loss_contrastive = contrastive_loss_poincare(z_hyp, positive_pairs, negative_pairs)
        loss_center = center_alignment_loss_poincare(z_hyp, clusters, model.cluster_centers)
        loss_radial = radial_hierarchy_loss(z_hyp, clusters, model.target_radii)
        loss_sep = separation_loss(z_hyp, clusters, N_CLUSTERS)

        # Weighted combination
        loss = (
            config["w_cluster"] * loss_cluster +
            config["w_contrastive"] * loss_contrastive +
            config["w_center"] * loss_center +
            config["w_radial"] * loss_radial +
            config["w_separation"] * loss_sep
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            preds = cluster_logits.argmax(dim=1)
            acc = (preds == clusters).float().mean().item()

            # V5.12.2: Use hyperbolic radius for Poincare ball embeddings
            radii = hyperbolic_radius(z_hyp).numpy()
            hier = spearmanr(clusters.numpy(), radii)[0]

        history["loss"].append(loss.item())
        history["cluster_acc"].append(acc)
        history["contrastive"].append(loss_contrastive.item())
        history["hierarchy"].append(hier)
        history["separation"].append(loss_sep.item())

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == config["n_epochs"] - 1:
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, "
                  f"acc={acc*100:.1f}%, hier={hier:.3f}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    return history, best_acc


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_model(model, data, vae_embeddings):
    """Evaluate the trained model."""
    model.eval()

    features = data["features"]
    clusters = data["clusters"]
    codons = data["codons"]

    with torch.no_grad():
        outputs = model(features)
        z_hyp = outputs["z_hyp"]
        cluster_logits = outputs["cluster_logits"]
        preds = cluster_logits.argmax(dim=1)

    # Metrics
    cluster_acc = (preds == clusters).float().mean().item()
    synonymous_acc = sum(1 for i, j in data["positive_pairs"] if preds[i] == preds[j])
    synonymous_acc /= len(data["positive_pairs"])

    # Distances
    within_dists = []
    between_dists = []
    for i in range(64):
        for j in range(i + 1, 64):
            dist = poincare_distance(z_hyp[i:i+1], z_hyp[j:j+1]).item()
            if clusters[i] == clusters[j]:
                within_dists.append(dist)
            else:
                between_dists.append(dist)

    mean_within = np.mean(within_dists) if within_dists else 0
    mean_between = np.mean(between_dists) if between_dists else 0
    separation = mean_between / mean_within if mean_within > 0 else 0

    # Hierarchy - V5.12.2: Use hyperbolic radius
    radii = hyperbolic_radius(z_hyp).numpy()
    hierarchy = spearmanr(clusters.numpy(), radii)[0]

    print("\n  Evaluation Results:")
    print(f"    Cluster accuracy: {cluster_acc*100:.1f}%")
    print(f"    Synonymous accuracy: {synonymous_acc*100:.1f}%")
    print(f"    Separation ratio: {separation:.2f}x")
    print(f"    Hierarchy correlation: {hierarchy:.4f}")
    print(f"    Radii range: [{radii.min():.4f}, {radii.max():.4f}]")

    return {
        "cluster_acc": cluster_acc,
        "synonymous_acc": synonymous_acc,
        "separation_ratio": separation,
        "hierarchy": hierarchy,
        "mean_within": mean_within,
        "mean_between": mean_between,
        "embeddings": z_hyp.numpy(),
        "predictions": preds.numpy(),
        "radii": radii,
    }


def assign_codons_to_positions(model, data, vae_embeddings, cluster_to_positions):
    """Assign codons to VAE positions."""
    model.eval()

    with torch.no_grad():
        outputs = model(data["features"])
        z_hyp = outputs["z_hyp"]
        preds = outputs["cluster_logits"].argmax(dim=1)

    codon_to_position = {}

    for i, codon in enumerate(data["codons"]):
        cluster_id = preds[i].item()
        positions = cluster_to_positions.get(cluster_id, [])

        if not positions:
            continue

        # Find nearest position
        codon_emb = z_hyp[i]
        min_dist = float("inf")
        best_pos = positions[0]

        for pos in positions:
            pos_emb = vae_embeddings[pos]
            dist = poincare_distance(codon_emb.unsqueeze(0), pos_emb.unsqueeze(0)).item()
            if dist < min_dist:
                min_dist = dist
                best_pos = pos

        codon_to_position[codon] = best_pos

    return codon_to_position


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("TRAIN CODON ENCODER (FUSED EMBEDDINGS)")
    print("Using dual-layer: v5_5 (coverage) + homeostatic_rich (hierarchy)")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"

    # Load fused embeddings
    fused_path = data_dir / "fused_embeddings.pt"
    compat_path = data_dir / "v5_11_3_embeddings.pt"

    if fused_path.exists():
        print(f"\nLoading fused embeddings from: {fused_path}")
        emb_data = torch.load(fused_path, weights_only=False)
        vae_embeddings = emb_data.get("z_fused", emb_data.get("z_B_hyp"))
        print(f"  Using: z_fused (fusion of coverage + hierarchy)")
    elif compat_path.exists():
        print(f"\nLoading embeddings from: {compat_path}")
        emb_data = torch.load(compat_path, weights_only=False)
        vae_embeddings = emb_data["z_B_hyp"]
        print(f"  Using: z_B_hyp (hierarchy layer)")
    else:
        print(f"ERROR: No embeddings found. Run 10_extract_fused_embeddings.py first")
        return 1

    print(f"  Shape: {vae_embeddings.shape}")

    if "metadata" in emb_data:
        meta = emb_data["metadata"]
        print(f"  Hierarchy: {meta.get('hierarchy_fused', meta.get('hierarchy_correlation', 'N/A'))}")

    # Discover natural positions
    print("\nDiscovering natural cluster positions...")
    cluster_to_positions, _ = discover_natural_positions(vae_embeddings)
    print(f"  Found {len(cluster_to_positions)} clusters")

    # Prepare codon data
    print("\nPreparing codon data...")
    codon_data = prepare_codon_data()

    # Training config
    config = {
        "n_epochs": 500,
        "lr": 0.005,
        "w_cluster": 1.0,
        "w_contrastive": 0.5,
        "w_center": 0.3,
        "w_radial": 0.2,
        "w_separation": 0.1,
    }

    # Create model
    print("\nCreating model...")
    model = CodonEncoderFused(
        input_dim=12, hidden_dim=64, embed_dim=16,
        n_clusters=N_CLUSTERS, max_radius=0.95
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    print("\nTraining with multi-loss Poincare geometry...")
    history, best_acc = train_model(
        model, codon_data, vae_embeddings, cluster_to_positions, config
    )

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model(model, codon_data, vae_embeddings)

    # Assign positions
    print("\nAssigning codons to natural positions...")
    codon_to_position = assign_codons_to_positions(
        model, codon_data, vae_embeddings, cluster_to_positions
    )

    # Save
    print("\nSaving...")
    output = {
        "model_state": model.state_dict(),
        "codon_to_position": codon_to_position,
        "aa_to_cluster": codon_data["aa_to_cluster"],
        "aa_sorted": codon_data["aa_sorted"],
        "config": config,
        "metadata": {
            "version": "fused",
            "source_embeddings": "fused_embeddings.pt",
            "cluster_accuracy": results["cluster_acc"],
            "synonymous_accuracy": results["synonymous_acc"],
            "separation_ratio": results["separation_ratio"],
            "hierarchy": results["hierarchy"],
            "timestamp": datetime.now().isoformat(),
        },
    }

    model_path = data_dir / "codon_encoder_fused.pt"
    torch.save(output, model_path)
    print(f"  Model saved: {model_path}")

    # JSON mapping
    mapping = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "cluster_accuracy": results["cluster_acc"],
        "synonymous_accuracy": results["synonymous_acc"],
        "separation_ratio": results["separation_ratio"],
        "hierarchy": results["hierarchy"],
        "codon_to_position": codon_to_position,
    }

    json_path = data_dir / "codon_mapping_fused.json"
    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Mapping saved: {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: codon_encoder_fused")
    print(f"  Cluster accuracy: {results['cluster_acc']*100:.1f}%")
    print(f"  Synonymous accuracy: {results['synonymous_acc']*100:.1f}%")
    print(f"  Separation ratio: {results['separation_ratio']:.2f}x")
    print(f"  Hierarchy correlation: {results['hierarchy']:.4f}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
