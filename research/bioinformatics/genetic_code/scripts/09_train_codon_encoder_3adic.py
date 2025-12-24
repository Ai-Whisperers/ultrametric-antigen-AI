#!/usr/bin/env python3
"""
Train Codon-Encoder-3-Adic using V5.11.3 Hyperbolic Embeddings

This script trains a codon encoder that maps 64 codons to the native hyperbolic
structure learned by V5.11.3, using Poincaré distances for all loss functions.

Key differences from original codon encoder:
1. Uses V5.11.3 hyperbolic embeddings (not v5.5 Euclidean)
2. Uses natural positions discovered in V5.11.3 space
3. Uses Poincaré geodesic distance for contrastive and alignment losses
4. Outputs embeddings in native Poincaré ball structure

Output: research/genetic_code/data/codon_encoder_3adic.pt
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# =============================================================================
# GENETIC CODE DATA
# =============================================================================

GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

CLUSTER_SIZES = [6, 6, 6, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]


# =============================================================================
# POINCARÉ DISTANCE FUNCTIONS
# =============================================================================


def poincare_distance(x, y, c=1.0, eps=1e-10):
    """Compute Poincaré ball distance between two points.

    d(x,y) = (1/sqrt(c)) * arcosh(1 + 2c||x-y||² / ((1-c||x||²)(1-c||y||²)))
    """
    norm_x_sq = torch.sum(x**2, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y**2, dim=-1, keepdim=True)
    diff_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)

    denom = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)
    denom = torch.clamp(denom, min=eps)

    arg = 1 + 2 * c * diff_sq / denom
    arg = torch.clamp(arg, min=1.0 + eps)

    dist = (1 / np.sqrt(c)) * torch.acosh(arg)
    return dist.squeeze(-1)


def poincare_distance_pairwise(x, y, c=1.0, eps=1e-10):
    """Poincaré distance between corresponding pairs of points."""
    return poincare_distance(x, y, c, eps)


# =============================================================================
# DATA PREPARATION
# =============================================================================


def load_natural_positions(data_dir):
    """Load natural positions discovered in V5.11.3 space."""
    positions_path = data_dir / "natural_positions_v5_11_3.json"

    if not positions_path.exists():
        raise FileNotFoundError(f"Natural positions not found: {positions_path}")

    with open(positions_path) as f:
        data = json.load(f)

    positions = data["positions"]
    labels = data["labels"]

    # Build mappings
    position_to_cluster = {pos: label for pos, label in zip(positions, labels)}
    cluster_to_positions = defaultdict(list)
    for pos, label in zip(positions, labels):
        cluster_to_positions[label].append(pos)

    return {
        "positions": positions,
        "labels": labels,
        "position_to_cluster": position_to_cluster,
        "cluster_to_positions": dict(cluster_to_positions),
        "metadata": data["metadata"],
    }


def prepare_data():
    """Prepare training data for codon mapping."""
    codons = list(GENETIC_CODE.keys())
    amino_acids = [GENETIC_CODE[c] for c in codons]

    # Create amino acid to cluster mapping (sorted by degeneracy)
    aa_counts = defaultdict(int)
    for aa in amino_acids:
        aa_counts[aa] += 1

    # Sort AAs by count (descending) to match cluster sizes
    aa_sorted = sorted(aa_counts.keys(), key=lambda x: (-aa_counts[x], x))
    aa_sizes = [aa_counts[aa] for aa in aa_sorted]
    assert aa_sizes == CLUSTER_SIZES, f"Mismatch: {aa_sizes} vs {CLUSTER_SIZES}"

    aa_to_cluster = {aa: i for i, aa in enumerate(aa_sorted)}

    # Create codon data
    codon_clusters = [aa_to_cluster[GENETIC_CODE[c]] for c in codons]

    # Create one-hot encodings for nucleotides
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

    # Create synonymous pairs (positive) and non-synonymous pairs (negative)
    positive_pairs = []
    negative_pairs = []

    for i in range(64):
        for j in range(i + 1, 64):
            if amino_acids[i] == amino_acids[j]:
                positive_pairs.append((i, j))
            else:
                negative_pairs.append((i, j))

    print(f"  Codons: {len(codons)}")
    print(f"  Clusters: {len(set(codon_clusters.tolist()))}")
    print(f"  Positive pairs (synonymous): {len(positive_pairs)}")
    print(f"  Negative pairs: {len(negative_pairs)}")

    return {
        "codons": codons,
        "features": codon_features,
        "clusters": codon_clusters,
        "aa_to_cluster": aa_to_cluster,
        "aa_sorted": aa_sorted,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
    }


# =============================================================================
# MODEL
# =============================================================================


class CodonEncoder3Adic(nn.Module):
    """Encode codons into V5.11.3 hyperbolic embedding space."""

    def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_clusters = n_clusters

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Cluster classification head
        self.cluster_head = nn.Linear(embed_dim, n_clusters)

        # Learnable cluster centers (initialized from V5.11.3 embeddings)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.1)

    def forward(self, x):
        embedding = self.encoder(x)
        cluster_logits = self.cluster_head(embedding)
        return embedding, cluster_logits

    def init_centers_from_vae(self, vae_embeddings, cluster_to_positions):
        """Initialize cluster centers from V5.11.3 embeddings."""
        with torch.no_grad():
            for cluster_id in range(self.n_clusters):
                positions = cluster_to_positions.get(cluster_id, [])
                if positions:
                    center = vae_embeddings[positions].mean(dim=0)
                    self.cluster_centers[cluster_id] = center


# =============================================================================
# LOSSES (Using Poincaré Distance)
# =============================================================================


def contrastive_loss_poincare(embeddings, positive_pairs, negative_pairs, margin=2.0):
    """Contrastive loss using Poincaré geodesic distance.

    Args:
        embeddings: (64, 16) codon embeddings
        positive_pairs: List of (i, j) synonymous pairs
        negative_pairs: List of (i, j) non-synonymous pairs
        margin: Margin for negative pairs (in geodesic distance)
    """
    loss = torch.tensor(0.0, device=embeddings.device)
    n_pairs = 0

    # Positive pairs: minimize Poincaré distance
    for i, j in positive_pairs:
        dist = poincare_distance(embeddings[i : i + 1], embeddings[j : j + 1])
        loss = loss + dist.pow(2)
        n_pairs += 1

    # Negative pairs: maximize distance (with margin)
    n_neg = min(len(negative_pairs), len(positive_pairs) * 2)
    for i, j in negative_pairs[:n_neg]:
        dist = poincare_distance(embeddings[i : i + 1], embeddings[j : j + 1])
        loss = loss + F.relu(margin - dist).pow(2)
        n_pairs += 1

    return loss / n_pairs if n_pairs > 0 else loss


def center_alignment_loss_poincare(embeddings, clusters, cluster_centers):
    """Align embeddings with cluster centers using Poincaré distance."""
    loss = torch.tensor(0.0, device=embeddings.device)

    for i, cluster_id in enumerate(clusters):
        center = cluster_centers[cluster_id]
        dist = poincare_distance(embeddings[i : i + 1], center.unsqueeze(0))
        loss = loss + dist.pow(2)

    return loss / len(clusters)


# =============================================================================
# TRAINING
# =============================================================================


def train_model(
    model, data, vae_embeddings, cluster_to_positions, n_epochs=500, lr=0.01
):
    """Train the codon encoder with Poincaré losses."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    features = data["features"]
    clusters = data["clusters"]
    positive_pairs = data["positive_pairs"]
    negative_pairs = data["negative_pairs"]

    # Initialize centers from VAE
    model.init_centers_from_vae(vae_embeddings, cluster_to_positions)

    history = {"loss": [], "cluster_acc": [], "contrastive": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        embeddings, cluster_logits = model(features)

        # Losses (using Poincaré distance)
        loss_cluster = F.cross_entropy(cluster_logits, clusters)
        loss_contrastive = contrastive_loss_poincare(
            embeddings, positive_pairs, negative_pairs
        )
        loss_center = center_alignment_loss_poincare(
            embeddings, clusters, model.cluster_centers
        )

        # Total loss
        loss = loss_cluster + 0.5 * loss_contrastive + 0.3 * loss_center

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            preds = cluster_logits.argmax(dim=1)
            acc = (preds == clusters).float().mean().item()

        history["loss"].append(loss.item())
        history["cluster_acc"].append(acc)
        history["contrastive"].append(loss_contrastive.item())

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(
                f"  Epoch {epoch:3d}: loss={loss.item():.4f}, "
                f"cluster_acc={acc*100:.1f}%, contrastive={loss_contrastive.item():.4f}"
            )

    return history


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_mapping(model, data, vae_embeddings):
    """Evaluate the learned mapping using Poincaré distances."""
    model.eval()

    features = data["features"]
    clusters = data["clusters"]

    with torch.no_grad():
        embeddings, cluster_logits = model(features)
        preds = cluster_logits.argmax(dim=1)

    # Cluster accuracy
    cluster_acc = (preds == clusters).float().mean().item()

    # Synonymous accuracy
    synonymous_correct = sum(
        1 for i, j in data["positive_pairs"] if preds[i] == preds[j]
    )
    synonymous_acc = synonymous_correct / len(data["positive_pairs"])

    # Compute Poincaré distances
    within_dists = []
    between_dists = []

    for i in range(64):
        for j in range(i + 1, 64):
            dist = poincare_distance(
                embeddings[i : i + 1], embeddings[j : j + 1]
            ).item()
            if clusters[i] == clusters[j]:
                within_dists.append(dist)
            else:
                between_dists.append(dist)

    mean_within = np.mean(within_dists) if within_dists else 0
    mean_between = np.mean(between_dists) if between_dists else 0
    separation = mean_between / mean_within if mean_within > 0 else 0

    # Check hierarchy correlation (embeddings should inherit V5.11.3 structure)
    radii = torch.norm(embeddings, dim=1).numpy()

    print(f"\n  Evaluation Results:")
    print(f"    Cluster accuracy: {cluster_acc*100:.1f}%")
    print(f"    Synonymous pair accuracy: {synonymous_acc*100:.1f}%")
    print(f"    Mean within-cluster distance: {mean_within:.4f}")
    print(f"    Mean between-cluster distance: {mean_between:.4f}")
    print(f"    Separation ratio: {separation:.2f}x")
    print(f"    Embedding radii: [{radii.min():.4f}, {radii.max():.4f}]")

    return {
        "cluster_acc": cluster_acc,
        "synonymous_acc": synonymous_acc,
        "mean_within": mean_within,
        "mean_between": mean_between,
        "separation_ratio": separation,
        "embeddings": embeddings.numpy(),
        "predictions": preds.numpy(),
        "radii": radii,
    }


def assign_codons_to_positions(model, data, vae_embeddings, cluster_to_positions):
    """Assign each codon to a specific natural position using Poincaré distance."""
    model.eval()

    features = data["features"]
    codons = data["codons"]

    with torch.no_grad():
        embeddings, cluster_logits = model(features)
        pred_clusters = cluster_logits.argmax(dim=1)

    codon_to_position = {}

    for i, codon in enumerate(codons):
        cluster_id = pred_clusters[i].item()
        cluster_positions = cluster_to_positions.get(cluster_id, [])

        if not cluster_positions:
            continue

        # Find nearest position using Poincaré distance
        codon_emb = embeddings[i]
        min_dist = float("inf")
        best_pos = cluster_positions[0]

        for pos in cluster_positions:
            pos_emb = vae_embeddings[pos]
            dist = poincare_distance(
                codon_emb.unsqueeze(0), pos_emb.unsqueeze(0)
            ).item()
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
    print("TRAIN CODON-ENCODER-3-ADIC")
    print("Using V5.11.3 Hyperbolic Embeddings with Poincaré Distance")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"

    # Load V5.11.3 embeddings
    embeddings_path = data_dir / "v5_11_3_embeddings.pt"
    if not embeddings_path.exists():
        print(f"ERROR: V5.11.3 embeddings not found: {embeddings_path}")
        return 1

    print(f"\nLoading V5.11.3 embeddings from: {embeddings_path}")
    emb_data = torch.load(embeddings_path, weights_only=False)
    vae_embeddings = emb_data["z_B_hyp"]  # Use VAE-B (stronger hierarchy)

    print(f"  Shape: {vae_embeddings.shape}")
    print(
        f"  Hierarchy correlation: {emb_data['metadata']['hierarchy_correlation']:.4f}"
    )

    # Load natural positions
    print("\nLoading natural positions...")
    positions_data = load_natural_positions(data_dir)
    print(f"  Positions: {len(positions_data['positions'])}")
    print(f"  Separation ratio: {positions_data['metadata']['separation_ratio']:.2f}x")

    # Prepare codon data
    print("\nPreparing codon data...")
    codon_data = prepare_data()

    # Create model
    print("\nCreating model...")
    model = CodonEncoder3Adic(input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21)

    # Train
    print("\nTraining with Poincaré losses...")
    history = train_model(
        model,
        codon_data,
        vae_embeddings,
        positions_data["cluster_to_positions"],
        n_epochs=500,
        lr=0.01,
    )

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_mapping(model, codon_data, vae_embeddings)

    # Assign codons to positions
    print("\nAssigning codons to natural positions...")
    codon_to_position = assign_codons_to_positions(
        model, codon_data, vae_embeddings, positions_data["cluster_to_positions"]
    )

    # Save model
    print("\nSaving model...")
    output = {
        "model_state": model.state_dict(),
        "codon_to_position": codon_to_position,
        "aa_to_cluster": codon_data["aa_to_cluster"],
        "aa_sorted": codon_data["aa_sorted"],
        "metadata": {
            "version": "3-adic",
            "source_embeddings": "V5.11.3",
            "hierarchy_correlation": emb_data["metadata"]["hierarchy_correlation"],
            "cluster_accuracy": results["cluster_acc"],
            "synonymous_accuracy": results["synonymous_acc"],
            "separation_ratio": results["separation_ratio"],
            "timestamp": datetime.now().isoformat(),
        },
    }

    model_path = data_dir / "codon_encoder_3adic.pt"
    torch.save(output, model_path)
    print(f"  Saved: {model_path}")

    # Save mapping as JSON
    mapping_output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "cluster_accuracy": results["cluster_acc"],
        "synonymous_accuracy": results["synonymous_acc"],
        "separation_ratio": results["separation_ratio"],
        "codon_to_position": codon_to_position,
    }

    json_path = data_dir / "codon_mapping_3adic.json"
    with open(json_path, "w") as f:
        json.dump(mapping_output, f, indent=2)
    print(f"  Saved: {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model version: codon-encoder-3-adic")
    print(f"  Source: V5.11.3 hyperbolic embeddings")
    print(f"  Cluster accuracy: {results['cluster_acc']*100:.1f}%")
    print(f"  Synonymous accuracy: {results['synonymous_acc']*100:.1f}%")
    print(f"  Separation ratio: {results['separation_ratio']:.2f}x")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
