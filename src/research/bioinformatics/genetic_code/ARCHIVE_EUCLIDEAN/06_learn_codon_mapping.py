"""
08_learn_codon_mapping.py - Learn optimal codon→index mapping

Train a small network to map 64 codons to 64 natural positions in the
embedding space such that synonymous codons map to the same cluster.

Architecture:
    Codon (one-hot 64) → MLP → Embedding (16D) → Heads:
        ├─→ Cluster classifier (21-way)
        └─→ Position selector (within-cluster)

Losses:
    1. Cluster classification (cross-entropy)
    2. Contrastive (pull synonymous together)
    3. Alignment (match VAE embedding structure)

Usage:
    python 08_learn_codon_mapping.py
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

# Natural positions from our discovery (64 indices forming 21 clusters)
NATURAL_POSITIONS = [
    732,
    737,
    738,
    762,
    974,
    987,  # Cluster 0 (size 6)
    407,
    416,
    596,
    677,
    2351,
    2354,  # Cluster 1 (size 6)
    3880,
    3882,
    5343,
    5960,
    6043,
    6066,  # Cluster 2 (size 6)
    788,
    947,
    952,
    1031,  # Cluster 3 (size 4)
    171,
    174,
    177,
    325,  # Cluster 4 (size 4)
    68,
    70,
    104,
    128,  # Cluster 5 (size 4)
    834,
    909,
    912,
    916,  # Cluster 6 (size 4)
    746,
    748,
    749,
    752,  # Cluster 7 (size 4)
    46,
    100,
    266,  # Cluster 8 (size 3)
    54,
    57,
    61,  # Cluster 9 (size 3)
    2427,
    2883,  # Cluster 10 (size 2)
    218,
    386,  # Cluster 11 (size 2)
    59,
    764,  # Cluster 12 (size 2)
    1,
    7,  # Cluster 13 (size 2)
    783,
    1035,  # Cluster 14 (size 2)
    751,
    830,  # Cluster 15 (size 2)
    831,
    897,  # Cluster 16 (size 2)
    17,
    44,  # Cluster 17 (size 2)
    773,
    878,  # Cluster 18 (size 2)
    164,  # Cluster 19 (size 1)
    467,  # Cluster 20 (size 1)
]

CLUSTER_SIZES = [6, 6, 6, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]

# Build cluster assignments for natural positions
POSITION_TO_CLUSTER = {}
CLUSTER_TO_POSITIONS = defaultdict(list)
idx = 0
for cluster_id, size in enumerate(CLUSTER_SIZES):
    for _ in range(size):
        pos = NATURAL_POSITIONS[idx]
        POSITION_TO_CLUSTER[pos] = cluster_id
        CLUSTER_TO_POSITIONS[cluster_id].append(pos)
        idx += 1


# =============================================================================
# DATA PREPARATION
# =============================================================================


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

    # Verify sizes match
    aa_sizes = [aa_counts[aa] for aa in aa_sorted]
    assert aa_sizes == CLUSTER_SIZES, f"Mismatch: {aa_sizes} vs {CLUSTER_SIZES}"

    aa_to_cluster = {aa: i for i, aa in enumerate(aa_sorted)}

    # Create codon data
    codon_to_idx = {c: i for i, c in enumerate(codons)}
    codon_clusters = [aa_to_cluster[GENETIC_CODE[c]] for c in codons]

    # Create one-hot encodings for nucleotides
    nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

    codon_features = []
    for codon in codons:
        # One-hot for each position (3 positions × 4 nucleotides = 12 features)
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
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
    }


# =============================================================================
# MODEL
# =============================================================================


class CodonEncoder(nn.Module):
    """Encode codons into embedding space matching VAE structure."""

    def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Cluster classification head
        self.cluster_head = nn.Linear(embed_dim, n_clusters)

        # Learnable cluster centers (initialized from VAE embeddings)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Codon features (batch, 12)

        Returns:
            embedding: Learned embedding (batch, 16)
            cluster_logits: Cluster predictions (batch, 21)
        """
        embedding = self.encoder(x)
        cluster_logits = self.cluster_head(embedding)
        return embedding, cluster_logits

    def init_centers_from_vae(self, vae_embeddings):
        """Initialize cluster centers from VAE embeddings of natural positions."""
        with torch.no_grad():
            for cluster_id in range(21):
                positions = CLUSTER_TO_POSITIONS[cluster_id]
                center = vae_embeddings[positions].mean(dim=0)
                self.cluster_centers[cluster_id] = center


# =============================================================================
# LOSSES
# =============================================================================


def contrastive_loss(embeddings, positive_pairs, negative_pairs, margin=1.0):
    """Contrastive loss: pull synonymous codons together, push others apart."""
    loss = 0.0
    n_pairs = 0

    # Positive pairs: minimize distance
    for i, j in positive_pairs:
        dist = F.pairwise_distance(embeddings[i : i + 1], embeddings[j : j + 1])
        loss += dist.pow(2)
        n_pairs += 1

    # Negative pairs: maximize distance (with margin)
    for i, j in negative_pairs[: len(positive_pairs) * 2]:  # Sample subset
        dist = F.pairwise_distance(embeddings[i : i + 1], embeddings[j : j + 1])
        loss += F.relu(margin - dist).pow(2)
        n_pairs += 1

    return loss / n_pairs if n_pairs > 0 else torch.tensor(0.0)


def center_alignment_loss(embeddings, clusters, cluster_centers):
    """Align embeddings with their cluster centers."""
    loss = 0.0
    for i, cluster_id in enumerate(clusters):
        center = cluster_centers[cluster_id]
        dist = F.pairwise_distance(embeddings[i : i + 1], center.unsqueeze(0))
        loss += dist.pow(2)
    return loss / len(clusters)


# =============================================================================
# TRAINING
# =============================================================================


def train_model(model, data, vae_embeddings, n_epochs=500, lr=0.01):
    """Train the codon encoder."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    features = data["features"]
    clusters = data["clusters"]
    positive_pairs = data["positive_pairs"]
    negative_pairs = data["negative_pairs"]

    # Initialize centers from VAE
    model.init_centers_from_vae(vae_embeddings)

    history = {"loss": [], "cluster_acc": [], "contrastive": []}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        embeddings, cluster_logits = model(features)

        # Losses
        loss_cluster = F.cross_entropy(cluster_logits, clusters)
        loss_contrastive = contrastive_loss(embeddings, positive_pairs, negative_pairs)
        loss_center = center_alignment_loss(embeddings, clusters, model.cluster_centers)

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
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, " f"cluster_acc={acc*100:.1f}%, contrastive={loss_contrastive.item():.4f}")

    return history


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_mapping(model, data, vae_embeddings):
    """Evaluate the learned mapping."""
    model.eval()

    features = data["features"]
    clusters = data["clusters"]
    codons = data["codons"]

    with torch.no_grad():
        embeddings, cluster_logits = model(features)
        preds = cluster_logits.argmax(dim=1)

    # Cluster accuracy
    cluster_acc = (preds == clusters).float().mean().item()

    # Check if synonymous codons have same prediction
    synonymous_correct = 0
    synonymous_total = 0
    for i, j in data["positive_pairs"]:
        if preds[i] == preds[j]:
            synonymous_correct += 1
        synonymous_total += 1

    synonymous_acc = synonymous_correct / synonymous_total if synonymous_total > 0 else 0

    # Compute embedding distances
    within_dists = []
    between_dists = []

    for i in range(64):
        for j in range(i + 1, 64):
            dist = F.pairwise_distance(embeddings[i : i + 1], embeddings[j : j + 1]).item()
            if clusters[i] == clusters[j]:
                within_dists.append(dist)
            else:
                between_dists.append(dist)

    mean_within = np.mean(within_dists) if within_dists else 0
    mean_between = np.mean(between_dists) if between_dists else 0
    separation = mean_between / mean_within if mean_within > 0 else 0

    print("\n  Evaluation Results:")
    print(f"    Cluster accuracy: {cluster_acc*100:.1f}%")
    print(f"    Synonymous pair accuracy: {synonymous_acc*100:.1f}%")
    print(f"    Mean within-cluster distance: {mean_within:.4f}")
    print(f"    Mean between-cluster distance: {mean_between:.4f}")
    print(f"    Separation ratio: {separation:.2f}x")

    return {
        "cluster_acc": cluster_acc,
        "synonymous_acc": synonymous_acc,
        "mean_within": mean_within,
        "mean_between": mean_between,
        "separation_ratio": separation,
        "embeddings": embeddings.numpy(),
        "predictions": preds.numpy(),
    }


def assign_codons_to_positions(model, data, vae_embeddings):
    """Assign each codon to a specific natural position."""
    model.eval()

    features = data["features"]
    clusters = data["clusters"]
    codons = data["codons"]

    with torch.no_grad():
        embeddings, cluster_logits = model(features)
        pred_clusters = cluster_logits.argmax(dim=1)

    # For each codon, find the nearest natural position in its predicted cluster
    codon_to_position = {}
    position_assignments = defaultdict(list)

    for i, codon in enumerate(codons):
        cluster_id = pred_clusters[i].item()
        cluster_positions = CLUSTER_TO_POSITIONS[cluster_id]

        # Find nearest position in this cluster
        codon_emb = embeddings[i]
        min_dist = float("inf")
        best_pos = cluster_positions[0]

        for pos in cluster_positions:
            pos_emb = vae_embeddings[pos]
            dist = F.pairwise_distance(codon_emb.unsqueeze(0), pos_emb.unsqueeze(0)).item()
            if dist < min_dist:
                min_dist = dist
                best_pos = pos

        codon_to_position[codon] = best_pos
        position_assignments[best_pos].append(codon)

    # Check coverage
    covered_positions = len(set(codon_to_position.values()))
    print("\n  Position Assignment:")
    print(f"    Unique positions covered: {covered_positions}/64")

    # Show sample mappings
    print("\n  Sample codon→position mappings:")
    for codon in list(codons)[:10]:
        aa = GENETIC_CODE[codon]
        pos = codon_to_position[codon]
        cluster = POSITION_TO_CLUSTER[pos]
        print(f"    {codon} ({aa}) → position {pos} (cluster {cluster})")

    return codon_to_position, position_assignments


# =============================================================================
# VISUALIZATION
# =============================================================================


def visualize_results(model, data, vae_embeddings, eval_results, output_dir):
    """Visualize the learned mapping."""
    from sklearn.decomposition import PCA

    features = data["features"]
    clusters = data["clusters"]
    codons = data["codons"]

    with torch.no_grad():
        embeddings, _ = model(features)
        embeddings = embeddings.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 1. PCA of learned embeddings
    ax1 = axes[0, 0]
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)

    unique_clusters = sorted(set(clusters.tolist()))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters.numpy() == cluster_id
        ax1.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[colors[i]],
            s=80,
            alpha=0.7,
            label=f"C{cluster_id}",
        )

    ax1.set_title("Learned Codon Embeddings (PCA)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    # 2. Compare with VAE embeddings
    ax2 = axes[0, 1]
    natural_emb = vae_embeddings[NATURAL_POSITIONS].numpy()
    natural_2d = pca.transform(natural_emb[:64])  # Use same PCA

    ax2.scatter(
        natural_2d[:, 0],
        natural_2d[:, 1],
        c="gray",
        s=50,
        alpha=0.5,
        label="VAE natural",
    )
    ax2.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c="red",
        s=30,
        alpha=0.7,
        label="Learned",
    )
    ax2.set_title("Learned vs VAE Natural Positions")
    ax2.legend()

    # 3. Training history
    ax3 = axes[1, 0]
    # (Would need history passed in)
    ax3.text(
        0.5,
        0.5,
        "Training complete\nCluster acc: {:.1f}%".format(eval_results["cluster_acc"] * 100),
        ha="center",
        va="center",
        fontsize=14,
    )
    ax3.set_title("Training Summary")
    ax3.axis("off")

    # 4. Distance distribution
    ax4 = axes[1, 1]
    within_dists = []
    between_dists = []

    for i in range(64):
        for j in range(i + 1, 64):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if clusters[i] == clusters[j]:
                within_dists.append(dist)
            else:
                between_dists.append(dist)

    ax4.hist(within_dists, bins=20, alpha=0.7, label="Within cluster", density=True)
    ax4.hist(
        between_dists,
        bins=20,
        alpha=0.7,
        label="Between clusters",
        density=True,
    )
    ax4.axvline(np.mean(within_dists), color="blue", linestyle="--")
    ax4.axvline(np.mean(between_dists), color="orange", linestyle="--")
    ax4.set_xlabel("Embedding Distance")
    ax4.set_title("Within vs Between Cluster Distances")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "learned_codon_mapping.png", dpi=150)
    plt.close()

    print(f"\n  Saved visualization to {output_dir}/learned_codon_mapping.png")


# =============================================================================
# MAIN
# =============================================================================


def main():
    output_dir = PROJECT_ROOT / "riemann_hypothesis_sandbox" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LEARNING OPTIMAL CODON → INDEX MAPPING")
    print("=" * 70)

    # Load VAE embeddings
    print("\nLoading VAE embeddings...")
    data_path = PROJECT_ROOT / "riemann_hypothesis_sandbox" / "embeddings" / "embeddings.pt"
    vae_data = torch.load(data_path, weights_only=False)

    z_B = vae_data.get("z_B_hyp", vae_data.get("z_hyperbolic"))
    if torch.is_tensor(z_B):
        vae_embeddings = z_B
    else:
        vae_embeddings = torch.tensor(z_B, dtype=torch.float32)

    print(f"  VAE embeddings shape: {vae_embeddings.shape}")

    # Prepare data
    print("\nPreparing training data...")
    data = prepare_data()

    # Create model
    print("\nCreating model...")
    model = CodonEncoder(input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    print("\nTraining...")
    history = train_model(model, data, vae_embeddings, n_epochs=500, lr=0.01)

    # Evaluate
    print("\nEvaluating...")
    eval_results = evaluate_mapping(model, data, vae_embeddings)

    # Assign codons to positions
    print("\nAssigning codons to positions...")
    codon_to_position, position_assignments = assign_codons_to_positions(model, data, vae_embeddings)

    # Visualize
    print("\nGenerating visualization...")
    visualize_results(model, data, vae_embeddings, eval_results, output_dir)

    # Save results
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "cluster_accuracy": eval_results["cluster_acc"],
        "synonymous_accuracy": eval_results["synonymous_acc"],
        "separation_ratio": eval_results["separation_ratio"],
        "codon_to_position": codon_to_position,
    }

    with open(output_dir / "learned_codon_mapping.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(
        {
            "model_state": model.state_dict(),
            "codon_to_position": codon_to_position,
        },
        output_dir / "codon_encoder.pt",
    )

    print(f"\n  Saved results to {output_dir}/learned_codon_mapping.json")
    print(f"  Saved model to {output_dir}/codon_encoder.pt")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"""
    Cluster Accuracy:     {eval_results['cluster_acc']*100:.1f}%
    Synonymous Accuracy:  {eval_results['synonymous_acc']*100:.1f}%
    Separation Ratio:     {eval_results['separation_ratio']:.2f}x

    {'*** SUCCESS: Synonymous codons map to same clusters! ***' if eval_results['synonymous_acc'] > 0.95 else 'Partial success - some synonymous codons in different clusters'}
    """
    )

    return results


if __name__ == "__main__":
    main()
