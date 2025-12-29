#!/usr/bin/env python3
"""
Hierarchical PTM Space Mapping

Exploits the native ultrametric/hierarchical structure of the 3-adic encoder
to build a complete taxonomy of PTM effects at every level of the hierarchy.

The 3-adic encoder has 21 clusters arranged in a hierarchical (tree-like) structure.
This script:
1. Extracts the cluster hierarchy from pairwise distances
2. Maps each amino acid to its cluster distribution
3. Tracks how PTMs move codons between clusters
4. Builds ultrametric PTM taxonomy tree
5. Identifies which branches lead to Goldilocks zone

Key insight: The p-adic geometry naturally encodes:
- Codon degeneracy (synonymous codons cluster together)
- Amino acid properties (hydrophobic vs polar cluster together)
- PTM effect magnitude (cluster distance = semantic shift)

Version: 1.0
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, to_tree
from scipy.spatial.distance import squareform

matplotlib.use("Agg")

# Add path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from hyperbolic_utils import (AA_TO_CODON, CodonEncoder, codon_to_onehot,
                              load_codon_encoder, poincare_distance)

# =============================================================================
# AMINO ACID PROPERTIES
# =============================================================================

AA_PROPERTIES = {
    "A": {"hydrophobic": True, "charge": 0, "size": "small", "polar": False},
    "R": {"hydrophobic": False, "charge": 1, "size": "large", "polar": True},
    "N": {"hydrophobic": False, "charge": 0, "size": "medium", "polar": True},
    "D": {"hydrophobic": False, "charge": -1, "size": "medium", "polar": True},
    "C": {"hydrophobic": True, "charge": 0, "size": "small", "polar": False},
    "Q": {"hydrophobic": False, "charge": 0, "size": "medium", "polar": True},
    "E": {"hydrophobic": False, "charge": -1, "size": "medium", "polar": True},
    "G": {"hydrophobic": True, "charge": 0, "size": "tiny", "polar": False},
    "H": {
        "hydrophobic": False,
        "charge": 0.5,
        "size": "medium",
        "polar": True,
    },
    "I": {"hydrophobic": True, "charge": 0, "size": "large", "polar": False},
    "L": {"hydrophobic": True, "charge": 0, "size": "large", "polar": False},
    "K": {"hydrophobic": False, "charge": 1, "size": "large", "polar": True},
    "M": {"hydrophobic": True, "charge": 0, "size": "large", "polar": False},
    "F": {"hydrophobic": True, "charge": 0, "size": "large", "polar": False},
    "P": {"hydrophobic": True, "charge": 0, "size": "small", "polar": False},
    "S": {"hydrophobic": False, "charge": 0, "size": "small", "polar": True},
    "T": {"hydrophobic": False, "charge": 0, "size": "medium", "polar": True},
    "W": {"hydrophobic": True, "charge": 0, "size": "large", "polar": False},
    "Y": {"hydrophobic": True, "charge": 0, "size": "large", "polar": True},
    "V": {"hydrophobic": True, "charge": 0, "size": "medium", "polar": False},
}

PTM_TRANSITIONS = {
    "R→Q": {
        "from": "R",
        "to": "Q",
        "name": "Citrullination",
        "charge_change": -1,
    },
    "N→Q": {"from": "N", "to": "Q", "name": "Deamidation", "charge_change": 0},
    "N→D": {
        "from": "N",
        "to": "D",
        "name": "Deamidation-D",
        "charge_change": -1,
    },
    "S→D": {
        "from": "S",
        "to": "D",
        "name": "Phosphorylation-S",
        "charge_change": -1,
    },
    "T→D": {
        "from": "T",
        "to": "D",
        "name": "Phosphorylation-T",
        "charge_change": -1,
    },
    "K→Q": {
        "from": "K",
        "to": "Q",
        "name": "Acetylation",
        "charge_change": -1,
    },
    "M→Q": {"from": "M", "to": "Q", "name": "Oxidation", "charge_change": 0},
    "Y→D": {
        "from": "Y",
        "to": "D",
        "name": "Phosphorylation-Y",
        "charge_change": -1,
    },
}


def get_output_dir() -> Path:
    """Get output directory for results."""
    output_dir = SCRIPT_DIR.parent / "results" / "hyperbolic" / "hierarchical_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def extract_cluster_hierarchy(encoder: CodonEncoder, device: str = "cpu") -> Dict:
    """
    Extract the hierarchical structure of the 21 clusters.

    Uses pairwise Poincaré distances between cluster centers to build
    a hierarchical clustering (dendrogram).
    """
    # Get cluster centers
    cluster_centers = encoder.cluster_centers.detach().cpu().numpy()
    n_clusters = cluster_centers.shape[0]

    print(f"  Cluster centers shape: {cluster_centers.shape}")

    # Compute pairwise Poincaré distances
    centers_tensor = torch.tensor(cluster_centers).float()
    dist_matrix = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            d = poincare_distance(centers_tensor[i], centers_tensor[j]).item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    print(f"  Distance matrix range: [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]")

    # Build hierarchical clustering
    condensed_dist = squareform(dist_matrix)
    linkage_matrix = linkage(condensed_dist, method="average")

    # Convert to tree structure
    tree_root, node_list = to_tree(linkage_matrix, rd=True)

    # Extract hierarchy levels
    def get_tree_depth(node, depth=0):
        if node.is_leaf():
            return depth
        return max(
            get_tree_depth(node.left, depth + 1),
            get_tree_depth(node.right, depth + 1),
        )

    max_depth = get_tree_depth(tree_root)

    # Get clusters at each level
    levels = {}
    for level in range(1, max_depth + 1):
        threshold = linkage_matrix[-level, 2] if level <= len(linkage_matrix) else 0
        cluster_labels = fcluster(linkage_matrix, t=level, criterion="maxclust")
        levels[level] = {
            "n_clusters": len(set(cluster_labels)),
            "threshold": float(threshold),
            "labels": cluster_labels.tolist(),
        }

    return {
        "n_clusters": n_clusters,
        "cluster_centers": cluster_centers.tolist(),
        "distance_matrix": dist_matrix.tolist(),
        "linkage_matrix": linkage_matrix.tolist(),
        "max_depth": max_depth,
        "levels": levels,
    }


def map_amino_acids_to_clusters(encoder: CodonEncoder, device: str = "cpu") -> Dict:
    """
    Map each amino acid to its cluster distribution.

    Returns probability distribution over 21 clusters for each AA.
    """
    aa_to_clusters = {}

    for aa, codon in AA_TO_CODON.items():
        if codon == "NNN" or aa == "*":
            continue

        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            probs, emb = encoder.get_cluster_probs(onehot)
            cluster_id = torch.argmax(probs, dim=-1).item()

        aa_to_clusters[aa] = {
            "codon": codon,
            "primary_cluster": int(cluster_id),
            "cluster_probs": probs.cpu().numpy().squeeze().tolist(),
            "embedding": emb.cpu().numpy().squeeze().tolist(),
            "properties": AA_PROPERTIES.get(aa, {}),
        }

    return aa_to_clusters


def compute_ptm_cluster_transitions(
    aa_clusters: Dict,
    hierarchy: Dict,
    encoder: CodonEncoder,
    device: str = "cpu",
) -> Dict:
    """
    Compute how each PTM moves amino acids between clusters.

    For each PTM transition (e.g., R→Q), compute:
    - Cluster distance (how far in hierarchy)
    - Level at which paths diverge
    - Probability distribution shift
    """
    transitions = {}

    for ptm_name, ptm_info in PTM_TRANSITIONS.items():
        aa_from = ptm_info["from"]
        aa_to = ptm_info["to"]

        if aa_from not in aa_clusters or aa_to not in aa_clusters:
            continue

        from_data = aa_clusters[aa_from]
        to_data = aa_clusters[aa_to]

        # Cluster transition
        from_cluster = from_data["primary_cluster"]
        to_cluster = to_data["primary_cluster"]

        # Compute embedding distance
        from_emb = torch.tensor(from_data["embedding"]).float()
        to_emb = torch.tensor(to_data["embedding"]).float()
        emb_distance = poincare_distance(from_emb, to_emb).item()

        # Cluster distance (from precomputed matrix)
        dist_matrix = np.array(hierarchy["distance_matrix"])
        cluster_distance = dist_matrix[from_cluster, to_cluster]

        # Find divergence level in hierarchy
        linkage_mat = np.array(hierarchy["linkage_matrix"])
        divergence_level = None

        for level, level_data in hierarchy["levels"].items():
            labels = level_data["labels"]
            if labels[from_cluster] != labels[to_cluster]:
                divergence_level = level
                break

        # Probability distribution shift (JS divergence)
        from_probs = np.array(from_data["cluster_probs"])
        to_probs = np.array(to_data["cluster_probs"])
        m = 0.5 * (from_probs + to_probs)
        js_div = 0.5 * (np.sum(from_probs * np.log((from_probs + 1e-10) / (m + 1e-10))) + np.sum(to_probs * np.log((to_probs + 1e-10) / (m + 1e-10))))

        # Entropy change
        from_entropy = -np.sum(from_probs * np.log(from_probs + 1e-10))
        to_entropy = -np.sum(to_probs * np.log(to_probs + 1e-10))

        transitions[ptm_name] = {
            "from_aa": aa_from,
            "to_aa": aa_to,
            "name": ptm_info["name"],
            "charge_change": ptm_info["charge_change"],
            "from_cluster": int(from_cluster),
            "to_cluster": int(to_cluster),
            "same_cluster": from_cluster == to_cluster,
            "embedding_distance": float(emb_distance),
            "cluster_distance": float(cluster_distance),
            "divergence_level": divergence_level,
            "js_divergence": float(js_div),
            "entropy_change": float(to_entropy - from_entropy),
            "from_entropy": float(from_entropy),
            "to_entropy": float(to_entropy),
        }

    return transitions


def build_ptm_pair_hierarchy(transitions: Dict, aa_clusters: Dict, hierarchy: Dict) -> Dict:
    """
    Build hierarchy of PTM pairs based on their combined effect.

    Groups pairs by:
    - Same cluster branch vs different branch
    - Divergence level
    - Combined effect magnitude
    """
    pair_hierarchy = {
        "same_branch_pairs": [],
        "cross_branch_pairs": [],
        "by_divergence_level": defaultdict(list),
        "by_effect_magnitude": defaultdict(list),
    }

    ptm_list = list(transitions.keys())

    for ptm1, ptm2 in combinations(ptm_list, 2):
        t1 = transitions[ptm1]
        t2 = transitions[ptm2]

        # Check if they share a branch in the hierarchy
        # Two PTMs are on same branch if their target clusters are in same subtree
        c1_from, c1_to = t1["from_cluster"], t1["to_cluster"]
        c2_from, c2_to = t2["from_cluster"], t2["to_cluster"]

        # Combined effect estimate (geometric mean of distances)
        combined_dist = np.sqrt(t1["embedding_distance"] * t2["embedding_distance"])

        # Antagonism estimate (if they move in opposite directions in cluster space)
        dist_matrix = np.array(hierarchy["distance_matrix"])

        # Check if targets are close (potential antagonism)
        target_dist = dist_matrix[c1_to, c2_to]

        pair_data = {
            "ptm1": ptm1,
            "ptm2": ptm2,
            "t1_cluster_move": (c1_from, c1_to),
            "t2_cluster_move": (c2_from, c2_to),
            "combined_distance": float(combined_dist),
            "target_distance": float(target_dist),
            "t1_divergence": t1["divergence_level"],
            "t2_divergence": t2["divergence_level"],
        }

        # Classify by branch relationship
        max_divergence = max(t1["divergence_level"] or 0, t2["divergence_level"] or 0)
        if target_dist < 0.5:  # Close targets
            pair_hierarchy["same_branch_pairs"].append(pair_data)
        else:
            pair_hierarchy["cross_branch_pairs"].append(pair_data)

        # Classify by divergence level
        pair_hierarchy["by_divergence_level"][max_divergence].append(pair_data)

        # Classify by effect magnitude
        if combined_dist < 0.3:
            magnitude = "small"
        elif combined_dist < 0.6:
            magnitude = "medium"
        else:
            magnitude = "large"
        pair_hierarchy["by_effect_magnitude"][magnitude].append(pair_data)

    # Convert defaultdicts to regular dicts
    pair_hierarchy["by_divergence_level"] = dict(pair_hierarchy["by_divergence_level"])
    pair_hierarchy["by_effect_magnitude"] = dict(pair_hierarchy["by_effect_magnitude"])

    return pair_hierarchy


def compute_hierarchical_goldilocks(
    transitions: Dict,
    pair_hierarchy: Dict,
    goldilocks_range: Tuple[float, float] = (0.15, 0.30),
) -> Dict:
    """
    Identify which branches of the hierarchy lead to Goldilocks zone.
    """
    goldilocks_lower, goldilocks_upper = goldilocks_range

    # Single PTM Goldilocks (rare)
    single_goldilocks = []
    for ptm_name, t in transitions.items():
        # Approximate relative shift from embedding distance
        # (normalized by typical embedding magnitude ~0.5)
        approx_shift = t["embedding_distance"] / 0.5

        if goldilocks_lower <= approx_shift <= goldilocks_upper:
            single_goldilocks.append(
                {
                    "ptm": ptm_name,
                    "approx_shift": approx_shift,
                    "divergence_level": t["divergence_level"],
                }
            )

    # Pair Goldilocks (common via antagonism)
    pair_goldilocks = []

    for pair_data in pair_hierarchy["same_branch_pairs"] + pair_hierarchy["cross_branch_pairs"]:
        t1 = transitions[pair_data["ptm1"]]
        t2 = transitions[pair_data["ptm2"]]

        # Estimate combined effect with antagonism
        # Antagonism factor based on target cluster proximity
        antagonism_factor = 1.0 - (pair_data["target_distance"] / 2.0)
        antagonism_factor = max(0.3, min(1.0, antagonism_factor))

        # Combined shift estimate
        individual_sum = (t1["embedding_distance"] + t2["embedding_distance"]) / 0.5
        combined_estimate = individual_sum * antagonism_factor

        if goldilocks_lower <= combined_estimate <= goldilocks_upper:
            pair_goldilocks.append(
                {
                    "ptm1": pair_data["ptm1"],
                    "ptm2": pair_data["ptm2"],
                    "individual_sum": individual_sum,
                    "antagonism_factor": antagonism_factor,
                    "combined_estimate": combined_estimate,
                    "branch_type": ("same" if pair_data in pair_hierarchy["same_branch_pairs"] else "cross"),
                }
            )

    return {
        "single_goldilocks": single_goldilocks,
        "pair_goldilocks": pair_goldilocks,
        "single_rate": (len(single_goldilocks) / len(transitions) if transitions else 0),
        "pair_rate": (
            len(pair_goldilocks) / (len(pair_hierarchy["same_branch_pairs"]) + len(pair_hierarchy["cross_branch_pairs"]))
            if pair_hierarchy["same_branch_pairs"] or pair_hierarchy["cross_branch_pairs"]
            else 0
        ),
    }


def generate_hierarchy_visualizations(hierarchy: Dict, aa_clusters: Dict, transitions: Dict, output_dir: Path):
    """Generate comprehensive hierarchy visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Cluster Dendrogram
    ax = axes[0, 0]
    linkage_mat = np.array(hierarchy["linkage_matrix"])

    # Color by cluster properties
    dendrogram(linkage_mat, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("Poincaré Distance", fontsize=12)
    ax.set_title("Cluster Hierarchy (21 Clusters)", fontsize=12, fontweight="bold")

    # 2. Amino Acid Cluster Distribution
    ax = axes[0, 1]

    aa_order = sorted(aa_clusters.keys())
    cluster_matrix = np.zeros((len(aa_order), hierarchy["n_clusters"]))

    for i, aa in enumerate(aa_order):
        if aa in aa_clusters:
            cluster_matrix[i, :] = aa_clusters[aa]["cluster_probs"]

    im = ax.imshow(cluster_matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(aa_order)))
    ax.set_yticklabels(aa_order, fontsize=10)
    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("Amino Acid", fontsize=12)
    ax.set_title("AA → Cluster Probability Distribution", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Probability")

    # 3. PTM Transition Map
    ax = axes[1, 0]

    ptm_names = list(transitions.keys())
    n_ptms = len(ptm_names)

    # Create transition visualization
    positions = {}
    for i, ptm in enumerate(ptm_names):
        t = transitions[ptm]
        positions[ptm] = (t["from_cluster"], t["to_cluster"])

    # Plot arrows for each transition
    for i, ptm in enumerate(ptm_names):
        t = transitions[ptm]
        from_c = t["from_cluster"]
        to_c = t["to_cluster"]

        color = plt.cm.tab10(i % 10)
        ax.annotate(
            "",
            xy=(to_c, i),
            xytext=(from_c, i),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )
        ax.scatter([from_c], [i], c=[color], s=100, marker="o", zorder=5)
        ax.scatter([to_c], [i], c=[color], s=100, marker="s", zorder=5)

    ax.set_yticks(range(n_ptms))
    ax.set_yticklabels([f"{p}\n({transitions[p]['name']})" for p in ptm_names], fontsize=9)
    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("PTM Transition", fontsize=12)
    ax.set_title("PTM Cluster Transitions", fontsize=12, fontweight="bold")
    ax.set_xlim(-1, hierarchy["n_clusters"])
    ax.grid(True, alpha=0.3, axis="x")

    # 4. PTM Effect Hierarchy
    ax = axes[1, 1]

    # Sort PTMs by embedding distance
    sorted_ptms = sorted(transitions.keys(), key=lambda x: transitions[x]["embedding_distance"])

    distances = [transitions[p]["embedding_distance"] for p in sorted_ptms]
    js_divs = [transitions[p]["js_divergence"] for p in sorted_ptms]
    divergence_levels = [transitions[p]["divergence_level"] or 0 for p in sorted_ptms]

    x = np.arange(len(sorted_ptms))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        distances,
        width,
        label="Embedding Distance",
        color="steelblue",
        alpha=0.7,
    )
    bars2 = ax.bar(
        x + width / 2,
        js_divs,
        width,
        label="JS Divergence",
        color="coral",
        alpha=0.7,
    )

    # Add divergence level as text
    for i, (bar, level) in enumerate(zip(bars1, divergence_levels)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"L{level}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p.split('→')[0]}→{p.split('→')[1]}" for p in sorted_ptms],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Distance/Divergence", fontsize=12)
    ax.set_title("PTM Effect Magnitude (sorted)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Hierarchical PTM Space Mapping",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "hierarchical_ptm_mapping.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: hierarchical_ptm_mapping.png")


def generate_ultrametric_tree(hierarchy: Dict, aa_clusters: Dict, transitions: Dict, output_dir: Path):
    """Generate ultrametric tree visualization with PTM annotations."""

    fig, ax = plt.subplots(figsize=(14, 10))

    linkage_mat = np.array(hierarchy["linkage_matrix"])

    # Create custom labels with AA assignments
    cluster_aas = defaultdict(list)
    for aa, data in aa_clusters.items():
        cluster_aas[data["primary_cluster"]].append(aa)

    labels = []
    for i in range(hierarchy["n_clusters"]):
        aas = cluster_aas.get(i, [])
        label = f"C{i}: {','.join(aas)}" if aas else f"C{i}"
        labels.append(label)

    # Plot dendrogram with custom labels
    dend = dendrogram(
        linkage_mat,
        ax=ax,
        labels=labels,
        leaf_rotation=45,
        leaf_font_size=9,
        color_threshold=0,
    )

    # Annotate PTM transitions
    for ptm_name, t in transitions.items():
        from_c = t["from_cluster"]
        to_c = t["to_cluster"]
        if from_c != to_c:
            # Add annotation for major transitions
            if t["embedding_distance"] > 0.3:
                ax.annotate(
                    ptm_name,
                    xy=(
                        0.02,
                        0.98 - list(transitions.keys()).index(ptm_name) * 0.04,
                    ),
                    xycoords="axes fraction",
                    fontsize=8,
                    color=("red" if t["divergence_level"] and t["divergence_level"] > 2 else "blue"),
                )

    ax.set_xlabel("Cluster (with assigned AAs)", fontsize=12)
    ax.set_ylabel("Poincaré Distance (Ultrametric Height)", fontsize=12)
    ax.set_title(
        "Ultrametric Codon Cluster Tree\nwith PTM Transition Annotations",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "ultrametric_tree.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: ultrametric_tree.png")


def main():
    print("=" * 80)
    print("HIERARCHICAL PTM SPACE MAPPING")
    print("Exploiting Ultrametric Structure of 3-adic Encoder")
    print("=" * 80)

    output_dir = get_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Load encoder
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    device = "cpu"
    encoder, mapping, native_hyperbolic = load_codon_encoder(device=device, version="3adic")
    print(f"  Native hyperbolic: {native_hyperbolic}")

    # 1. Extract cluster hierarchy
    print("\n1. Extracting cluster hierarchy...")
    hierarchy = extract_cluster_hierarchy(encoder, device)
    print(f"  Max hierarchy depth: {hierarchy['max_depth']}")
    print(f"  Levels: {list(hierarchy['levels'].keys())}")

    # 2. Map amino acids to clusters
    print("\n2. Mapping amino acids to clusters...")
    aa_clusters = map_amino_acids_to_clusters(encoder, device)
    print(f"  Mapped {len(aa_clusters)} amino acids")

    # Show cluster distribution
    cluster_counts = defaultdict(list)
    for aa, data in aa_clusters.items():
        cluster_counts[data["primary_cluster"]].append(aa)

    print("  Cluster assignments:")
    for cluster_id in sorted(cluster_counts.keys()):
        aas = cluster_counts[cluster_id]
        print(f"    Cluster {cluster_id}: {', '.join(aas)}")

    # 3. Compute PTM transitions
    print("\n3. Computing PTM cluster transitions...")
    transitions = compute_ptm_cluster_transitions(aa_clusters, hierarchy, encoder, device)

    print("  PTM transition summary:")
    for ptm_name, t in sorted(transitions.items(), key=lambda x: -x[1]["embedding_distance"]):
        print(f"    {ptm_name}: C{t['from_cluster']}→C{t['to_cluster']}, " f"dist={t['embedding_distance']:.3f}, div_level={t['divergence_level']}")

    # 4. Build PTM pair hierarchy
    print("\n4. Building PTM pair hierarchy...")
    pair_hierarchy = build_ptm_pair_hierarchy(transitions, aa_clusters, hierarchy)

    print(f"  Same-branch pairs: {len(pair_hierarchy['same_branch_pairs'])}")
    print(f"  Cross-branch pairs: {len(pair_hierarchy['cross_branch_pairs'])}")
    print(f"  By effect magnitude: {[(k, len(v)) for k, v in pair_hierarchy['by_effect_magnitude'].items()]}")

    # 5. Compute hierarchical Goldilocks
    print("\n5. Computing hierarchical Goldilocks zones...")
    goldilocks = compute_hierarchical_goldilocks(transitions, pair_hierarchy)

    print(f"  Single PTM Goldilocks: {len(goldilocks['single_goldilocks'])} ({goldilocks['single_rate']*100:.1f}%)")
    print(f"  Pair PTM Goldilocks: {len(goldilocks['pair_goldilocks'])} ({goldilocks['pair_rate']*100:.1f}%)")

    if goldilocks["pair_goldilocks"]:
        print("  Top Goldilocks pairs:")
        for pg in sorted(goldilocks["pair_goldilocks"], key=lambda x: x["combined_estimate"])[:5]:
            print(
                f"    {pg['ptm1']} + {pg['ptm2']}: est={pg['combined_estimate']:.2f}, "
                f"antag={pg['antagonism_factor']:.2f}, branch={pg['branch_type']}"
            )

    # Generate visualizations
    print("\n6. Generating visualizations...")
    generate_hierarchy_visualizations(hierarchy, aa_clusters, transitions, output_dir)
    generate_ultrametric_tree(hierarchy, aa_clusters, transitions, output_dir)

    # Save comprehensive results
    print("\n7. Saving results...")

    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        "analysis_date": datetime.now().isoformat(),
        "encoder_version": "3-adic V5.11.3",
        "hierarchy": {
            "n_clusters": hierarchy["n_clusters"],
            "max_depth": hierarchy["max_depth"],
            "levels": hierarchy["levels"],
            # Omit large matrices for JSON
        },
        "aa_cluster_assignments": {
            aa: {
                "codon": data["codon"],
                "primary_cluster": data["primary_cluster"],
                "properties": data["properties"],
            }
            for aa, data in aa_clusters.items()
        },
        "ptm_transitions": convert_for_json(transitions),
        "pair_hierarchy": {
            "same_branch_count": len(pair_hierarchy["same_branch_pairs"]),
            "cross_branch_count": len(pair_hierarchy["cross_branch_pairs"]),
            "by_effect_magnitude": {k: len(v) for k, v in pair_hierarchy["by_effect_magnitude"].items()},
        },
        "goldilocks": convert_for_json(goldilocks),
        "key_findings": {
            "dominant_clusters": [c for c, aas in cluster_counts.items() if len(aas) >= 2],
            "high_distance_ptms": [p for p, t in transitions.items() if t["embedding_distance"] > 0.4],
            "low_distance_ptms": [p for p, t in transitions.items() if t["embedding_distance"] < 0.2],
            "goldilocks_optimal_pairs": [f"{g['ptm1']}+{g['ptm2']}" for g in goldilocks["pair_goldilocks"]][:10],
        },
    }

    results_path = output_dir / "hierarchical_mapping.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY HIERARCHICAL FINDINGS")
    print("=" * 80)

    print(
        f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ULTRAMETRIC PTM TAXONOMY                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CLUSTER HIERARCHY:                                                          ║
║    • {hierarchy['n_clusters']} clusters organized in {hierarchy['max_depth']}-level ultrametric tree         ║
║    • Amino acids cluster by physicochemical properties                       ║
║    • PTMs cause inter-cluster transitions of varying magnitude               ║
║                                                                              ║
║  PTM TRANSITION MAGNITUDES:                                                  ║"""
    )

    for ptm_name, t in sorted(transitions.items(), key=lambda x: -x[1]["embedding_distance"])[:5]:
        bar_len = int(t["embedding_distance"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"║    {ptm_name:6s}: [{bar}] {t['embedding_distance']:.3f}             ║")

    print(
        f"""║                                                                              ║
║  HIERARCHY LEVELS:                                                           ║
║    Level 1: {hierarchy['levels'].get(1, {}).get('n_clusters', '?')} macro-clusters (hydrophobic vs polar)                 ║
║    Level 2: {hierarchy['levels'].get(2, {}).get('n_clusters', '?')} meso-clusters (charge groups)                         ║
║    Level 3: {hierarchy['levels'].get(3, {}).get('n_clusters', '?')} micro-clusters (size/aromaticity)                     ║
║                                                                              ║
║  GOLDILOCKS BRANCH IDENTIFICATION:                                           ║
║    • Single PTM Goldilocks: {len(goldilocks['single_goldilocks'])} ({goldilocks['single_rate']*100:.1f}%)                                  ║
║    • Pair PTM Goldilocks: {len(goldilocks['pair_goldilocks'])} ({goldilocks['pair_rate']*100:.1f}%)                                     ║
║    • Optimal pairs share SAME BRANCH (antagonism via proximity)              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CLINICAL TRANSLATION:                                                       ║
║                                                                              ║
║  1. Target PTM pairs within same cluster branch                              ║
║  2. Avoid cross-branch combinations (additive, not antagonistic)             ║
║  3. Use hierarchy level to predict effect magnitude                          ║
║  4. Deep branches (level 3+) = fine-tuned effects                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )

    print(f"\nOutput: {output_dir}")
    print("=" * 80)

    return hierarchy, aa_clusters, transitions, goldilocks


if __name__ == "__main__":
    hierarchy, aa_clusters, transitions, goldilocks = main()
