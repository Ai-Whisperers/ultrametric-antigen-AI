#!/usr/bin/env python3
"""
Triple PTM Combinatorics: Higher-Order Synergy Analysis

Extends the dual R-N analysis to test if a third PTM multiplies potency.

Questions:
1. Does adding a third PTM (S→D, K→Q, T→D) amplify potentiation?
2. Does PTM type matter? (phosphorylation vs acetylation vs oxidation)
3. Are there "critical triplets" that massively shift geometry?

PTM Types Tested:
- R→Q: Citrullination (arginine → citrulline, approximated as glutamine)
- N→Q: Deglycosylation (asparagine → deamidated form)
- S→D: Phosphorylation (serine → phosphoserine, approximated as aspartate)
- T→D: Phosphorylation (threonine → phosphothreonine)
- K→Q: Acetylation (lysine → acetyllysine, approximated as glutamine)
- M→O: Oxidation (methionine → methionine sulfoxide, approximated as Q)

Version: 1.0
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def hyperbolic_radius(embedding: np.ndarray, c: float = 1.0) -> float:
    """V5.12.2: Proper hyperbolic distance from origin."""
    sqrt_c = np.sqrt(c)
    euclidean_norm = np.linalg.norm(embedding)
    clamped = np.clip(euclidean_norm * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


matplotlib.use("Agg")

# Add path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot, load_codon_encoder,
                              poincare_distance)

# =============================================================================
# PTM DEFINITIONS
# =============================================================================

PTM_TYPES = {
    "R": {"target": "Q", "name": "Citrullination", "category": "deimination"},
    "N": {"target": "Q", "name": "Deglycosylation", "category": "glycan_loss"},
    "S": {
        "target": "D",
        "name": "Phosphorylation-S",
        "category": "phosphorylation",
    },
    "T": {
        "target": "D",
        "name": "Phosphorylation-T",
        "category": "phosphorylation",
    },
    "K": {"target": "Q", "name": "Acetylation", "category": "acetylation"},
    "M": {"target": "Q", "name": "Oxidation", "category": "oxidation"},
}

# Alternative targets for testing PTM type effects
ALTERNATIVE_TARGETS = {
    "R": [
        "Q",
        "A",
        "E",
    ],  # Citrulline-like, Alanine (size), Glutamate (charge)
    "N": ["Q", "D", "A"],  # Deamidated, Aspartate, Alanine
    "S": ["D", "E", "A"],  # Phospho-mimetic, Alt phospho, Alanine
}


@dataclass
class PTMTriplet:
    """A triplet of modifiable sites for combinatorial analysis."""

    protein: str
    positions: Tuple[int, int, int]  # (R_pos, N_pos, X_pos)
    residues: Tuple[str, str, str]  # (R, N, X)
    x_type: str  # Type of third residue (S, T, K, M)
    max_distance: int  # Maximum pairwise distance


def get_output_dir() -> Path:
    """Get output directory for results."""
    output_dir = SCRIPT_DIR.parent / "results" / "hyperbolic" / "triple_ptm"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_acpa_proteins() -> Dict:
    """Load ACPA protein data."""
    data_path = SCRIPT_DIR.parent / "data" / "acpa_proteins.json"
    with open(data_path) as f:
        return json.load(f)


def find_triplets(
    protein: Dict,
    max_distance: int = 12,
    third_residues: Set[str] = {"S", "T", "K"},
) -> List[PTMTriplet]:
    """
    Find all R-N-X triplets within proximity threshold.

    Args:
        protein: Protein data with modifiable sites
        max_distance: Maximum distance between any pair
        third_residues: Set of valid third residue types

    Returns:
        List of PTMTriplet objects
    """
    triplets = []
    sequence = protein["sequence"]

    # Get positions for each residue type
    r_positions = [s["position"] for s in protein["modifiable_sites"].get("R", [])]
    n_positions = [s["position"] for s in protein["modifiable_sites"].get("N", [])]

    # Find third residue positions from sequence
    x_positions = defaultdict(list)
    for i, aa in enumerate(sequence):
        if aa in third_residues:
            x_positions[aa].append(i + 1)  # 1-indexed

    # Find valid triplets
    for r_pos in r_positions:
        for n_pos in n_positions:
            rn_dist = abs(r_pos - n_pos)
            if rn_dist > max_distance or rn_dist == 0:
                continue

            for x_type, x_pos_list in x_positions.items():
                for x_pos in x_pos_list:
                    rx_dist = abs(r_pos - x_pos)
                    nx_dist = abs(n_pos - x_pos)

                    # All pairwise distances must be within threshold
                    if rx_dist <= max_distance and nx_dist <= max_distance:
                        if rx_dist > 0 and nx_dist > 0:  # No overlapping positions
                            triplets.append(
                                PTMTriplet(
                                    protein=protein["name"],
                                    positions=(r_pos, n_pos, x_pos),
                                    residues=("R", "N", x_type),
                                    x_type=x_type,
                                    max_distance=max(rn_dist, rx_dist, nx_dist),
                                )
                            )

    return triplets


def compute_multi_ptm_effect(
    sequence: str,
    modifications: List[Tuple[int, str, str]],  # [(position, original, target), ...]
    encoder,
    device: str = "cpu",
) -> Optional[Dict]:
    """
    Compute geometric effect of multiple PTMs on a sequence.

    Args:
        sequence: Original protein sequence
        modifications: List of (position, original_residue, target_residue) tuples
        encoder: Codon encoder model
        device: Compute device

    Returns:
        Dict with centroid shift, entropy change, JS divergence
    """
    if not modifications:
        return None

    positions = [m[0] for m in modifications]
    min_pos = min(positions)
    max_pos = max(positions)
    start = max(0, min_pos - 8)
    end = min(len(sequence), max_pos + 8)

    # Extract context
    original_context = sequence[start:end]

    # Apply modifications
    modified_context = list(original_context)
    for pos, orig_res, target_res in modifications:
        local_pos = pos - start - 1
        if 0 <= local_pos < len(modified_context):
            if modified_context[local_pos] == orig_res:
                modified_context[local_pos] = target_res
    modified_context = "".join(modified_context)

    # Encode both versions
    def encode_sequence(seq):
        embeddings = []
        cluster_probs = []
        for aa in seq:
            codon = AA_TO_CODON.get(aa)
            if codon is None or codon == "NNN":
                continue
            onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs, emb = encoder.get_cluster_probs(onehot)
                embeddings.append(emb.cpu().numpy().squeeze())
                cluster_probs.append(probs.cpu().numpy().squeeze())
        return np.array(embeddings), np.array(cluster_probs)

    orig_emb, orig_probs = encode_sequence(original_context)
    mod_emb, mod_probs = encode_sequence(modified_context)

    if len(orig_emb) == 0 or len(mod_emb) == 0:
        return None

    # Compute centroids
    orig_centroid = np.mean(orig_emb, axis=0)
    mod_centroid = np.mean(mod_emb, axis=0)

    # Centroid shift (Poincaré distance)
    centroid_shift = poincare_distance(torch.tensor(orig_centroid).float(), torch.tensor(mod_centroid).float()).item()

    # Entropy
    orig_mean_probs = np.mean(orig_probs, axis=0)
    mod_mean_probs = np.mean(mod_probs, axis=0)

    orig_entropy = -np.sum(orig_mean_probs * np.log(orig_mean_probs + 1e-10))
    mod_entropy = -np.sum(mod_mean_probs * np.log(mod_mean_probs + 1e-10))
    entropy_change = mod_entropy - orig_entropy

    # JS divergence
    m = 0.5 * (orig_mean_probs + mod_mean_probs)
    js_div = 0.5 * (
        np.sum(orig_mean_probs * np.log((orig_mean_probs + 1e-10) / (m + 1e-10)))
        + np.sum(mod_mean_probs * np.log((mod_mean_probs + 1e-10) / (m + 1e-10)))
    )

    # Relative shift (V5.12.2: use hyperbolic radius)
    orig_norm = hyperbolic_radius(orig_centroid)
    relative_shift = centroid_shift / (orig_norm + 1e-10)

    return {
        "centroid_shift": centroid_shift,
        "relative_shift": relative_shift,
        "entropy_change": entropy_change,
        "js_divergence": js_div,
    }


def analyze_triplet(triplet: PTMTriplet, sequence: str, encoder, device: str = "cpu") -> Optional[Dict]:
    """
    Analyze all 7 combinations for a triplet: singles, pairs, triple.

    Returns:
        Dict with effects for all combinations and synergy metrics
    """
    r_pos, n_pos, x_pos = triplet.positions
    r_res, n_res, x_res = triplet.residues

    # Get target residues
    r_target = PTM_TYPES["R"]["target"]
    n_target = PTM_TYPES["N"]["target"]
    x_target = PTM_TYPES[x_res]["target"]

    # Define all 7 modifications
    mods = {
        "R": [(r_pos, "R", r_target)],
        "N": [(n_pos, "N", n_target)],
        "X": [(x_pos, x_res, x_target)],
        "RN": [(r_pos, "R", r_target), (n_pos, "N", n_target)],
        "RX": [(r_pos, "R", r_target), (x_pos, x_res, x_target)],
        "NX": [(n_pos, "N", n_target), (x_pos, x_res, x_target)],
        "RNX": [
            (r_pos, "R", r_target),
            (n_pos, "N", n_target),
            (x_pos, x_res, x_target),
        ],
    }

    # Compute effects for all combinations
    effects = {}
    for name, mod_list in mods.items():
        effect = compute_multi_ptm_effect(sequence, mod_list, encoder, device)
        if effect is None:
            return None
        effects[name] = effect

    # Compute synergy metrics
    # Pairwise synergies (same as before)
    rn_expected = effects["R"]["relative_shift"] + effects["N"]["relative_shift"]
    rn_synergy = effects["RN"]["relative_shift"] / (rn_expected + 1e-10)

    rx_expected = effects["R"]["relative_shift"] + effects["X"]["relative_shift"]
    rx_synergy = effects["RX"]["relative_shift"] / (rx_expected + 1e-10)

    nx_expected = effects["N"]["relative_shift"] + effects["X"]["relative_shift"]
    nx_synergy = effects["NX"]["relative_shift"] / (nx_expected + 1e-10)

    # Triple synergy metrics
    # Method 1: vs sum of singles
    triple_vs_singles = effects["RNX"]["relative_shift"] / (
        effects["R"]["relative_shift"] + effects["N"]["relative_shift"] + effects["X"]["relative_shift"] + 1e-10
    )

    # Method 2: vs sum of pairs (higher-order interaction)
    triple_vs_pairs = effects["RNX"]["relative_shift"] / (
        effects["RN"]["relative_shift"] + effects["RX"]["relative_shift"] + effects["NX"]["relative_shift"] + 1e-10
    )

    # Method 3: vs best pair + remaining single (incremental effect)
    best_pair_shift = max(
        effects["RN"]["relative_shift"],
        effects["RX"]["relative_shift"],
        effects["NX"]["relative_shift"],
    )
    incremental_gain = effects["RNX"]["relative_shift"] - best_pair_shift

    # Goldilocks analysis
    goldilocks_lower = 0.15
    goldilocks_upper = 0.30

    def in_goldilocks(shift):
        return goldilocks_lower <= shift <= goldilocks_upper

    singles_in_goldilocks = sum(
        [
            in_goldilocks(effects["R"]["relative_shift"]),
            in_goldilocks(effects["N"]["relative_shift"]),
            in_goldilocks(effects["X"]["relative_shift"]),
        ]
    )

    pairs_in_goldilocks = sum(
        [
            in_goldilocks(effects["RN"]["relative_shift"]),
            in_goldilocks(effects["RX"]["relative_shift"]),
            in_goldilocks(effects["NX"]["relative_shift"]),
        ]
    )

    triple_in_goldilocks = in_goldilocks(effects["RNX"]["relative_shift"])

    # Triple potentiation: enters Goldilocks only as triple
    triple_potentiation = triple_in_goldilocks and singles_in_goldilocks == 0 and pairs_in_goldilocks == 0

    # Progressive potentiation: pair needed, then triple
    progressive_potentiation = triple_in_goldilocks and pairs_in_goldilocks > 0 and singles_in_goldilocks == 0

    return {
        "triplet": {
            "protein": triplet.protein,
            "positions": triplet.positions,
            "residues": triplet.residues,
            "x_type": triplet.x_type,
            "max_distance": triplet.max_distance,
        },
        "effects": {k: v for k, v in effects.items()},
        "synergy": {
            "rn_ratio": rn_synergy,
            "rx_ratio": rx_synergy,
            "nx_ratio": nx_synergy,
            "triple_vs_singles": triple_vs_singles,
            "triple_vs_pairs": triple_vs_pairs,
            "incremental_gain": incremental_gain,
        },
        "goldilocks": {
            "singles_in_zone": singles_in_goldilocks,
            "pairs_in_zone": pairs_in_goldilocks,
            "triple_in_zone": triple_in_goldilocks,
            "triple_potentiation": triple_potentiation,
            "progressive_potentiation": progressive_potentiation,
        },
        "shifts": {
            "R": effects["R"]["relative_shift"],
            "N": effects["N"]["relative_shift"],
            "X": effects["X"]["relative_shift"],
            "RN": effects["RN"]["relative_shift"],
            "RX": effects["RX"]["relative_shift"],
            "NX": effects["NX"]["relative_shift"],
            "RNX": effects["RNX"]["relative_shift"],
        },
    }


def test_ptm_type_variation(triplet: PTMTriplet, sequence: str, encoder, device: str = "cpu") -> Optional[Dict]:
    """
    Test how changing the PTM target residue affects the outcome.

    For R position, test: R→Q (citrulline), R→A (alanine), R→E (glutamate)
    """
    r_pos, n_pos, x_pos = triplet.positions
    _, n_res, x_res = triplet.residues

    results = {}

    for r_target in ALTERNATIVE_TARGETS["R"]:
        n_target = PTM_TYPES["N"]["target"]
        x_target = PTM_TYPES[x_res]["target"]

        mods = [
            (r_pos, "R", r_target),
            (n_pos, "N", n_target),
            (x_pos, x_res, x_target),
        ]

        effect = compute_multi_ptm_effect(sequence, mods, encoder, device)
        if effect:
            results[f"R→{r_target}"] = effect

    return results if results else None


def plot_triple_analysis(results: List[Dict], output_dir: Path):
    """Generate visualizations for triple PTM analysis."""

    # Extract data
    triple_shifts = [r["shifts"]["RNX"] * 100 for r in results]
    rn_shifts = [r["shifts"]["RN"] * 100 for r in results]
    best_pair_shifts = [max(r["shifts"]["RN"], r["shifts"]["RX"], r["shifts"]["NX"]) * 100 for r in results]
    triple_vs_singles = [r["synergy"]["triple_vs_singles"] for r in results]
    triple_vs_pairs = [r["synergy"]["triple_vs_pairs"] for r in results]
    x_types = [r["triplet"]["x_type"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Triple vs Best Pair comparison
    ax = axes[0, 0]
    ax.scatter(
        best_pair_shifts,
        triple_shifts,
        alpha=0.5,
        s=30,
        c="steelblue",
        edgecolors="white",
    )

    max_val = max(max(best_pair_shifts), max(triple_shifts))
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="No change")
    ax.axhspan(15, 30, alpha=0.2, color="gold", label="Goldilocks Zone")

    ax.set_xlabel("Best Pair Shift (%)", fontsize=12)
    ax.set_ylabel("Triple Shift (%)", fontsize=12)
    ax.set_title(
        "Triple vs Best Pair: Does Third PTM Help?",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 2. Synergy ratio distributions
    ax = axes[0, 1]
    ax.hist(
        triple_vs_singles,
        bins=25,
        alpha=0.6,
        label="Triple/Singles",
        color="steelblue",
    )
    ax.hist(
        triple_vs_pairs,
        bins=25,
        alpha=0.6,
        label="Triple/Pairs",
        color="coral",
    )
    ax.axvline(1.0, color="black", linestyle="--", lw=2)
    ax.axvline(0.33, color="green", linestyle=":", lw=2, label="Expected (1/3)")

    ax.set_xlabel("Synergy Ratio", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Triple Synergy Distributions", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Effect by third PTM type
    ax = axes[1, 0]
    x_type_data = defaultdict(list)
    for r in results:
        x_type_data[r["triplet"]["x_type"]].append(r["shifts"]["RNX"] * 100)

    positions = []
    labels = []
    data = []
    for i, (x_type, shifts) in enumerate(sorted(x_type_data.items())):
        positions.append(i)
        labels.append(f"{x_type}\n({PTM_TYPES[x_type]['name']})")
        data.append(shifts)

    bp = ax.boxplot(data, positions=positions, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")

    ax.axhspan(15, 30, alpha=0.2, color="gold", label="Goldilocks")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Triple Shift (%)", fontsize=12)
    ax.set_title("Effect by Third PTM Type", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Potentiation summary
    ax = axes[1, 1]

    triple_pot = sum(1 for r in results if r["goldilocks"]["triple_potentiation"])
    progressive_pot = sum(1 for r in results if r["goldilocks"]["progressive_potentiation"])
    pair_pot = sum(1 for r in results if r["goldilocks"]["pairs_in_zone"] > 0 and not r["goldilocks"]["triple_in_zone"])
    no_pot = len(results) - triple_pot - progressive_pot - pair_pot

    categories = [
        "Triple\nPotentiation",
        "Progressive\nPotentiation",
        "Pair Only\nPotentiation",
        "No\nPotentiation",
    ]
    values = [triple_pot, progressive_pot, pair_pot, no_pot]
    colors = ["#9c27b0", "#2196f3", "#ff9800", "#757575"]

    bars = ax.bar(
        categories,
        values,
        color=colors,
        alpha=0.7,
        edgecolor="white",
        linewidth=2,
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val}\n({100*val/len(results):.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Potentiation Categories (n={len(results)})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Triple PTM Combinatorics Analysis",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "triple_ptm_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: triple_ptm_analysis.png")


def plot_ptm_type_comparison(type_results: List[Dict], output_dir: Path):
    """Compare effects of different PTM target residues."""

    # Aggregate by target type
    target_shifts = defaultdict(list)
    for r in type_results:
        for target, effect in r.items():
            target_shifts[target].append(effect["relative_shift"] * 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = list(range(len(target_shifts)))
    labels = list(target_shifts.keys())
    data = [target_shifts[l] for l in labels]

    bp = ax.boxplot(data, positions=positions, patch_artist=True)
    colors = ["#e53935", "#43a047", "#1e88e5"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhspan(15, 30, alpha=0.2, color="gold", label="Goldilocks Zone")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Triple Shift (%)", fontsize=12)
    ax.set_xlabel("R Target Residue", fontsize=12)
    ax.set_title(
        "PTM Type Effect: How Target Residue Changes Outcome",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ptm_type_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: ptm_type_comparison.png")


def main():
    print("=" * 80)
    print("TRIPLE PTM COMBINATORICS ANALYSIS")
    print("Testing Higher-Order Synergy Effects")
    print("=" * 80)

    output_dir = get_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Load encoder
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    device = "cpu"
    encoder, _, _ = load_codon_encoder(device=device, version="3adic")

    # Load protein data
    print("\nLoading ACPA protein data...")
    data = load_acpa_proteins()
    proteins = data["proteins"]
    print(f"  Loaded {len(proteins)} proteins")

    # Find all triplets
    print("\nFinding R-N-X triplets (max distance 12 residues)...")
    all_triplets = []
    for protein in proteins:
        triplets = find_triplets(protein, max_distance=12, third_residues={"S", "T", "K"})
        all_triplets.extend([(t, protein["sequence"]) for t in triplets])

    print(f"  Found {len(all_triplets)} triplets across all proteins")

    # Count by X type
    x_type_counts = defaultdict(int)
    for t, _ in all_triplets:
        x_type_counts[t.x_type] += 1
    print(f"  By third residue: {dict(x_type_counts)}")

    # Analyze triplets (limit for speed)
    max_triplets = min(500, len(all_triplets))
    print(f"\nAnalyzing {max_triplets} triplets...")

    results = []
    type_variation_results = []

    for i, (triplet, sequence) in enumerate(all_triplets[:max_triplets]):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{max_triplets} triplets...")

        analysis = analyze_triplet(triplet, sequence, encoder, device)
        if analysis:
            results.append(analysis)

        # Test PTM type variation for a subset
        if i < 100:
            type_var = test_ptm_type_variation(triplet, sequence, encoder, device)
            if type_var:
                type_variation_results.append(type_var)

    print(f"  Successfully analyzed {len(results)} triplets")

    # Summary statistics
    print("\n" + "=" * 80)
    print("TRIPLE PTM SYNERGY SUMMARY")
    print("=" * 80)

    # Synergy ratios
    triple_vs_singles = [r["synergy"]["triple_vs_singles"] for r in results]
    triple_vs_pairs = [r["synergy"]["triple_vs_pairs"] for r in results]
    incremental_gains = [r["synergy"]["incremental_gain"] for r in results]

    print("\n  Triple vs Sum of Singles:")
    print(f"    Mean ratio: {np.mean(triple_vs_singles):.3f} ± {np.std(triple_vs_singles):.3f}")
    print(f"    Range: [{min(triple_vs_singles):.3f}, {max(triple_vs_singles):.3f}]")

    print("\n  Triple vs Sum of Pairs:")
    print(f"    Mean ratio: {np.mean(triple_vs_pairs):.3f} ± {np.std(triple_vs_pairs):.3f}")

    print("\n  Incremental Gain (Triple - Best Pair):")
    print(f"    Mean: {np.mean(incremental_gains)*100:.2f}%")
    print(f"    Positive gains: {sum(1 for g in incremental_gains if g > 0)} ({100*sum(1 for g in incremental_gains if g > 0)/len(results):.1f}%)")

    # Potentiation analysis
    triple_pot = [r for r in results if r["goldilocks"]["triple_potentiation"]]
    progressive_pot = [r for r in results if r["goldilocks"]["progressive_potentiation"]]

    print("\n  Potentiation Cases:")
    print(f"    Triple-only potentiation: {len(triple_pot)} ({100*len(triple_pot)/len(results):.1f}%)")
    print(f"    Progressive potentiation: {len(progressive_pot)} ({100*len(progressive_pot)/len(results):.1f}%)")

    if triple_pot:
        print("\n  Top Triple Potentiation Cases:")
        for p in triple_pot[:5]:
            print(
                f"    {p['triplet']['protein']}: R{p['triplet']['positions'][0]}/"
                f"N{p['triplet']['positions'][1]}/{p['triplet']['x_type']}{p['triplet']['positions'][2]} "
                f"→ {p['shifts']['RNX']*100:.1f}%"
            )

    # Effect by X type
    print("\n  Effect by Third PTM Type:")
    for x_type in ["S", "T", "K"]:
        x_results = [r for r in results if r["triplet"]["x_type"] == x_type]
        if x_results:
            shifts = [r["shifts"]["RNX"] for r in x_results]
            pot_count = sum(1 for r in x_results if r["goldilocks"]["triple_in_zone"])
            print(
                f"    {x_type} ({PTM_TYPES[x_type]['name']}): " f"mean shift={np.mean(shifts)*100:.1f}%, " f"Goldilocks={pot_count}/{len(x_results)}"
            )

    # PTM type variation analysis
    if type_variation_results:
        print("\n  PTM Target Variation (R→Q vs R→A vs R→E):")
        for target in ["R→Q", "R→A", "R→E"]:
            shifts = [r[target]["relative_shift"] for r in type_variation_results if target in r]
            if shifts:
                print(f"    {target}: mean={np.mean(shifts)*100:.1f}%, std={np.std(shifts)*100:.1f}%")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_triple_analysis(results, output_dir)
    if type_variation_results:
        plot_ptm_type_comparison(type_variation_results, output_dir)

    # Save results
    print("\nSaving results...")

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(item) for item in obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Find critical triplets
    critical_triplets = sorted(
        results,
        key=lambda r: r["goldilocks"]["triple_potentiation"] * 10 + r["goldilocks"]["triple_in_zone"],
        reverse=True,
    )[:30]

    results_json = {
        "analysis_date": datetime.now().isoformat(),
        "total_triplets": len(results),
        "summary": {
            "triple_vs_singles_mean": float(np.mean(triple_vs_singles)),
            "triple_vs_pairs_mean": float(np.mean(triple_vs_pairs)),
            "incremental_gain_mean": float(np.mean(incremental_gains)),
            "triple_potentiation_count": len(triple_pot),
            "progressive_potentiation_count": len(progressive_pot),
        },
        "by_x_type": {
            x_type: {
                "count": len([r for r in results if r["triplet"]["x_type"] == x_type]),
                "mean_shift": (
                    float(np.mean([r["shifts"]["RNX"] for r in results if r["triplet"]["x_type"] == x_type]))
                    if any(r["triplet"]["x_type"] == x_type for r in results)
                    else 0
                ),
            }
            for x_type in ["S", "T", "K"]
        },
        "critical_triplets": [convert_numpy(r) for r in critical_triplets],
        "triple_potentiation_cases": [convert_numpy(r) for r in triple_pot],
    }

    results_path = output_dir / "triple_ptm_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {results_path}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print(
        f"""
TRIPLE PTM COMBINATORICS COMPLETE

Question 1: Does a third PTM multiply potency?
  → Triple/Singles ratio: {np.mean(triple_vs_singles):.3f}
  → {'YES' if np.mean(triple_vs_singles) < 0.5 else 'PARTIAL'}: Combined effect is {'MORE' if np.mean(triple_vs_singles) > 0.33 else 'LESS'} than additive

Question 2: Does PTM type matter?
"""
    )

    for x_type in ["S", "T", "K"]:
        x_shifts = [r["shifts"]["RNX"] * 100 for r in results if r["triplet"]["x_type"] == x_type]
        if x_shifts:
            print(f"  → {x_type} ({PTM_TYPES[x_type]['name']}): {np.mean(x_shifts):.1f}% shift")

    print(
        f"""
Question 3: Are there critical triplets?
  → Triple potentiation cases: {len(triple_pot)}
  → These enter Goldilocks ONLY as triple (not singles or pairs)

Interpretation:
"""
    )

    if len(triple_pot) > 0:
        print("  ✓ HIGHER-ORDER POTENTIATION DETECTED")
        print(f"    {len(triple_pot)} triplets require ALL THREE modifications")
        print("    to enter immunogenic Goldilocks Zone")
    else:
        print("  ✗ No triple-specific potentiation")
        print("    Third PTM does not add unique immunogenic potential")

    if np.mean(triple_vs_singles) < 0.4:
        print("\n  Overall trend: STRONGLY ANTAGONISTIC")
        print("    Triple modifications compensate geometrically")
    elif np.mean(triple_vs_singles) < 0.6:
        print("\n  Overall trend: MODERATELY ANTAGONISTIC")
        print("    Some geometric compensation with third PTM")
    else:
        print("\n  Overall trend: NEAR-ADDITIVE")
        print("    Third PTM adds independent contribution")

    print(f"\nOutput: {output_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
