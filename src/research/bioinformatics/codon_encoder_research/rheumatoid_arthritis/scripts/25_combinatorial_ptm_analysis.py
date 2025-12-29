#!/usr/bin/env python3
"""
Combinatorial PTM Analysis: Citrullination + Deglycosylation

Explores whether glycan loss potentiates citrullination immunogenicity.

Hypothesis:
- Glycans may shield nearby citrullination sites from immune recognition
- Removing glycans (N→Q) near citrullination sites (R→Q) could expose
  the modified epitope, pushing it into the Goldilocks Zone

Analysis:
1. Find R-N proximity pairs (within 5-15 residue window)
2. Compute geometric effects of:
   - R→Q alone (citrullination)
   - N→Q alone (deglycosylation)
   - R→Q + N→Q together (combinatorial)
3. Test for synergy: combined effect > sum of individual effects
4. Identify "sentinel glycan" sites that shield citrullination

Version: 1.0
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


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


@dataclass
class PTMSite:
    """A modifiable site in a protein."""

    position: int  # 1-indexed
    residue: str
    context: str  # 11-mer context window


@dataclass
class RNPair:
    """A proximal R-N pair for combinatorial analysis."""

    protein: str
    r_position: int
    n_position: int
    distance: int  # residue distance
    r_context: str
    n_context: str
    combined_context: str


def get_output_dir() -> Path:
    """Get output directory for results."""
    output_dir = SCRIPT_DIR.parent / "results" / "hyperbolic" / "combinatorial_ptm"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_acpa_proteins() -> Dict:
    """Load ACPA protein data."""
    data_path = SCRIPT_DIR.parent / "data" / "acpa_proteins.json"
    with open(data_path) as f:
        return json.load(f)


def find_rn_pairs(protein: Dict, max_distance: int = 15) -> List[RNPair]:
    """
    Find all R-N pairs within proximity threshold.

    Args:
        protein: Protein data with modifiable sites
        max_distance: Maximum residue distance to consider proximal

    Returns:
        List of RNPair objects
    """
    pairs = []
    sequence = protein["sequence"]

    # Get R and N positions
    r_sites = protein["modifiable_sites"].get("R", [])
    n_sites = protein["modifiable_sites"].get("N", [])

    for r_site in r_sites:
        r_pos = r_site["position"]
        r_context = r_site.get("context_11mer", "")

        for n_site in n_sites:
            n_pos = n_site["position"]
            n_context = n_site.get("context_11mer", "")

            distance = abs(r_pos - n_pos)

            if 0 < distance <= max_distance:
                # Get combined context (region spanning both sites)
                start = max(0, min(r_pos, n_pos) - 6)
                end = min(len(sequence), max(r_pos, n_pos) + 6)
                combined_context = sequence[start:end]

                pairs.append(
                    RNPair(
                        protein=protein["name"],
                        r_position=r_pos,
                        n_position=n_pos,
                        distance=distance,
                        r_context=r_context,
                        n_context=n_context,
                        combined_context=combined_context,
                    )
                )

    return pairs


def compute_ptm_effect(
    sequence: str,
    positions: List[int],
    original_residues: List[str],
    target_residue: str,
    encoder,
    device: str = "cpu",
) -> Dict:
    """
    Compute geometric effect of PTM(s) on a sequence.

    Args:
        sequence: Original protein sequence
        positions: List of positions to modify (1-indexed)
        original_residues: Original residues at those positions
        target_residue: Target residue after modification
        encoder: Codon encoder model
        device: Compute device

    Returns:
        Dict with centroid shift, entropy change, JS divergence
    """
    # Define context window around modifications
    min_pos = min(positions)
    max_pos = max(positions)
    start = max(0, min_pos - 8)
    end = min(len(sequence), max_pos + 8)

    # Extract context
    original_context = sequence[start:end]

    # Apply modifications
    modified_context = list(original_context)
    for pos, orig_res in zip(positions, original_residues):
        local_pos = pos - start - 1  # Convert to 0-indexed local position
        if 0 <= local_pos < len(modified_context):
            if modified_context[local_pos] == orig_res:
                modified_context[local_pos] = target_residue
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
        "original_entropy": orig_entropy,
        "modified_entropy": mod_entropy,
    }


def analyze_rn_pair(pair: RNPair, sequence: str, encoder, device: str = "cpu") -> Optional[Dict]:
    """
    Analyze combinatorial effects for an R-N pair.

    Returns metrics for:
    - R→Q alone
    - N→Q alone
    - R→Q + N→Q together
    - Synergy score
    """
    # R→Q alone (citrullination)
    r_effect = compute_ptm_effect(
        sequence,
        [pair.r_position],
        ["R"],
        "Q",  # Citrulline approximated as Q
        encoder,
        device,
    )

    # N→Q alone (deglycosylation)
    n_effect = compute_ptm_effect(
        sequence,
        [pair.n_position],
        ["N"],
        "Q",  # Deglycosylated N approximated as Q
        encoder,
        device,
    )

    # Combined R→Q + N→Q
    combined_effect = compute_ptm_effect(
        sequence,
        [pair.r_position, pair.n_position],
        ["R", "N"],
        "Q",
        encoder,
        device,
    )

    if r_effect is None or n_effect is None or combined_effect is None:
        return None

    # Compute synergy scores
    # Synergy = Combined - (R_alone + N_alone)
    # Positive synergy = combinatorial effect is larger than sum of parts

    shift_synergy = combined_effect["centroid_shift"] - (r_effect["centroid_shift"] + n_effect["centroid_shift"])
    entropy_synergy = combined_effect["entropy_change"] - (r_effect["entropy_change"] + n_effect["entropy_change"])
    js_synergy = combined_effect["js_divergence"] - (r_effect["js_divergence"] + n_effect["js_divergence"])

    # Multiplicative synergy ratio
    expected_shift = r_effect["centroid_shift"] + n_effect["centroid_shift"]
    synergy_ratio = combined_effect["centroid_shift"] / (expected_shift + 1e-10)

    # Check if combined pushes into Goldilocks Zone (15-30% relative shift)
    goldilocks_lower = 0.15
    goldilocks_upper = 0.30

    r_in_goldilocks = goldilocks_lower <= r_effect["relative_shift"] <= goldilocks_upper
    n_in_goldilocks = goldilocks_lower <= n_effect["relative_shift"] <= goldilocks_upper
    combined_in_goldilocks = goldilocks_lower <= combined_effect["relative_shift"] <= goldilocks_upper

    # Key insight: Does combination push into Goldilocks when individual didn't?
    potentiation = combined_in_goldilocks and not r_in_goldilocks and not n_in_goldilocks

    return {
        "pair": {
            "protein": pair.protein,
            "r_position": pair.r_position,
            "n_position": pair.n_position,
            "distance": pair.distance,
            "r_context": pair.r_context,
            "n_context": pair.n_context,
        },
        "r_alone": r_effect,
        "n_alone": n_effect,
        "combined": combined_effect,
        "synergy": {
            "shift_synergy": shift_synergy,
            "entropy_synergy": entropy_synergy,
            "js_synergy": js_synergy,
            "synergy_ratio": synergy_ratio,
            "is_synergistic": synergy_ratio > 1.1,  # >10% more than expected
            "is_antagonistic": synergy_ratio < 0.9,  # <10% less than expected
        },
        "goldilocks": {
            "r_in_zone": r_in_goldilocks,
            "n_in_zone": n_in_goldilocks,
            "combined_in_zone": combined_in_goldilocks,
            "potentiation": potentiation,
        },
    }


def plot_synergy_analysis(results: List[Dict], output_dir: Path):
    """Generate visualizations of synergy analysis."""

    # Extract data
    r_shifts = [r["r_alone"]["relative_shift"] * 100 for r in results]
    n_shifts = [r["n_alone"]["relative_shift"] * 100 for r in results]
    combined_shifts = [r["combined"]["relative_shift"] * 100 for r in results]
    expected_shifts = [r + n for r, n in zip(r_shifts, n_shifts)]
    synergy_ratios = [r["synergy"]["synergy_ratio"] for r in results]
    distances = [r["pair"]["distance"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Combined vs Expected (synergy detection)
    ax = axes[0, 0]
    ax.scatter(
        expected_shifts,
        combined_shifts,
        c=distances,
        cmap="viridis",
        alpha=0.7,
        s=60,
        edgecolors="white",
    )

    # Diagonal line (no synergy)
    max_val = max(max(expected_shifts), max(combined_shifts))
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="No synergy")

    # Goldilocks zone
    ax.axhspan(15, 30, alpha=0.2, color="gold", label="Goldilocks Zone")

    ax.set_xlabel("Expected Shift (R + N) [%]", fontsize=12)
    ax.set_ylabel("Combined Shift [%]", fontsize=12)
    ax.set_title(
        "Synergy Analysis: Combined vs Expected",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("R-N Distance (residues)")
    ax.grid(True, alpha=0.3)

    # 2. Synergy ratio distribution
    ax = axes[0, 1]
    ax.hist(
        synergy_ratios,
        bins=20,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
    )
    ax.axvline(1.0, color="red", linestyle="--", lw=2, label="No synergy (ratio=1)")
    ax.axvline(1.1, color="green", linestyle="--", lw=1, label="Synergistic threshold")
    ax.axvline(
        0.9,
        color="orange",
        linestyle="--",
        lw=1,
        label="Antagonistic threshold",
    )

    synergistic = sum(1 for r in synergy_ratios if r > 1.1)
    antagonistic = sum(1 for r in synergy_ratios if r < 0.9)

    ax.set_xlabel("Synergy Ratio (Combined / Expected)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Synergy Ratio Distribution\nSynergistic: {synergistic}, Antagonistic: {antagonistic}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Effect of R-N distance on synergy
    ax = axes[1, 0]
    ax.scatter(
        distances,
        synergy_ratios,
        alpha=0.6,
        s=50,
        c="steelblue",
        edgecolors="white",
    )

    # Trend line
    z = np.polyfit(distances, synergy_ratios, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(distances), max(distances), 100)
    ax.plot(x_line, p(x_line), "r-", lw=2, label=f"Trend (slope={z[0]:.3f})")

    ax.axhline(1.0, color="gray", linestyle="--", lw=1)
    ax.set_xlabel("R-N Distance (residues)", fontsize=12)
    ax.set_ylabel("Synergy Ratio", fontsize=12)
    ax.set_title("Synergy vs R-N Distance", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Goldilocks potentiation
    ax = axes[1, 1]

    # Count potentiation cases
    potentiation_count = sum(1 for r in results if r["goldilocks"]["potentiation"])
    r_in_zone = sum(1 for r in results if r["goldilocks"]["r_in_zone"])
    n_in_zone = sum(1 for r in results if r["goldilocks"]["n_in_zone"])
    combined_in_zone = sum(1 for r in results if r["goldilocks"]["combined_in_zone"])

    categories = [
        "R alone\nin zone",
        "N alone\nin zone",
        "Combined\nin zone",
        "Potentiation\n(new entry)",
    ]
    values = [r_in_zone, n_in_zone, combined_in_zone, potentiation_count]
    colors = ["#e53935", "#1e88e5", "#9c27b0", "#ff9800"]

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
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Goldilocks Zone Analysis (n={len(results)} pairs)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Combinatorial PTM Analysis: Citrullination + Deglycosylation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "combinatorial_synergy_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: combinatorial_synergy_analysis.png")


def plot_top_synergistic_pairs(results: List[Dict], output_dir: Path, top_n: int = 15):
    """Plot the top synergistic pairs."""

    # Sort by synergy ratio
    sorted_results = sorted(results, key=lambda x: x["synergy"]["synergy_ratio"], reverse=True)
    top_pairs = sorted_results[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))

    labels = [f"{r['pair']['protein'][:12]}\nR{r['pair']['r_position']}-N{r['pair']['n_position']}" for r in top_pairs]

    r_shifts = [r["r_alone"]["relative_shift"] * 100 for r in top_pairs]
    n_shifts = [r["n_alone"]["relative_shift"] * 100 for r in top_pairs]
    combined_shifts = [r["combined"]["relative_shift"] * 100 for r in top_pairs]

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        r_shifts,
        width,
        label="R→Q alone",
        color="#e53935",
        alpha=0.7,
    )
    bars2 = ax.bar(x, n_shifts, width, label="N→Q alone", color="#1e88e5", alpha=0.7)
    bars3 = ax.bar(
        x + width,
        combined_shifts,
        width,
        label="Combined",
        color="#9c27b0",
        alpha=0.7,
    )

    # Goldilocks zone
    ax.axhspan(15, 30, alpha=0.15, color="gold", label="Goldilocks Zone")

    ax.set_xlabel("R-N Pair", fontsize=12)
    ax.set_ylabel("Relative Shift (%)", fontsize=12)
    ax.set_title(f"Top {top_n} Synergistic R-N Pairs", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "top_synergistic_pairs.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: top_synergistic_pairs.png")


def main():
    print("=" * 80)
    print("COMBINATORIAL PTM ANALYSIS")
    print("Citrullination + Deglycosylation Synergy")
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

    # Find all R-N pairs
    print("\nFinding proximal R-N pairs (distance ≤ 15 residues)...")
    all_pairs = []
    for protein in proteins:
        pairs = find_rn_pairs(protein, max_distance=15)
        all_pairs.extend([(pair, protein["sequence"]) for pair in pairs])

    print(f"  Found {len(all_pairs)} R-N pairs across all proteins")

    # Analyze each pair
    print("\nAnalyzing combinatorial effects...")
    results = []

    for i, (pair, sequence) in enumerate(all_pairs):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_pairs)} pairs...")

        analysis = analyze_rn_pair(pair, sequence, encoder, device)
        if analysis:
            results.append(analysis)

    print(f"  Successfully analyzed {len(results)} pairs")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SYNERGY ANALYSIS SUMMARY")
    print("=" * 80)

    synergistic = [r for r in results if r["synergy"]["is_synergistic"]]
    antagonistic = [r for r in results if r["synergy"]["is_antagonistic"]]
    additive = [r for r in results if not r["synergy"]["is_synergistic"] and not r["synergy"]["is_antagonistic"]]

    print(f"\n  Total pairs analyzed: {len(results)}")
    print(f"  Synergistic (ratio > 1.1): {len(synergistic)} ({100*len(synergistic)/len(results):.1f}%)")
    print(f"  Antagonistic (ratio < 0.9): {len(antagonistic)} ({100*len(antagonistic)/len(results):.1f}%)")
    print(f"  Additive (0.9-1.1): {len(additive)} ({100*len(additive)/len(results):.1f}%)")

    # Goldilocks potentiation
    potentiation = [r for r in results if r["goldilocks"]["potentiation"]]
    print(f"\n  Goldilocks Potentiation Cases: {len(potentiation)}")
    print("  (Combined enters Goldilocks when neither individual did)")

    if potentiation:
        print("\n  Potentiation pairs:")
        for p in potentiation[:10]:
            print(
                f"    {p['pair']['protein']}: R{p['pair']['r_position']}-N{p['pair']['n_position']} "
                f"(combined shift: {p['combined']['relative_shift']*100:.1f}%)"
            )

    # Distance effect
    synergy_ratios = [r["synergy"]["synergy_ratio"] for r in results]
    distances = [r["pair"]["distance"] for r in results]

    corr_r = None
    if len(distances) > 2:
        corr_r, corr_p = stats.pearsonr(distances, synergy_ratios)
        print(f"\n  Distance-Synergy Correlation: r={corr_r:.3f}, p={corr_p:.4f}")
        if corr_p < 0.05:
            if corr_r > 0:
                print("  → Synergy INCREASES with distance (unexpected)")
            else:
                print("  → Synergy DECREASES with distance (glycan proximity matters)")

    # Top synergistic pairs
    sorted_by_synergy = sorted(results, key=lambda x: x["synergy"]["synergy_ratio"], reverse=True)
    print("\n  Top 5 Synergistic Pairs:")
    for res in sorted_by_synergy[:5]:
        print(
            f"    {res['pair']['protein']}: R{res['pair']['r_position']}-N{res['pair']['n_position']} "
            f"(ratio: {res['synergy']['synergy_ratio']:.2f}, distance: {res['pair']['distance']})"
        )

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_synergy_analysis(results, output_dir)
    plot_top_synergistic_pairs(results, output_dir)

    # Save results
    print("\nSaving results...")

    # Convert for JSON
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_json = {
        "analysis_date": datetime.now().isoformat(),
        "total_pairs": len(results),
        "synergistic_count": len(synergistic),
        "antagonistic_count": len(antagonistic),
        "additive_count": len(additive),
        "potentiation_count": len(potentiation),
        "summary": {
            "synergistic_rate": (float(len(synergistic) / len(results)) if results else 0),
            "potentiation_rate": (float(len(potentiation) / len(results)) if results else 0),
            "mean_synergy_ratio": (float(np.mean(synergy_ratios)) if synergy_ratios else 0),
            "distance_synergy_correlation": (float(corr_r) if corr_r is not None else None),
        },
        "top_synergistic": [convert_numpy(r) for r in sorted_by_synergy[:20]],
        "potentiation_cases": [convert_numpy(r) for r in potentiation],
    }

    results_path = output_dir / "combinatorial_ptm_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {results_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print(
        f"""
COMBINATORIAL PTM ANALYSIS COMPLETE

Hypothesis: Glycan loss potentiates citrullination immunogenicity

Results:
  - Analyzed {len(results)} proximal R-N pairs
  - Synergistic pairs: {len(synergistic)} ({100*len(synergistic)/len(results):.1f}%)
  - Goldilocks potentiation: {len(potentiation)} cases

Interpretation:
"""
    )

    if len(potentiation) > 0:
        print("  ✓ POTENTIATION DETECTED")
        print(f"    {len(potentiation)} R-N pairs enter Goldilocks Zone ONLY when combined")
        print("    This suggests glycan removal can expose citrullination sites")
        print("    to immune recognition")
    else:
        print("  ✗ No potentiation detected")
        print("    Glycan removal does not push citrullination into Goldilocks Zone")

    if len(synergistic) > len(antagonistic):
        print("\n  Overall trend: SYNERGISTIC")
        print("    Combined effects tend to be larger than sum of parts")
    elif len(antagonistic) > len(synergistic):
        print("\n  Overall trend: ANTAGONISTIC")
        print("    Combined effects tend to be smaller than sum of parts")
        print("    Glycans may stabilize protein geometry")
    else:
        print("\n  Overall trend: ADDITIVE")
        print("    Combined effects approximately equal sum of parts")

    print(f"\nOutput: {output_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
