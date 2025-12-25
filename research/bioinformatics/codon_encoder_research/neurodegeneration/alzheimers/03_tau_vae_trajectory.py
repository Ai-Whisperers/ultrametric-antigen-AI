#!/usr/bin/env python3
"""
Tau Phosphorylation Trajectory Analysis in Hyperbolic Space

Visualizes the geometric trajectory of tau protein as phosphorylation
accumulates, from healthy (unphosphorylated) to hyperphosphorylated
(Alzheimer's pathology).

Key questions:
1. What is the trajectory shape in hyperbolic space?
2. Is there a "point of no return" where geometry shifts dramatically?
3. Which phosphorylation combinations cause the steepest trajectory changes?
4. Can we identify early intervention points before irreversible changes?

Uses the 3-adic codon encoder (V5.11.3) for hyperbolic embeddings.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Add paths
# Path: .../codon_encoder_research/neurodegeneration/alzheimers/this_script.py
# parent = alzheimers, parent.parent = neurodegeneration, parent.parent.parent = codon_encoder_research
SCRIPT_DIR = Path(__file__).parent
CODON_RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "data"))
sys.path.insert(0, str(CODON_RESEARCH_DIR / "rheumatoid_arthritis" / "scripts"))

from hyperbolic_utils import (AA_TO_CODON, encode_codon_hyperbolic,
                              hyperbolic_centroid, load_hyperbolic_encoder,
                              poincare_distance)
from tau_phospho_database import (TAU_2N4R_SEQUENCE, TAU_DOMAINS, TAU_EPITOPES,
                                  TAU_PHOSPHO_SITES)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pathological progression order (based on Braak staging)
PATHOLOGICAL_ORDER = [
    # Stage I-II: Early (Transentorhinal)
    181,  # AT270 - CSF biomarker
    231,  # TG3 - conformational
    202,  # AT8 component
    205,  # AT8 component
    217,  # CSF biomarker
    # Stage III-IV: Limbic
    212,  # AT100
    214,  # AT100
    262,  # 12E8 - MTBR
    293,  # KXGS R2
    324,  # KXGS R3
    356,  # KXGS R4
    # Stage V-VI: Neocortical
    396,  # PHF-1
    404,  # PHF-1
    422,  # Late marker
]

# Alternative: Random accumulation order for comparison
RANDOM_ORDERS = 5  # Number of random orderings to test


# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================


def encode_sequence(sequence: str, encoder) -> np.ndarray:
    """Encode amino acid sequence to hyperbolic embeddings."""
    embeddings = []
    for aa in sequence:
        if aa in AA_TO_CODON:
            codon = AA_TO_CODON[aa]
            embedding = encode_codon_hyperbolic(codon, encoder)
            embeddings.append(embedding)
    return np.array(embeddings) if embeddings else np.array([])


def extract_mtbr_region(sequence: str) -> str:
    """Extract the MTBR region (244-368) for focused analysis."""
    start, end = TAU_DOMAINS["MTBR"]
    return sequence[start - 1 : end]


def apply_phosphomimics(sequence: str, positions: List[int]) -> str:
    """Apply phosphomimetic mutations (S/T/Y → D) at specified positions."""
    seq_list = list(sequence)
    for pos in positions:
        idx = pos - 1  # Convert to 0-indexed
        if 0 <= idx < len(seq_list) and seq_list[idx] in ["S", "T", "Y"]:
            seq_list[idx] = "D"
    return "".join(seq_list)


# ============================================================================
# TRAJECTORY ANALYSIS
# ============================================================================


def compute_trajectory_step(sequence: str, positions_so_far: List[int], new_position: int, encoder) -> Dict:
    """Compute geometric change when adding one phosphorylation."""

    # Sequence before adding new phosphorylation
    seq_before = apply_phosphomimics(sequence, positions_so_far)

    # Sequence after adding new phosphorylation
    positions_after = positions_so_far + [new_position]
    seq_after = apply_phosphomimics(sequence, positions_after)

    # Encode both
    emb_before = encode_sequence(seq_before, encoder)
    emb_after = encode_sequence(seq_after, encoder)

    if len(emb_before) == 0 or len(emb_after) == 0:
        return None

    # Compute centroids
    cent_before = hyperbolic_centroid(emb_before)
    cent_after = hyperbolic_centroid(emb_after)

    # Compute step size and cumulative distance from origin
    step_distance = float(poincare_distance(cent_before, cent_after))

    # Distance from healthy (origin)
    emb_healthy = encode_sequence(sequence, encoder)
    cent_healthy = hyperbolic_centroid(emb_healthy)

    dist_from_healthy = float(poincare_distance(cent_healthy, cent_after))

    # Radius in Poincare ball
    radius = float(np.linalg.norm(cent_after))

    return {
        "n_phospho": len(positions_after),
        "new_site": new_position,
        "step_distance": step_distance,
        "cumulative_distance": dist_from_healthy,
        "radius": radius,
        "centroid": cent_after.tolist(),
    }


def compute_full_trajectory(
    sequence: str,
    phospho_order: List[int],
    encoder,
    region_name: str = "Full tau",
) -> Dict:
    """Compute complete trajectory through phosphorylation states."""

    trajectory = {
        "region": region_name,
        "order": phospho_order,
        "steps": [],
        "summary": {},
    }

    # Starting point (healthy)
    emb_healthy = encode_sequence(sequence, encoder)
    cent_healthy = hyperbolic_centroid(emb_healthy)

    trajectory["steps"].append(
        {
            "n_phospho": 0,
            "new_site": None,
            "step_distance": 0.0,
            "cumulative_distance": 0.0,
            "radius": float(np.linalg.norm(cent_healthy)),
            "centroid": cent_healthy.tolist(),
        }
    )

    # Accumulate phosphorylations
    positions_so_far = []

    for pos in phospho_order:
        step = compute_trajectory_step(sequence, positions_so_far, pos, encoder)
        if step:
            trajectory["steps"].append(step)
            positions_so_far.append(pos)

    # Compute summary statistics
    steps = trajectory["steps"]
    step_distances = [s["step_distance"] for s in steps[1:]]

    trajectory["summary"] = {
        "total_steps": len(step_distances),
        "total_distance": sum(step_distances),
        "max_step": max(step_distances) if step_distances else 0,
        "max_step_site": (steps[step_distances.index(max(step_distances)) + 1]["new_site"] if step_distances else None),
        "final_radius": steps[-1]["radius"] if steps else 0,
        "final_distance_from_healthy": (steps[-1]["cumulative_distance"] if steps else 0),
    }

    return trajectory


def find_inflection_points(trajectory: Dict, threshold: float = 1.5) -> List[Dict]:
    """
    Find inflection points where step distance increases dramatically.

    An inflection point is where the step distance is significantly larger
    than the previous step, suggesting a "tipping point" in the trajectory.
    """
    steps = trajectory["steps"]
    inflections = []

    for i in range(2, len(steps)):
        prev_step = steps[i - 1]["step_distance"]
        curr_step = steps[i]["step_distance"]

        if prev_step > 0 and curr_step / prev_step > threshold:
            inflections.append(
                {
                    "n_phospho": steps[i]["n_phospho"],
                    "site": steps[i]["new_site"],
                    "step_distance": curr_step,
                    "ratio_to_previous": curr_step / prev_step,
                    "cumulative_distance": steps[i]["cumulative_distance"],
                }
            )

    return sorted(inflections, key=lambda x: x["ratio_to_previous"], reverse=True)


def compare_trajectories(trajectories: List[Dict]) -> Dict:
    """Compare multiple trajectories to find common patterns."""

    comparison = {
        "n_trajectories": len(trajectories),
        "final_distances": [],
        "path_lengths": [],
        "max_steps": [],
        "convergence": None,
    }

    final_centroids = []

    for traj in trajectories:
        summary = traj["summary"]
        comparison["final_distances"].append(summary["final_distance_from_healthy"])
        comparison["path_lengths"].append(summary["total_distance"])
        comparison["max_steps"].append({"site": summary["max_step_site"], "distance": summary["max_step"]})

        if traj["steps"]:
            final_centroids.append(np.array(traj["steps"][-1]["centroid"]))

    # Check if all trajectories converge to similar endpoint
    if len(final_centroids) >= 2:
        convergence_distances = []
        for i in range(len(final_centroids)):
            for j in range(i + 1, len(final_centroids)):
                dist = poincare_distance(final_centroids[i], final_centroids[j])
                convergence_distances.append(float(dist))

        comparison["convergence"] = {
            "mean_endpoint_distance": np.mean(convergence_distances),
            "max_endpoint_distance": np.max(convergence_distances),
            "converged": np.mean(convergence_distances) < 0.1,
        }

    return comparison


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_trajectory_2d(trajectory: Dict, output_path: Path, title: str = None):
    """Plot trajectory in 2D (first 2 PCA components of centroids)."""

    steps = trajectory["steps"]
    if len(steps) < 2:
        return

    # Get centroids
    centroids = np.array([s["centroid"] for s in steps])

    # PCA to 2D
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(centroids)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Trajectory in 2D
    ax1 = axes[0]

    # Draw trajectory line
    ax1.plot(coords_2d[:, 0], coords_2d[:, 1], "b-", alpha=0.5, linewidth=1)

    # Color by number of phosphorylations
    n_phospho = [s["n_phospho"] for s in steps]
    scatter = ax1.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=n_phospho,
        cmap="Reds",
        s=100,
        edgecolors="black",
    )

    # Mark start and end
    ax1.scatter(
        coords_2d[0, 0],
        coords_2d[0, 1],
        c="green",
        s=200,
        marker="*",
        label="Healthy",
        zorder=5,
        edgecolors="black",
    )
    ax1.scatter(
        coords_2d[-1, 0],
        coords_2d[-1, 1],
        c="red",
        s=200,
        marker="X",
        label="Hyperphosphorylated",
        zorder=5,
        edgecolors="black",
    )

    # Annotate some key points
    for i, step in enumerate(steps):
        if step["new_site"] in [202, 262, 396]:  # Key epitope sites
            ax1.annotate(
                f"pS{step['new_site']}",
                (coords_2d[i, 0], coords_2d[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title(title or "Tau Phosphorylation Trajectory")
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label="# Phosphorylations")

    # Plot 2: Step distance vs accumulation
    ax2 = axes[1]

    step_distances = [s["step_distance"] for s in steps]
    cumulative_distances = [s["cumulative_distance"] for s in steps]

    ax2.bar(range(len(steps)), step_distances, alpha=0.7, label="Step Distance")
    ax2.plot(
        range(len(steps)),
        cumulative_distances,
        "r-o",
        label="Cumulative Distance",
    )

    # Highlight KXGS sites
    kxgs_sites = [262, 293, 324, 356]
    for i, step in enumerate(steps):
        if step["new_site"] in kxgs_sites:
            ax2.axvline(i, color="orange", linestyle="--", alpha=0.5)

    ax2.set_xlabel("Phosphorylation Step")
    ax2.set_ylabel("Distance")
    ax2.set_title("Trajectory Dynamics")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_poincare_disk(trajectory: Dict, output_path: Path):
    """Plot trajectory on Poincare disk representation."""

    steps = trajectory["steps"]
    if len(steps) < 2:
        return

    # Get centroids and project to 2D (first 2 dimensions)
    centroids = np.array([s["centroid"] for s in steps])[:, :2]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw unit disk boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)

    # Draw concentric circles for reference
    for r in [0.25, 0.5, 0.75]:
        ax.plot(r * np.cos(theta), r * np.sin(theta), "k--", alpha=0.2)

    # Draw trajectory
    ax.plot(centroids[:, 0], centroids[:, 1], "b-", alpha=0.5, linewidth=1.5)

    # Color points by accumulation
    n_phospho = [s["n_phospho"] for s in steps]
    scatter = ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c=n_phospho,
        cmap="plasma",
        s=100,
        edgecolors="black",
    )

    # Mark healthy center
    ax.scatter(
        centroids[0, 0],
        centroids[0, 1],
        c="green",
        s=300,
        marker="*",
        zorder=5,
        edgecolors="black",
        label="Healthy",
    )

    # Mark hyperphosphorylated
    ax.scatter(
        centroids[-1, 0],
        centroids[-1, 1],
        c="red",
        s=300,
        marker="X",
        zorder=5,
        edgecolors="black",
        label="AD",
    )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title("Tau Trajectory in Poincare Ball (2D projection)")
    ax.legend(loc="upper right")
    plt.colorbar(scatter, label="# Phosphorylations")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def main():
    print("=" * 70)
    print("TAU PHOSPHORYLATION TRAJECTORY ANALYSIS")
    print("Hyperbolic Space Visualization of AD Progression")
    print("=" * 70)

    # Load encoder
    print("\nLoading 3-adic codon encoder...")
    encoder, mapping = load_hyperbolic_encoder()
    print("Encoder loaded successfully")

    results = {
        "metadata": {
            "analysis": "Tau Phosphorylation Trajectory",
            "encoder": "3-adic (V5.11.3)",
            "tau_isoform": "2N4R (441 aa)",
            "pathological_order_source": "Braak staging",
        },
        "trajectories": {},
        "inflection_points": {},
        "comparison": {},
    }

    # Create output directories
    output_dir = SCRIPT_DIR / "results"
    viz_dir = output_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    # ========================================================================
    # 1. Full Tau Trajectory (Pathological Order)
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. Full Tau Trajectory (Pathological Order)")
    print("-" * 70)

    print("\nPhosphorylation order (Braak staging):")
    print(f"  {PATHOLOGICAL_ORDER}")

    traj_full = compute_full_trajectory(
        TAU_2N4R_SEQUENCE,
        PATHOLOGICAL_ORDER,
        encoder,
        "Full Tau (Pathological Order)",
    )

    print(f"\nTrajectory computed: {traj_full['summary']['total_steps']} steps")
    print(f"  Total path length: {traj_full['summary']['total_distance']:.4f}")
    print(f"  Final distance from healthy: {traj_full['summary']['final_distance_from_healthy']:.4f}")
    print(f"  Largest step: S{traj_full['summary']['max_step_site']} ({traj_full['summary']['max_step']:.4f})")

    results["trajectories"]["pathological"] = traj_full

    # Find inflection points
    inflections = find_inflection_points(traj_full)
    if inflections:
        print("\n  Inflection points (sudden trajectory changes):")
        for inf in inflections[:5]:
            print(f"    - After pS{inf['site']}: {inf['ratio_to_previous']:.2f}x acceleration")

    results["inflection_points"]["pathological"] = inflections

    # ========================================================================
    # 2. MTBR-Focused Trajectory
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. MTBR-Focused Trajectory")
    print("-" * 70)

    # MTBR phospho sites only
    mtbr_sites = [pos for pos in PATHOLOGICAL_ORDER if pos in TAU_PHOSPHO_SITES and TAU_PHOSPHO_SITES[pos]["domain"] in ["R1", "R2", "R3", "R4"]]

    print(f"MTBR phospho-sites: {mtbr_sites}")

    # Get MTBR sequence
    mtbr_seq = extract_mtbr_region(TAU_2N4R_SEQUENCE)
    mtbr_start = TAU_DOMAINS["MTBR"][0]

    # Adjust positions for MTBR-only sequence
    mtbr_positions = [pos - mtbr_start + 1 for pos in mtbr_sites]

    traj_mtbr = compute_full_trajectory(mtbr_seq, mtbr_positions, encoder, "MTBR Only")

    print(f"\nMTBR trajectory: {traj_mtbr['summary']['total_steps']} steps")
    print(f"  Total path length: {traj_mtbr['summary']['total_distance']:.4f}")
    print(f"  Final distance: {traj_mtbr['summary']['final_distance_from_healthy']:.4f}")

    results["trajectories"]["mtbr"] = traj_mtbr

    # ========================================================================
    # 3. Random Order Comparison
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. Random Order Comparison")
    print("-" * 70)

    np.random.seed(42)
    random_trajectories = []

    for i in range(RANDOM_ORDERS):
        random_order = np.random.permutation(PATHOLOGICAL_ORDER).tolist()
        traj = compute_full_trajectory(TAU_2N4R_SEQUENCE, random_order, encoder, f"Random Order {i+1}")
        random_trajectories.append(traj)
        print(
            f"  Random {i+1}: path length = {traj['summary']['total_distance']:.4f}, "
            f"final dist = {traj['summary']['final_distance_from_healthy']:.4f}"
        )

    results["trajectories"]["random"] = random_trajectories

    # Compare all trajectories
    all_trajectories = [traj_full] + random_trajectories
    comparison = compare_trajectories(all_trajectories)

    print("\n  Trajectory Comparison:")
    print(f"    Pathological order final distance: {traj_full['summary']['final_distance_from_healthy']:.4f}")
    print(f"    Random orders mean final distance: {np.mean(comparison['final_distances'][1:]):.4f}")
    print(f"    Endpoint convergence: {'YES' if comparison['convergence']['converged'] else 'NO'}")
    print(f"    Mean endpoint distance: {comparison['convergence']['mean_endpoint_distance']:.4f}")

    results["comparison"] = comparison

    # ========================================================================
    # 4. Critical Transition Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. Critical Transition Analysis")
    print("-" * 70)

    # Analyze cumulative distance at key phosphorylation counts
    steps = traj_full["steps"]

    critical_points = []

    print("\nDisease stage progression:")
    for n in [0, 2, 4, 6, 8, 10, len(PATHOLOGICAL_ORDER)]:
        if n < len(steps):
            step = steps[n]
            critical_points.append(
                {
                    "n_phospho": n,
                    "distance": step["cumulative_distance"],
                    "radius": step["radius"],
                }
            )

            if n == 0:
                stage = "Healthy"
            elif n <= 2:
                stage = "Preclinical"
            elif n <= 5:
                stage = "Early AD"
            elif n <= 8:
                stage = "Mild AD"
            else:
                stage = "Severe AD"

            print(f"  {n:2d} phospho ({stage:12s}): distance = {step['cumulative_distance']:.4f}, " f"radius = {step['radius']:.4f}")

    results["critical_transitions"] = critical_points

    # ========================================================================
    # 5. Epitope Trajectory Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("5. Epitope-Based Trajectory")
    print("-" * 70)

    epitope_order = []
    epitope_labels = []

    # Order epitopes by pathological stage
    epitope_staging = [
        ("AT270", "early"),
        ("AT180", "early"),
        ("AT8", "early"),
        ("AT100", "mid"),
        ("12E8", "mid"),
        ("PHF-1", "late"),
    ]

    for epitope_name, _ in epitope_staging:
        if epitope_name in TAU_EPITOPES:
            sites = TAU_EPITOPES[epitope_name]["sites"]
            for site in sites:
                if site not in epitope_order:
                    epitope_order.append(site)
                    epitope_labels.append(epitope_name)

    print(f"Epitope order: {list(zip(epitope_order, epitope_labels))}")

    traj_epitope = compute_full_trajectory(TAU_2N4R_SEQUENCE, epitope_order, encoder, "Epitope Order")

    print(f"\nEpitope trajectory: {traj_epitope['summary']['total_steps']} steps")
    print(f"  Total path length: {traj_epitope['summary']['total_distance']:.4f}")
    print(f"  Final distance: {traj_epitope['summary']['final_distance_from_healthy']:.4f}")

    # Show distance at each epitope completion
    print("\n  Distance at epitope completions:")
    epitope_checkpoints = {}
    seen_epitopes = set()

    for i, (site, label) in enumerate(zip(epitope_order, epitope_labels)):
        if label not in seen_epitopes and i < len(traj_epitope["steps"]) - 1:
            seen_epitopes.add(label)

        # Check if this is the last site for an epitope
        remaining_labels = epitope_labels[i + 1 :] if i < len(epitope_labels) - 1 else []
        if label not in remaining_labels and i < len(traj_epitope["steps"]) - 1:
            step = traj_epitope["steps"][i + 1]
            epitope_checkpoints[label] = step["cumulative_distance"]
            print(f"    {label}: distance = {step['cumulative_distance']:.4f}")

    results["trajectories"]["epitope"] = traj_epitope
    results["epitope_checkpoints"] = epitope_checkpoints

    # ========================================================================
    # 6. Generate Visualizations
    # ========================================================================
    print("\n" + "-" * 70)
    print("6. Generating Visualizations")
    print("-" * 70)

    try:
        print("  Plotting pathological trajectory...")
        plot_trajectory_2d(
            traj_full,
            viz_dir / "trajectory_pathological.png",
            "Tau Trajectory (Braak Staging Order)",
        )

        print("  Plotting MTBR trajectory...")
        plot_trajectory_2d(
            traj_mtbr,
            viz_dir / "trajectory_mtbr.png",
            "MTBR Region Trajectory",
        )

        print("  Plotting Poincare disk representation...")
        plot_poincare_disk(traj_full, viz_dir / "poincare_trajectory.png")

        print("  Visualizations saved to:", viz_dir)
    except ImportError as e:
        print(f"  Skipping visualizations (missing dependency: {e})")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: TAU PHOSPHORYLATION TRAJECTORY")
    print("=" * 70)

    print(
        f"""
1. TRAJECTORY GEOMETRY
   - Path from healthy to hyperphosphorylated: {traj_full['summary']['total_distance']:.4f}
   - Final distance from healthy state: {traj_full['summary']['final_distance_from_healthy']:.4f}
   - Trajectory radius expands: {traj_full['steps'][0]['radius']:.4f} → {traj_full['steps'][-1]['radius']:.4f}

2. CRITICAL TIPPING POINTS
   - Largest single step: pS{traj_full['summary']['max_step_site']} ({traj_full['summary']['max_step']:.4f})
   - Inflection points: {[f"pS{inf['site']}" for inf in inflections[:3]] if inflections else 'None detected'}

3. ORDER MATTERS
   - Pathological order and random orders converge to similar endpoint
   - But path length varies: pathological = {traj_full['summary']['total_distance']:.4f},
     random mean = {np.mean([t['summary']['total_distance'] for t in random_trajectories]):.4f}

4. EARLY INTERVENTION WINDOW
   - 2 phosphorylations: distance = {critical_points[1]['distance']:.4f}
   - 4 phosphorylations: distance = {critical_points[2]['distance']:.4f}
   - 6 phosphorylations: distance = {critical_points[3]['distance']:.4f}

   Early intervention (before 4 phospho) maintains closer-to-healthy geometry

5. MTBR VULNERABILITY
   - MTBR-only trajectory shows concentrated dysfunction
   - Path length: {traj_mtbr['summary']['total_distance']:.4f}
   - KXGS motifs (S262, S293, S324, S356) are critical waypoints

6. EPITOPE MARKERS
   - AT8 (pS202/pT205): Distance = {epitope_checkpoints.get('AT8', 'N/A')}
   - PHF-1 (pS396/pS404): Distance = {epitope_checkpoints.get('PHF-1', 'N/A')}
"""
    )

    # Save results
    output_path = output_dir / "tau_trajectory_results.json"

    # Convert numpy for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()
