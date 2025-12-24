#!/usr/bin/env python3
"""
Variational Orthogonality Test (Conjecture 36)
==============================================

Tests whether hyperbolic curvature creates additional effective degrees of freedom
by measuring intervention independence: which latent dimensions control which output trits.

Key insight: Degrees of freedom = statistically independent directions of variation,
not geometric perpendicularity. If latent dims control non-overlapping output trits,
they're truly independent regardless of their geometric angle.

Hypothesis: Points at higher radius (near boundary) will show LESS control overlap,
meaning curvature "creates" effective degrees of freedom.

Optimization: Use model's O(log n) hierarchical structure instead of O(n) brute force.
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))


def load_decoder():
    """Load the v5.5 decoder for intervention testing."""
    from src.models.ternary_vae import FrozenDecoder

    checkpoint_path = (
        Path(__file__).parent.parent
        / "sandbox-training"
        / "checkpoints"
        / "v5_5"
        / "best.pt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model", checkpoint.get("model_state", {}))

    # Build decoder
    decoder = FrozenDecoder(latent_dim=16)

    # Load decoder weights
    dec_state = {}
    for key, value in model_state.items():
        if key.startswith("decoder_A."):
            new_key = key.replace("decoder_A.", "")
            dec_state[new_key] = value

    decoder.load_state_dict(dec_state, strict=False)
    decoder.to(device)
    decoder.eval()

    return decoder, device


def decode_to_trits(decoder, z, device):
    """Decode latent z to trit predictions."""
    with torch.no_grad():
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device)
        if z_tensor.dim() == 1:
            z_tensor = z_tensor.unsqueeze(0)

        # Decode (decoder expects Euclidean, we pass hyperbolic but it still works)
        logits = decoder(z_tensor)  # [B, 9, 3]
        trits = torch.argmax(logits, dim=-1)  # [B, 9]

    return trits


def compute_control_matrix(decoder, z, device, epsilon=0.01):
    """
    Compute which latent dimensions control which output trits.

    For point z, perturb each of 16 latent dims by ±epsilon,
    record which of 9 output trits change.

    Returns: 16x9 control matrix where entry (i,j) = 1 if dim i affects trit j
    """
    latent_dim = z.shape[-1]
    n_trits = 9

    # Get baseline prediction
    baseline_trits = decode_to_trits(decoder, z, device)[0].cpu().numpy()

    control_matrix = np.zeros((latent_dim, n_trits))

    for dim in range(latent_dim):
        # Positive perturbation
        z_plus = z.copy()
        z_plus[dim] += epsilon

        trits_plus = decode_to_trits(decoder, z_plus, device)[0].cpu().numpy()

        # Negative perturbation
        z_minus = z.copy()
        z_minus[dim] -= epsilon
        trits_minus = decode_to_trits(decoder, z_minus, device)[0].cpu().numpy()

        # Record which trits changed
        changed_plus = (trits_plus != baseline_trits).astype(float)
        changed_minus = (trits_minus != baseline_trits).astype(float)

        control_matrix[dim] = np.maximum(changed_plus, changed_minus)

    return control_matrix


def compute_control_overlap(control_matrix):
    """
    Compute overlap between latent dimensions' control.

    If dims i and j both control trit k, they're not independent.
    Returns average pairwise overlap (0 = fully independent, 1 = fully overlapping)
    """
    latent_dim = control_matrix.shape[0]

    overlaps = []
    for i in range(latent_dim):
        for j in range(i + 1, latent_dim):
            # Overlap = how many trits both dims control
            both_control = np.sum(control_matrix[i] * control_matrix[j])
            either_control = np.sum(np.maximum(control_matrix[i], control_matrix[j]))

            if either_control > 0:
                overlap = both_control / either_control
            else:
                overlap = 0

            overlaps.append(overlap)

    return np.mean(overlaps) if overlaps else 0


def compute_effective_dimensions(control_matrix):
    """
    Compute effective degrees of freedom from control matrix.

    Uses SVD: number of significant singular values = effective dimensions
    """
    u, s, vh = np.linalg.svd(control_matrix)

    # Count singular values above 10% of max
    threshold = 0.1 * s[0] if s[0] > 0 else 0.1
    effective_dims = np.sum(s > threshold)

    return effective_dims, s


def hierarchical_sampling(embeddings, valuations, n_per_level=20):
    """
    O(log n) sampling using hierarchical structure.

    Instead of sampling all 19,683 points, sample from each valuation level.
    This uses the model's learned hierarchy for efficient exploration.
    """
    samples = []
    sample_info = []

    unique_vals = sorted(set(valuations))

    for v in unique_vals:
        if v >= 99:  # Skip special cases
            continue

        mask = valuations == v
        level_embeddings = embeddings[mask]
        level_indices = np.where(mask)[0]

        # Sample up to n_per_level from this level
        n_available = len(level_embeddings)
        n_sample = min(n_per_level, n_available)

        if n_sample > 0:
            idx = np.random.choice(n_available, n_sample, replace=False)
            samples.extend(level_embeddings[idx])
            sample_info.extend(
                [{"valuation": v, "index": int(level_indices[i])} for i in idx]
            )

    return np.array(samples), sample_info


def run_test():
    """Run the variational orthogonality test."""
    print("=" * 70)
    print("VARIATIONAL ORTHOGONALITY TEST (Conjecture 36)")
    print("Testing: Does curvature create effective degrees of freedom?")
    print("=" * 70)

    # Load decoder
    print("\nLoading decoder...")
    decoder, device = load_decoder()

    # Load embeddings
    embeddings_path = Path(__file__).parent / "embeddings" / "z_hyperbolic.npy"
    embeddings = np.load(embeddings_path)

    # Compute valuations for hierarchical sampling
    def v3(n):
        if n == 0:
            return 99
        v = 0
        while n % 3 == 0:
            n //= 3
            v += 1
        return v

    valuations = np.array([v3(i) for i in range(len(embeddings))])
    radii = np.linalg.norm(embeddings, axis=1)

    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Radius range: [{radii.min():.4f}, {radii.max():.4f}]")

    # Hierarchical sampling (O(log n) instead of O(n))
    print("\nHierarchical sampling (O(log n))...")
    samples, sample_info = hierarchical_sampling(embeddings, valuations, n_per_level=15)
    print(
        f"Sampled {len(samples)} points across {len(set(valuations))-1} hierarchy levels"
    )

    # Compute control matrices for all samples
    print("\nComputing intervention independence (control matrices)...")

    results_by_radius = {}
    all_results = []

    # Stratify by radius bins
    sample_radii = np.linalg.norm(samples, axis=1)
    radius_bins = [(0.4, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95)]

    for i, (z, info) in enumerate(zip(samples, sample_info)):
        if i % 20 == 0:
            print(f"  Processing sample {i+1}/{len(samples)}...")

        r = np.linalg.norm(z)

        # Compute control matrix
        control = compute_control_matrix(decoder, z, device, epsilon=0.02)

        # Compute metrics
        overlap = compute_control_overlap(control)
        eff_dims, singular_vals = compute_effective_dimensions(control)
        active_dims = np.sum(np.any(control > 0, axis=1))
        active_trits = np.sum(np.any(control > 0, axis=0))

        result = {
            "radius": float(r),
            "valuation": info["valuation"],
            "overlap": float(overlap),
            "effective_dims": int(eff_dims),
            "active_dims": int(active_dims),
            "active_trits": int(active_trits),
            "singular_values": singular_vals.tolist(),
            "control_density": float(np.mean(control)),
        }
        all_results.append(result)

        # Bin by radius
        for r_min, r_max in radius_bins:
            if r_min <= r < r_max:
                bin_key = f"{r_min:.2f}-{r_max:.2f}"
                if bin_key not in results_by_radius:
                    results_by_radius[bin_key] = []
                results_by_radius[bin_key].append(result)
                break

    # Analyze by radius bin
    print("\n" + "=" * 70)
    print("RESULTS: Effective Dimensions by Radius")
    print("=" * 70)
    print(
        f"\n{'Radius Bin':<15} {'N':<5} {'Eff Dims':<10} {'Overlap':<10} {'Active Dims':<12}"
    )
    print("-" * 60)

    bin_summary = {}
    for bin_key in sorted(results_by_radius.keys()):
        bin_results = results_by_radius[bin_key]
        n = len(bin_results)

        avg_eff_dims = np.mean([r["effective_dims"] for r in bin_results])
        avg_overlap = np.mean([r["overlap"] for r in bin_results])
        avg_active = np.mean([r["active_dims"] for r in bin_results])

        print(
            f"{bin_key:<15} {n:<5} {avg_eff_dims:<10.2f} {avg_overlap:<10.3f} {avg_active:<12.2f}"
        )

        bin_summary[bin_key] = {
            "n_samples": n,
            "avg_effective_dims": float(avg_eff_dims),
            "avg_overlap": float(avg_overlap),
            "avg_active_dims": float(avg_active),
        }

    # Test the hypothesis
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST: Does curvature create degrees of freedom?")
    print("=" * 70)

    sorted_bins = sorted(bin_summary.keys())
    if len(sorted_bins) >= 2:
        inner_bin = sorted_bins[0]
        outer_bin = sorted_bins[-1]

        inner_dims = bin_summary[inner_bin]["avg_effective_dims"]
        outer_dims = bin_summary[outer_bin]["avg_effective_dims"]
        inner_overlap = bin_summary[inner_bin]["avg_overlap"]
        outer_overlap = bin_summary[outer_bin]["avg_overlap"]

        dims_increase = outer_dims - inner_dims
        overlap_decrease = inner_overlap - outer_overlap

        print(f"\nInner region ({inner_bin}):")
        print(f"  Effective dimensions: {inner_dims:.2f}")
        print(f"  Control overlap: {inner_overlap:.3f}")

        print(f"\nOuter region ({outer_bin}):")
        print(f"  Effective dimensions: {outer_dims:.2f}")
        print(f"  Control overlap: {outer_overlap:.3f}")

        print(f"\nChange from inner to outer:")
        print(f"  Effective dims: {dims_increase:+.2f}")
        print(f"  Overlap: {-overlap_decrease:+.3f}")

        # Verdict
        print("\n" + "-" * 70)
        print("VERDICT:")
        print("-" * 70)

        if dims_increase > 0.5 or overlap_decrease > 0.05:
            print(
                "SUPPORTED: Curvature appears to create additional degrees of freedom."
            )
            print(f"  - Effective dims increase by {dims_increase:.1f} toward boundary")
            print(f"  - Control overlap decreases by {overlap_decrease:.3f}")
            verdict = "supported"
        elif dims_increase < -0.5 or overlap_decrease < -0.05:
            print(
                "CONTRARY: Curvature appears to REDUCE degrees of freedom near boundary."
            )
            verdict = "contrary"
        else:
            print("INCONCLUSIVE: No clear trend in effective dimensions with radius.")
            verdict = "inconclusive"
    else:
        verdict = "insufficient_data"
        print("Insufficient data for hypothesis test")

    # Additional analysis: correlation with valuation
    print("\n" + "=" * 70)
    print("BONUS: Effective Dimensions by Hierarchy Level (Valuation)")
    print("=" * 70)

    val_results = {}
    for r in all_results:
        v = r["valuation"]
        if v not in val_results:
            val_results[v] = []
        val_results[v].append(r)

    print(f"\n{'Valuation':<10} {'N':<5} {'Eff Dims':<10} {'Overlap':<10}")
    print("-" * 40)

    for v in sorted(val_results.keys()):
        vr = val_results[v]
        print(
            f"v_3 = {v:<5} {len(vr):<5} "
            f"{np.mean([r['effective_dims'] for r in vr]):<10.2f} "
            f"{np.mean([r['overlap'] for r in vr]):<10.3f}"
        )

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "conjecture": "Conjecture 36: Variational Orthogonality",
        "hypothesis": "Curvature creates effective degrees of freedom",
        "verdict": verdict,
        "bin_summary": bin_summary,
        "all_results": all_results,
    }

    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {
                str(k) if isinstance(k, np.integer) else k: convert_numpy(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    output_path = results_dir / "variational_orthogonality_test.json"
    with open(output_path, "w") as f:
        json.dump(convert_numpy(output), f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    np.random.seed(42)
    run_test()
