#!/usr/bin/env python3
"""Projection/Deformation Analysis: Is viral space a "fuzzy projection" of p-adic space?

Core Hypothesis:
If 3-adic (codon grammar) shows ρ ≈ 0 with viral conservation, but the viral space
IS ultrametric under some other prime p, then there might be a PROJECTION or
DEFORMATION relationship:

    viral_distance ≈ f(3-adic_distance) + noise

where f is some monotonic (or non-linear) transformation.

This script tests:
1. Is there a systematic relationship (even non-linear) between 3-adic and viral distances?
2. If we find the "correct" prime for viral space, can we find a projection matrix?
3. What is the "residual structure" after removing p-adic effects?

Mathematical Framework:
- Let d_3(x,y) = 3-adic distance (from codon encoder)
- Let d_v(x,y) = viral evolutionary distance (sequence similarity)
- Let d_p(x,y) = p-adic distance for various primes

Test: Is d_v ≈ A * d_3 + B * d_p + noise for some p ≠ 3?

If yes: Viral space is a "mixed projection" of multiple p-adic structures
If no: Viral evolution operates in a fundamentally different geometry

Usage:
    python 02_projection_deformation_analysis.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Setup paths
_script_dir = Path(__file__).resolve().parent
_package_root = _script_dir.parents[1]
_project_root = _package_root.parents[3]

sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_package_root))


@dataclass
class ProjectionAnalysis:
    """Results from projection/deformation analysis."""
    # Linear projection coefficients
    coefficients: dict  # prime -> coefficient
    intercept: float
    r2_score: float
    residual_std: float

    # Non-linear analysis
    spearman_by_prime: dict  # prime -> spearman with viral distance
    best_nonlinear_prime: int
    best_nonlinear_rho: float

    # Residual structure
    residual_ultrametric_compliance: float
    residual_has_structure: bool


def compute_sequence_distance(seq1: str, seq2: str) -> float:
    """Compute normalized Hamming distance between sequences."""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]

    if len(seq1) == 0:
        return 0.0

    mismatches = sum(1 for a, b in zip(seq1.upper(), seq2.upper()) if a != b)
    return mismatches / len(seq1)


def padic_valuation(n: int, p: int) -> int:
    """Compute p-adic valuation."""
    if n == 0:
        return 100
    val = 0
    while n % p == 0:
        val += 1
        n //= p
    return val


def padic_distance(x: int, y: int, p: int) -> float:
    """Compute p-adic distance."""
    diff = abs(x - y)
    if diff == 0:
        return 0.0
    val = padic_valuation(diff, p)
    return float(p) ** (-val)


def sequence_to_index(seq: str) -> int:
    """Convert sequence to integer index (base-4)."""
    base_map = {'A': 0, 'T': 1, 'U': 1, 'G': 2, 'C': 3}
    idx = 0
    for base in seq.upper():
        if base in base_map:
            idx = idx * 4 + base_map[base]
    return idx


def compute_hyperbolic_distances(sequences: list[str]) -> np.ndarray:
    """Compute pairwise hyperbolic distances using TrainableCodonEncoder."""

    try:
        import torch
        from src.encoders.trainable_codon_encoder import TrainableCodonEncoder
        from src.geometry import poincare_distance

        # Load encoder
        checkpoint_path = _project_root / "research" / "codon-encoder" / "training" / "results" / "trained_codon_encoder.pt"

        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found at {checkpoint_path}")
            return None

        encoder = TrainableCodonEncoder(latent_dim=16, hidden_dim=64)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        encoder.load_state_dict(ckpt['model_state_dict'])
        encoder.eval()

        # Encode sequences
        n = len(sequences)
        embeddings = []

        for seq in sequences:
            # Get codon embeddings and average
            codon_embs = []
            seq = seq.upper().replace('U', 'T')
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                try:
                    from src.biology.codons import CODON_TO_INDEX
                    if codon in CODON_TO_INDEX:
                        idx = CODON_TO_INDEX[codon]
                        with torch.no_grad():
                            emb = encoder(torch.tensor([idx]))
                        codon_embs.append(emb.numpy())
                except Exception:
                    pass

            if codon_embs:
                mean_emb = np.mean(codon_embs, axis=0)
                embeddings.append(mean_emb.flatten())
            else:
                embeddings.append(np.zeros(16))

        embeddings = np.array(embeddings)

        # Compute pairwise hyperbolic distances
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                emb_i = torch.tensor(embeddings[i]).unsqueeze(0)
                emb_j = torch.tensor(embeddings[j]).unsqueeze(0)
                d = poincare_distance(emb_i, emb_j, c=1.0).item()
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix

    except Exception as e:
        print(f"Could not compute hyperbolic distances: {e}")
        return None


def analyze_projection(
    sequences: list[str],
    primes: list[int] = [2, 3, 5, 7, 11]
) -> ProjectionAnalysis:
    """Analyze if viral distances are a projection of p-adic distances."""

    n = len(sequences)
    print(f"Analyzing projection for {n} sequences...")

    # 1. Compute viral (Hamming) distance matrix
    print("  Computing viral distances...")
    viral_dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_sequence_distance(sequences[i], sequences[j])
            viral_dists[i, j] = d
            viral_dists[j, i] = d

    # 2. Compute p-adic distance matrices for each prime
    print("  Computing p-adic distances for each prime...")
    padic_dists = {}
    indices = [sequence_to_index(seq) for seq in sequences]

    for p in primes:
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = padic_distance(indices[i], indices[j], p)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        padic_dists[p] = dist_matrix

    # 3. Compute hyperbolic (3-adic trained) distances
    print("  Computing hyperbolic (TrainableCodonEncoder) distances...")
    hyp_dists = compute_hyperbolic_distances(sequences)

    # 4. Extract upper triangular (unique pairs)
    tri_indices = np.triu_indices(n, k=1)
    viral_flat = viral_dists[tri_indices]

    # 5. Spearman correlation for each prime (non-linear relationship test)
    print("  Testing non-linear relationships (Spearman)...")
    spearman_results = {}
    for p in primes:
        padic_flat = padic_dists[p][tri_indices]
        rho, pval = stats.spearmanr(viral_flat, padic_flat)
        spearman_results[p] = (float(rho) if not np.isnan(rho) else 0.0,
                               float(pval) if not np.isnan(pval) else 1.0)
        print(f"    {p}-adic: ρ = {rho:.4f}, p = {pval:.4e}")

    # Add hyperbolic if available
    if hyp_dists is not None:
        hyp_flat = hyp_dists[tri_indices]
        rho, pval = stats.spearmanr(viral_flat, hyp_flat)
        spearman_results['hyperbolic'] = (float(rho) if not np.isnan(rho) else 0.0,
                                          float(pval) if not np.isnan(pval) else 1.0)
        print(f"    hyperbolic: ρ = {rho:.4f}, p = {pval:.4e}")

    # 6. Linear projection: viral ≈ Σ(a_p * d_p) + b
    print("  Fitting linear projection model...")
    X = np.column_stack([padic_dists[p][tri_indices] for p in primes])
    if hyp_dists is not None:
        X = np.column_stack([X, hyp_dists[tri_indices]])
        feature_names = [f"{p}-adic" for p in primes] + ['hyperbolic']
    else:
        feature_names = [f"{p}-adic" for p in primes]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge regression (regularized to handle collinearity)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, viral_flat)

    predictions = model.predict(X_scaled)
    r2 = r2_score(viral_flat, predictions)
    residuals = viral_flat - predictions
    residual_std = float(np.std(residuals))

    print(f"    R² = {r2:.4f}")
    print(f"    Residual std = {residual_std:.4f}")

    # Coefficients (in standardized space)
    coefficients = {name: float(coef) for name, coef in zip(feature_names, model.coef_)}
    print("    Coefficients (standardized):")
    for name, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"      {name}: {coef:.4f}")

    # 7. Test residual structure
    print("  Analyzing residual structure...")

    # Reconstruct residual distance matrix
    residual_matrix = np.zeros((n, n))
    residual_matrix[tri_indices] = residuals
    residual_matrix = residual_matrix + residual_matrix.T

    # Test if residuals are ultrametric (hidden structure?)
    n_tested = 0
    n_violations = 0
    np.random.seed(42)

    for _ in range(min(5000, n * (n-1) * (n-2) // 6)):
        i, j, k = np.random.choice(n, 3, replace=False)
        d_ij = abs(residual_matrix[i, j])
        d_jk = abs(residual_matrix[j, k])
        d_ik = abs(residual_matrix[i, k])

        max_side = max(d_ij, d_jk)
        if d_ik > max_side + 1e-10:
            n_violations += 1
        n_tested += 1

    residual_compliance = 1.0 - (n_violations / n_tested) if n_tested > 0 else 1.0
    print(f"    Residual ultrametric compliance: {residual_compliance:.4f}")

    # Best non-linear prime
    best_prime = max(primes, key=lambda p: abs(spearman_results[p][0]))
    best_rho = spearman_results[best_prime][0]

    return ProjectionAnalysis(
        coefficients=coefficients,
        intercept=float(model.intercept_),
        r2_score=r2,
        residual_std=residual_std,
        spearman_by_prime={str(k): v for k, v in spearman_results.items()},
        best_nonlinear_prime=best_prime,
        best_nonlinear_rho=best_rho,
        residual_ultrametric_compliance=residual_compliance,
        residual_has_structure=residual_compliance > 0.90
    )


def load_sequences() -> list[str]:
    """Load DENV-4 sequence windows."""
    genome_file = _package_root / "results" / "ml_ready" / "denv4_genome_sequences.json"

    if not genome_file.exists():
        print("Using synthetic sequences for testing")
        np.random.seed(42)
        return [''.join(np.random.choice(['A', 'T', 'G', 'C'], 75)) for _ in range(100)]

    with open(genome_file) as f:
        data = json.load(f)

    sequences = []
    window_size = 75

    # Handle schema: data['data'][accession] = sequence_string
    if isinstance(data, dict) and 'data' in data:
        genomes = list(data['data'].values())
    elif isinstance(data, dict):
        genomes = list(data.values())
    else:
        genomes = data

    print(f"Found {len(genomes)} genomes")

    for genome_seq in genomes[:30]:  # Limit for computation
        if isinstance(genome_seq, dict):
            seq = genome_seq.get('sequence', '')
        else:
            seq = str(genome_seq)

        # Extract windows
        for pos in range(0, min(len(seq) - window_size, 2000), 200):
            window = seq[pos:pos + window_size]
            if len(window) == window_size and all(b in 'ATGCU' for b in window.upper()):
                sequences.append(window.upper().replace('U', 'T'))

    return sequences[:150]  # Limit for matrix computation


def main():
    """Run projection/deformation analysis."""

    print("=" * 70)
    print("PROJECTION/DEFORMATION ANALYSIS")
    print("Is viral evolutionary distance a projection of p-adic distances?")
    print("=" * 70)

    # Load sequences
    print("\nLoading sequences...")
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequence windows")

    # Run analysis
    print("\n" + "-" * 70)
    result = analyze_projection(sequences)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"\n1. LINEAR PROJECTION MODEL (R² = {result.r2_score:.4f})")
    if result.r2_score > 0.5:
        print("   → STRONG: Viral distance IS substantially explained by p-adic distances")
        print("   → Multiple p-adic structures contribute (mixed projection)")
    elif result.r2_score > 0.2:
        print("   → MODERATE: Some relationship exists but significant unexplained variance")
        print("   → p-adic structure captures PART of viral evolution")
    else:
        print("   → WEAK: p-adic distances do NOT explain viral evolutionary distance")
        print("   → Viral space geometry is fundamentally different from p-adic")

    print(f"\n2. NON-LINEAR RELATIONSHIPS (best: {result.best_nonlinear_prime}-adic, ρ = {result.best_nonlinear_rho:.4f})")
    if abs(result.best_nonlinear_rho) > 0.5:
        print(f"   → STRONG monotonic relationship with {result.best_nonlinear_prime}-adic distance")
        print("   → Consider non-linear transformation for output module")
    elif abs(result.best_nonlinear_rho) > 0.2:
        print(f"   → MODERATE monotonic trend with {result.best_nonlinear_prime}-adic")
    else:
        print("   → WEAK: No strong monotonic relationship for any prime")

    print(f"\n3. RESIDUAL STRUCTURE (ultrametric compliance = {result.residual_ultrametric_compliance:.4f})")
    if result.residual_has_structure:
        print("   → Residuals ARE ultrametric - hidden p-adic structure not captured!")
        print("   → May need higher primes or alternative encoding")
    else:
        print("   → Residuals are NOT ultrametric - no hidden p-adic structure")
        print("   → Unexplained variance is genuinely non-p-adic")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION FOR OUTPUT MODULE")
    print("=" * 70)

    if result.r2_score > 0.3 or abs(result.best_nonlinear_rho) > 0.3:
        print("""
An output module CAN be designed to adjust projections:

    adjusted_distance = f(hyperbolic_distance, prime_distances)

where f combines:
""")
        for name, coef in sorted(result.coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
            print(f"    - {name}: weight = {coef:.3f}")

        print("""
Implementation approach:
    1. Train a small MLP: [hyp_dist, d_2, d_3, d_5, d_7] → viral_dist_pred
    2. Or use weighted combination with learned coefficients
    3. Validate on held-out sequence pairs
""")
    else:
        print("""
WARNING: Statistics do NOT support a p-adic projection model.

The viral combinatorial space appears to operate under a geometry that is:
    - NOT ultrametric for any tested prime
    - NOT linearly related to 3-adic codon embeddings
    - NOT monotonically related to any p-adic distance

HONEST CONCLUSION:
    The TrainableCodonEncoder learns GENETIC CODE GRAMMAR (universal)
    but viral evolution operates in a DIFFERENT geometric space.

    An "output module" that claims to adjust p-adic projections for viral
    conservation would be MISLEADING - the mathematics doesn't support it.

ALTERNATIVE APPROACHES:
    1. Train encoder directly on viral sequences (abandon codon grammar)
    2. Use ensemble: p-adic for codon effects + separate model for evolution
    3. Acknowledge orthogonality as a FEATURE: two independent signals
""")

    # Save results
    output_dir = _script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'n_sequences': len(sequences),
        'linear_projection': {
            'r2_score': result.r2_score,
            'residual_std': result.residual_std,
            'intercept': result.intercept,
            'coefficients': result.coefficients
        },
        'nonlinear_analysis': {
            'spearman_by_prime': result.spearman_by_prime,
            'best_prime': result.best_nonlinear_prime,
            'best_rho': result.best_nonlinear_rho
        },
        'residual_analysis': {
            'ultrametric_compliance': result.residual_ultrametric_compliance,
            'has_hidden_structure': result.residual_has_structure
        },
        'recommendation': 'output_module_viable' if (result.r2_score > 0.3 or abs(result.best_nonlinear_rho) > 0.3) else 'no_padic_projection'
    }

    output_file = output_dir / "projection_deformation_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return result


if __name__ == "__main__":
    main()
