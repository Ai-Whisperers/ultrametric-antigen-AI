"""
09_binary_ternary_decomposition.py - Investigate the 2×3 structure in the 1/6 exponent

Key insight: The radial exponent c = 1/6 = 1/(2×3) suggests the embedding
may be capturing BOTH binary (p=2) and ternary (p=3) structure simultaneously.

Hypothesis: The 6 "effective dimensions" per valuation level decompose as:
- 2 dimensions for binary bifurcations (even/odd structure)
- 3 dimensions for ternary refinement (mod 3 structure)

This would mean multi-prime structure is IMPLICITLY encoded, even though
we only trained on 3-adic loss.

Tests:
1. Check if embedding dimensions separate into 2×3 = 6 factor structure
2. Test if even/odd (mod 2) creates systematic structure within valuation levels
3. Analyze if the radial decay combines 2-adic and 3-adic contributions
4. Test prediction: exponent might be related to lcm(2,3)/something

Usage:
    python 09_binary_ternary_decomposition.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA


def v_p(n: int, p: int) -> int:
    """Compute p-adic valuation."""
    if n == 0:
        return float("inf")
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def analyze_dimension_factorization(embeddings):
    """Test if the 16 embedding dimensions factor as 2×3 structure."""
    print("\n" + "=" * 60)
    print("DIMENSION FACTORIZATION ANALYSIS")
    print("=" * 60)

    z_B = embeddings["z_B"]
    n_ops = len(z_B)

    # Group by v_3 valuation
    v3_vals = np.array([v_p(i, 3) if i > 0 else 9 for i in range(n_ops)])

    results = {}

    # For each valuation level, analyze the within-level structure
    for v3 in range(10):
        mask = v3_vals == v3
        if mask.sum() < 10:
            continue

        z_level = z_B[mask]
        indices = np.where(mask)[0]

        # Within this v3 level, check mod-2 structure
        mod2 = indices % 2

        # PCA on this level
        pca = PCA(n_components=min(6, len(z_level) - 1))
        z_pca = pca.fit_transform(z_level)

        # How much variance is explained by first 2 vs first 3 components?
        var_2 = pca.explained_variance_ratio_[:2].sum() if len(pca.explained_variance_ratio_) >= 2 else 0
        var_3 = pca.explained_variance_ratio_[:3].sum() if len(pca.explained_variance_ratio_) >= 3 else 0
        var_6 = pca.explained_variance_ratio_[:6].sum() if len(pca.explained_variance_ratio_) >= 6 else pca.explained_variance_ratio_.sum()

        # Test if PC1 or PC2 correlates with mod-2 structure
        if len(z_pca) > 10:
            corr_pc1_mod2, p1 = stats.pointbiserialr(mod2, z_pca[:, 0])
            corr_pc2_mod2, p2 = stats.pointbiserialr(mod2, z_pca[:, 1]) if z_pca.shape[1] > 1 else (0, 1)
        else:
            corr_pc1_mod2, p1, corr_pc2_mod2, p2 = 0, 1, 0, 1

        results[f"v3={v3}"] = {
            "n_points": int(mask.sum()),
            "var_explained_2d": float(var_2),
            "var_explained_3d": float(var_3),
            "var_explained_6d": float(var_6),
            "pc1_mod2_corr": float(corr_pc1_mod2),
            "pc1_mod2_pval": float(p1),
        }

        sig = "***" if p1 < 0.001 else "**" if p1 < 0.01 else "*" if p1 < 0.05 else ""
        print(f"\n  v₃={v3} (n={mask.sum()}):")
        print(f"    Variance: 2D={var_2:.1%}, 3D={var_3:.1%}, 6D={var_6:.1%}")
        print(f"    PC1 ↔ mod2: r={corr_pc1_mod2:.3f} {sig}")

    return results


def analyze_joint_valuation_structure(embeddings):
    """Test if radius encodes v_2 × v_3 joint structure."""
    print("\n" + "=" * 60)
    print("JOINT v₂ × v₃ STRUCTURE")
    print("=" * 60)

    z_B = embeddings["z_B"]
    radii = np.linalg.norm(z_B, axis=1)
    n_ops = len(z_B)

    # Compute both valuations
    v2_vals = np.array([v_p(i, 2) if i > 0 else 0 for i in range(n_ops)])
    v3_vals = np.array([v_p(i, 3) if i > 0 else 9 for i in range(n_ops)])

    # Multiple regression: radius ~ v2 + v3 + v2*v3
    from sklearn.linear_model import LinearRegression

    # Feature matrix: [v2, v3, v2*v3]
    X = np.column_stack([v2_vals, v3_vals, v2_vals * v3_vals])
    y = radii

    reg = LinearRegression()
    reg.fit(X, y)

    # Individual contributions
    print(f"\n  Linear model: radius = {reg.intercept_:.4f} + {reg.coef_[0]:.4f}×v₂ + {reg.coef_[1]:.4f}×v₃ + {reg.coef_[2]:.4f}×v₂v₃")
    print(f"  R² = {reg.score(X, y):.4f}")

    # Compare to v3-only model
    reg_v3 = LinearRegression()
    reg_v3.fit(v3_vals.reshape(-1, 1), y)
    r2_v3_only = reg_v3.score(v3_vals.reshape(-1, 1), y)
    print(f"  R² (v₃ only) = {r2_v3_only:.4f}")

    improvement = (reg.score(X, y) - r2_v3_only) / r2_v3_only * 100
    print(f"  Improvement from adding v₂: {improvement:.2f}%")

    # Test multiplicative model: log(radius) ~ v2*log(2) + v3*log(3)
    log_radii = np.log(radii + 1e-10)
    X_mult = np.column_stack([v2_vals, v3_vals])

    reg_mult = LinearRegression()
    reg_mult.fit(X_mult, log_radii)

    print(f"\n  Multiplicative model: r = exp({reg_mult.intercept_:.4f}) × 2^({reg_mult.coef_[0]:.4f}×v₂) × 3^({reg_mult.coef_[1]:.4f}×v₃)")
    print(f"  Predicted: 2-exponent = {reg_mult.coef_[0]/np.log(2):.4f}, 3-exponent = {reg_mult.coef_[1]/np.log(3):.4f}")

    # The key question: does the 3-exponent ≈ 1/6 and is there a 2-exponent?
    exp_2 = reg_mult.coef_[0] / np.log(2)
    exp_3 = reg_mult.coef_[1] / np.log(3)

    print("\n  KEY TEST: Is exponent ≈ 1/(2×3) = 1/6?")
    print(f"    3-adic exponent: {exp_3:.4f} (expected: -0.167 = -1/6)")
    print(f"    2-adic exponent: {exp_2:.4f} (expected: ~0 if not trained on v₂)")

    return {
        "linear_coef_v2": float(reg.coef_[0]),
        "linear_coef_v3": float(reg.coef_[1]),
        "linear_coef_v2v3": float(reg.coef_[2]),
        "r2_full": float(reg.score(X, y)),
        "r2_v3_only": float(r2_v3_only),
        "mult_exp_2": float(exp_2),
        "mult_exp_3": float(exp_3),
    }


def analyze_within_level_binary_structure(embeddings):
    """Within each v₃ level, test for binary (mod 2) organization."""
    print("\n" + "=" * 60)
    print("WITHIN-LEVEL BINARY STRUCTURE")
    print("=" * 60)

    z_B = embeddings["z_B"]
    radii = np.linalg.norm(z_B, axis=1)
    n_ops = len(z_B)

    v3_vals = np.array([v_p(i, 3) if i > 0 else 9 for i in range(n_ops)])

    results = {}

    print("\n  Testing if even/odd creates systematic structure within v₃ levels:")

    for v3 in range(8):  # v3 = 0 to 7
        mask = v3_vals == v3
        if mask.sum() < 20:
            continue

        indices = np.where(mask)[0]
        z_level = z_B[mask]
        r_level = radii[mask]

        # Split by even/odd within the level
        even_within = indices % 2 == 0
        odd_within = indices % 2 == 1

        r_even = r_level[even_within]
        r_odd = r_level[odd_within]

        if len(r_even) > 5 and len(r_odd) > 5:
            # T-test for radius difference
            t_stat, p_val = stats.ttest_ind(r_even, r_odd)

            # Effect size
            diff = r_even.mean() - r_odd.mean()

            results[f"v3={v3}"] = {
                "n_even": len(r_even),
                "n_odd": len(r_odd),
                "r_even_mean": float(r_even.mean()),
                "r_odd_mean": float(r_odd.mean()),
                "difference": float(diff),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            }

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"    v₃={v3}: r_even={r_even.mean():.4f}, r_odd={r_odd.mean():.4f}, diff={diff:.4f} {sig}")

    return results


def test_six_equals_two_times_three(embeddings):
    """Test the hypothesis that 6 = 2×3 represents factored prime structure."""
    print("\n" + "=" * 60)
    print("TESTING 6 = 2×3 HYPOTHESIS")
    print("=" * 60)

    z_B = embeddings["z_B"]
    n_ops = len(z_B)

    # For each integer, compute joint (v2, v3) classification
    v2_vals = np.array([v_p(i, 2) if i > 0 else 0 for i in range(n_ops)])
    v3_vals = np.array([v_p(i, 3) if i > 0 else 9 for i in range(n_ops)])

    # Create 2×3 = 6 classes based on (v2 mod 2, v3 mod 3)
    # This gives 6 distinct "prime residue" classes
    class_6 = (v2_vals % 2) * 3 + (v3_vals % 3)

    # PCA to 6 dimensions
    pca = PCA(n_components=6)
    z_pca = pca.fit_transform(z_B)

    print(f"\n  6D PCA explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance")

    # Test if each PC correlates with one of the 6 classes
    print("\n  PC correlations with 6-class structure:")

    for pc in range(6):
        # One-way ANOVA: does this PC differ across the 6 classes?
        groups = [z_pca[class_6 == c, pc] for c in range(6)]
        groups = [g for g in groups if len(g) > 5]

        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"    PC{pc+1}: F={f_stat:.1f}, p={p_val:.2e} {sig}")

    # Alternative: test if (v2 mod 2) and (v3 mod 3) independently predict different PCs
    print("\n  Separating binary and ternary contributions:")

    mod2 = v2_vals % 2
    mod3 = v3_vals % 3

    for pc in range(min(6, z_pca.shape[1])):
        corr2, p2 = stats.pointbiserialr(mod2, z_pca[:, pc])
        # For mod3 (3 classes), use ANOVA
        groups_3 = [z_pca[mod3 == m, pc] for m in range(3)]
        f3, p3 = stats.f_oneway(*groups_3)

        print(f"    PC{pc+1}: mod2 r={corr2:.3f} (p={p2:.2e}), mod3 F={f3:.1f} (p={p3:.2e})")

    return {"pca_variance": pca.explained_variance_ratio_.tolist()}


def analyze_radial_formula_decomposition(embeddings):
    """Decompose the radial formula to test 2×3 structure."""
    print("\n" + "=" * 60)
    print("RADIAL FORMULA DECOMPOSITION")
    print("=" * 60)

    z_B = embeddings["z_B"]
    radii = np.linalg.norm(z_B, axis=1)
    n_ops = len(z_B)

    v2_vals = np.array([v_p(i, 2) if i > 0 else 0 for i in range(n_ops)])
    v3_vals = np.array([v_p(i, 3) if i > 0 else 9 for i in range(n_ops)])

    # The discovered formula: r(v) = 0.929 × 3^(-0.172v)
    # where 0.172 ≈ 1/6 = 1/(2×3)

    # Test alternative formulations:

    # 1. r = a × 2^(-b×v2) × 3^(-c×v3)
    print("\n  Model 1: r = a × 2^(-b×v₂) × 3^(-c×v₃)")
    log_r = np.log(radii + 1e-10)

    # Fit: log(r) = log(a) - b×v2×log(2) - c×v3×log(3)
    X = np.column_stack([v2_vals, v3_vals])
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(X, log_r)

    a = np.exp(reg.intercept_)
    b = -reg.coef_[0] / np.log(2)
    c = -reg.coef_[1] / np.log(3)

    print(f"    a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
    print(f"    Interpretation: r = {a:.3f} × 2^(-{b:.4f}×v₂) × 3^(-{c:.4f}×v₃)")

    # 2. Test if c ≈ 1/6 and b ≈ 0
    print("\n  KEY FINDING:")
    print(f"    3-adic exponent c = {c:.4f}")
    print("    Expected if 6 = 2×3 structure: c ≈ 1/6 = 0.1667")
    print(f"    Difference: {abs(c - 1/6):.4f}")

    # 3. Test if there's hidden 2-adic contribution even though not trained
    print(f"\n    2-adic exponent b = {b:.4f}")
    print(f"    If model implicitly captures 2×3: expect b ≈ {1/6:.4f}")
    print("    If model is purely 3-adic: expect b ≈ 0")

    # 4. Alternative: r = a × 6^(-d×v6) where v6 = v2 + v3 (joint valuation?)
    v6_sum = v2_vals + v3_vals
    reg6 = LinearRegression()
    reg6.fit(v6_sum.reshape(-1, 1), log_r)
    d = -reg6.coef_[0] / np.log(6)

    print("\n  Model 2: r = a × 6^(-d×(v₂+v₃))")
    print(f"    d = {d:.4f}")
    print(f"    R² = {reg6.score(v6_sum.reshape(-1, 1), log_r):.4f}")

    return {
        "model1_a": float(a),
        "model1_b_2adic": float(b),
        "model1_c_3adic": float(c),
        "model2_d_6adic": float(d),
    }


def main():
    output_dir = PROJECT_ROOT / "research/spectral_analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    data = torch.load(
        PROJECT_ROOT / "research/spectral_analysis" / "embeddings" / "embeddings.pt",
        weights_only=False,
    )

    embeddings = {
        "z_B": (
            data.get("z_B_hyp", data.get("z_hyperbolic")).numpy()
            if torch.is_tensor(data.get("z_B_hyp", data.get("z_hyperbolic")))
            else data.get("z_B_hyp", data.get("z_hyperbolic"))
        ),
    }

    print(f"Loaded embeddings: shape = {embeddings['z_B'].shape}")

    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "analysis": "binary_ternary_2x3_decomposition",
    }

    # Analysis 1: Dimension factorization
    dim_results = analyze_dimension_factorization(embeddings)
    results["dimension_factorization"] = dim_results

    # Analysis 2: Joint v2 × v3 structure
    joint_results = analyze_joint_valuation_structure(embeddings)
    results["joint_valuation"] = joint_results

    # Analysis 3: Within-level binary structure
    binary_results = analyze_within_level_binary_structure(embeddings)
    results["within_level_binary"] = binary_results

    # Analysis 4: 6 = 2×3 hypothesis test
    six_results = test_six_equals_two_times_three(embeddings)
    results["six_hypothesis"] = six_results

    # Analysis 5: Radial formula decomposition
    formula_results = analyze_radial_formula_decomposition(embeddings)
    results["radial_decomposition"] = formula_results

    # Summary
    print("\n" + "=" * 60)
    print("2×3 DECOMPOSITION SUMMARY")
    print("=" * 60)

    print(
        f"""
THE 1/6 = 1/(2×3) HYPOTHESIS:

The radial exponent c ≈ 1/6 suggests the model allocates 6 "effective
dimensions" per valuation level, and 6 = 2×3 factors into:

  - 2 dimensions: binary bifurcation structure
  - 3 dimensions: ternary refinement structure

EVIDENCE FROM THIS ANALYSIS:

1. RADIAL FORMULA DECOMPOSITION:
   - 3-adic exponent: {formula_results['model1_c_3adic']:.4f} (expected: 0.167)
   - 2-adic exponent: {formula_results['model1_b_2adic']:.4f} (expected: ~0 if not trained)

2. JOINT STRUCTURE:
   - Adding v₂ to the model improves R² by {(joint_results['r2_full'] - joint_results['r2_v3_only'])/joint_results['r2_v3_only']*100:.1f}%

3. INTERPRETATION:
   The exponent 1/6 emerges from the architecture constraint:

     c = 1/(latent_dim - n_trits - 1) = 1/(16 - 9 - 1) = 1/6

   This can be read as: the model has 6 = 2×3 dimensions of "slack" per
   valuation level, naturally encoding both binary and ternary structure.

IMPLICATION FOR MULTI-PRIME:

   If 1/6 = 1/(2×3) encodes joint 2-adic and 3-adic structure implicitly,
   then multi-prime behavior might emerge from architectural constraints
   rather than explicit multi-prime training.

   PREDICTION: Training with latent_dim = 16 + k should give exponent
   1/(6+k), and the 2×3 factorization would shift to 2×3×... structure
   if k is chosen appropriately.
"""
    )

    # Save results
    results_file = output_dir / "binary_ternary_decomposition.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(results_file, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\nSaved results to {results_file}")

    return results


if __name__ == "__main__":
    main()
