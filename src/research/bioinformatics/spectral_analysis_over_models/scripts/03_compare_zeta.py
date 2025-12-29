"""
03_compare_zeta.py - Compare eigenvalue spectrum to Riemann zeta zeros

This script downloads/loads known Riemann zeta zeros and compares them
to the eigenvalue spectrum of our hyperbolic Laplacian.

Usage:
    python 03_compare_zeta.py [--results-dir PATH]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# First 100 imaginary parts of Riemann zeta zeros (high precision known values)
# Source: LMFDB / Andrew Odlyzko's tables
ZETA_ZEROS_100 = np.array(
    [
        14.134725141734693790,
        21.022039638771554993,
        25.010857580145688763,
        30.424876125859513210,
        32.935061587739189691,
        37.586178158825671257,
        40.918719012147495187,
        43.327073280914999519,
        48.005150881167159727,
        49.773832477672302181,
        52.970321477714460644,
        56.446247697063394804,
        59.347044002602353079,
        60.831778524609809844,
        65.112544048081606660,
        67.079810529494173714,
        69.546401711173979252,
        72.067157674481907582,
        75.704690699083933168,
        77.144840068874805372,
        79.337375020249367922,
        82.910380854086030183,
        84.735492980517050105,
        87.425274613125229406,
        88.809111207634465423,
        92.491899270558484297,
        94.651344040519886966,
        95.870634228245309758,
        98.831194218193692234,
        101.31785100573139122,
        103.72553804047833941,
        105.44662305232609449,
        107.16861118427640751,
        111.02953554316967452,
        111.87465917699263708,
        114.32022091545271276,
        116.22668032085755438,
        118.79078286597621733,
        121.37012500242064591,
        122.94682929355258820,
        124.25681855434576718,
        127.51668387959649512,
        129.57870419995605098,
        131.08768853093265672,
        133.49773720299758646,
        134.75650975337387133,
        138.11604205453344320,
        139.73620895212138895,
        141.12370740402112376,
        143.11184580762063273,
        146.00098248680048918,
        147.42276534919928971,
        150.05352042078395757,
        150.92525761645737443,
        153.02469388478088856,
        156.11290929488542804,
        157.59759166389549466,
        158.84998819391903457,
        161.18896413047820800,
        163.03070968662348549,
        165.53706942685091296,
        167.18443992563737322,
        169.09451541574791865,
        169.91197647941169896,
        173.41153628461824611,
        174.75419152917556958,
        176.44143386338821657,
        178.37740777609098423,
        179.91648402025700898,
        182.20707848436646108,
        184.87446784742265304,
        185.59878367769787686,
        187.22892258142234542,
        189.41615865959812894,
        192.02665636781364659,
        193.07972660330851588,
        195.26539668143130628,
        196.87648174678348640,
        198.01530956325059576,
        201.26475194370446731,
        202.49359453678860297,
        204.18967180042754564,
        205.39469720928828530,
        207.90625889011885648,
        209.57650969649856361,
        211.69086259164145329,
        213.34791935629091578,
        214.54704478393905007,
        216.16953848274682713,
        219.06759635718080498,
        220.71491885794096567,
        221.43070548923461542,
        224.00700025328681230,
        224.98332466958052543,
        227.42144501747689228,
        229.33741330869986678,
        231.25018870043791927,
        231.98723519112696132,
        233.69340355929794530,
    ]
)


def download_zeta_zeros(n_zeros: int = 10000, output_dir: Path = None) -> np.ndarray:
    """Download first n Riemann zeta zeros from LMFDB or use cached.

    Returns imaginary parts of zeros (the real parts are all 1/2 assuming RH).
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "research/spectral_analysis" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / f"zeta_zeros_{n_zeros}.npy"

    if cache_file.exists():
        print(f"Loading cached zeta zeros from {cache_file}")
        return np.load(cache_file)

    # For now, use the hardcoded first 100 zeros
    # For more zeros, would need to download from LMFDB
    if n_zeros <= 100:
        zeros = ZETA_ZEROS_100[:n_zeros]
    else:
        print("Note: Only have 100 hardcoded zeros, returning those")
        print("For more zeros, download from https://www.lmfdb.org/zeros/zeta/")
        zeros = ZETA_ZEROS_100

    np.save(cache_file, zeros)
    return zeros


def compute_zeta_spacings(zeros: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute normalized spacings between consecutive zeta zeros."""
    spacings = np.diff(zeros)

    if normalize:
        # Unfolding: normalize by average local density
        # For zeta zeros, average spacing ~ 2*pi / log(t/(2*pi))
        t_avg = (zeros[:-1] + zeros[1:]) / 2
        expected_spacing = 2 * np.pi / np.log(t_avg / (2 * np.pi))
        spacings = spacings / expected_spacing

    return spacings


def compare_spectra(eigenvalues: np.ndarray, zeta_zeros: np.ndarray, output_dir: Path) -> dict:
    """Compare eigenvalue spectrum to zeta zeros."""

    # Normalize both spectra to same range
    eigen_norm = (eigenvalues - eigenvalues.min()) / (eigenvalues.max() - eigenvalues.min())
    zeta_norm = (zeta_zeros - zeta_zeros.min()) / (zeta_zeros.max() - zeta_zeros.min())

    # Compute spacings
    eigen_spacings = np.diff(eigen_norm)
    eigen_spacings = eigen_spacings / eigen_spacings.mean()  # Normalize to mean 1

    zeta_spacings = compute_zeta_spacings(zeta_zeros, normalize=True)

    # Truncate to same length
    n_compare = min(len(eigen_spacings), len(zeta_spacings))
    eigen_spacings = eigen_spacings[:n_compare]
    zeta_spacings = zeta_spacings[:n_compare]

    # Statistical comparisons
    results = {}

    # 1. Pearson correlation of spacings
    r_spacings, p_spacings = stats.pearsonr(eigen_spacings, zeta_spacings)
    results["pearson_spacings"] = {
        "r": float(r_spacings),
        "p": float(p_spacings),
    }

    # 2. Spearman rank correlation
    rho_spacings, p_spearman = stats.spearmanr(eigen_spacings, zeta_spacings)
    results["spearman_spacings"] = {
        "rho": float(rho_spacings),
        "p": float(p_spearman),
    }

    # 3. KS test on spacing distributions
    ks_stat, ks_pval = stats.ks_2samp(eigen_spacings, zeta_spacings)
    results["ks_test"] = {
        "statistic": float(ks_stat),
        "p_value": float(ks_pval),
    }

    # 4. Compare cumulative distributions
    eigen_cdf = np.sort(eigen_spacings)
    zeta_cdf = np.sort(zeta_spacings)

    # Interpolate to common grid
    common_grid = np.linspace(0, max(eigen_cdf.max(), zeta_cdf.max()), 100)
    eigen_cdf_interp = np.interp(
        common_grid,
        np.sort(eigen_spacings),
        np.arange(len(eigen_spacings)) / len(eigen_spacings),
    )
    zeta_cdf_interp = np.interp(
        common_grid,
        np.sort(zeta_spacings),
        np.arange(len(zeta_spacings)) / len(zeta_spacings),
    )

    cdf_correlation, _ = stats.pearsonr(eigen_cdf_interp, zeta_cdf_interp)
    results["cdf_correlation"] = float(cdf_correlation)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Spacing distributions
    ax1 = axes[0, 0]
    ax1.hist(
        eigen_spacings,
        bins=30,
        density=True,
        alpha=0.6,
        label="Laplacian eigenvalues",
        color="steelblue",
    )
    ax1.hist(
        zeta_spacings,
        bins=30,
        density=True,
        alpha=0.6,
        label="Zeta zeros",
        color="crimson",
    )
    ax1.set_xlabel("Normalized spacing")
    ax1.set_ylabel("Density")
    ax1.set_title("Spacing Distributions")
    ax1.legend()

    # 2. Spacing scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(eigen_spacings, zeta_spacings, alpha=0.5, s=10)
    ax2.plot(
        [0, max(eigen_spacings.max(), zeta_spacings.max())],
        [0, max(eigen_spacings.max(), zeta_spacings.max())],
        "r--",
        label="y=x",
    )
    ax2.set_xlabel("Laplacian spacings")
    ax2.set_ylabel("Zeta spacings")
    ax2.set_title(f"Spacing Correlation (r={r_spacings:.4f})")
    ax2.legend()

    # 3. CDFs
    ax3 = axes[1, 0]
    ax3.plot(
        common_grid,
        eigen_cdf_interp,
        label="Laplacian eigenvalues",
        linewidth=2,
    )
    ax3.plot(common_grid, zeta_cdf_interp, label="Zeta zeros", linewidth=2)
    ax3.set_xlabel("Spacing")
    ax3.set_ylabel("Cumulative probability")
    ax3.set_title(f"Cumulative Distributions (r={cdf_correlation:.4f})")
    ax3.legend()

    # 4. Q-Q plot (manual)
    ax4 = axes[1, 1]
    n_qq = min(len(eigen_spacings), len(zeta_spacings))
    eigen_sorted = np.sort(eigen_spacings)[:n_qq]
    zeta_sorted = np.sort(zeta_spacings)[:n_qq]
    ax4.scatter(zeta_sorted, eigen_sorted[: len(zeta_sorted)], alpha=0.5, s=10)
    max_val = max(zeta_sorted.max(), eigen_sorted.max())
    ax4.plot([0, max_val], [0, max_val], "r--", label="y=x")
    ax4.set_xlabel("Zeta zero spacings (sorted)")
    ax4.set_ylabel("Eigenvalue spacings (sorted)")
    ax4.set_title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig(output_dir / "zeta_comparison.png", dpi=150)
    plt.savefig(output_dir / "zeta_comparison.pdf")
    plt.close()

    print(f"Saved comparison plots to {output_dir}")

    return results


def compute_pair_correlation(spacings: np.ndarray, output_dir: Path = None) -> dict:
    """Compute pair correlation function R_2(x).

    The Montgomery conjecture (proven for RH zeros) predicts:
    R_2(x) = 1 - (sin(pi*x)/(pi*x))^2

    This is the same as GUE pair correlation.
    """
    # Compute pair correlation via histogram
    n = len(spacings)

    # Compute all pairwise differences
    all_diffs = []
    for i in range(n):
        for j in range(i + 1, min(i + 50, n)):  # Look at nearby pairs
            all_diffs.append(abs(spacings[i] - spacings[j]))

    all_diffs = np.array(all_diffs)

    # Normalize
    all_diffs = all_diffs / np.mean(spacings)

    # Histogram
    x_range = np.linspace(0, 4, 100)
    hist, bin_edges = np.histogram(all_diffs, bins=50, density=True, range=(0, 4))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Montgomery prediction
    x = np.linspace(0.01, 4, 100)
    montgomery = 1 - (np.sin(np.pi * x) / (np.pi * x)) ** 2

    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.bar(
            bin_centers,
            hist,
            width=bin_centers[1] - bin_centers[0],
            alpha=0.7,
            label="Observed",
            color="steelblue",
        )
        plt.plot(
            x,
            montgomery,
            "r-",
            linewidth=2,
            label="Montgomery prediction (GUE)",
        )
        plt.xlabel("Normalized spacing difference")
        plt.ylabel("Pair correlation R₂(x)")
        plt.title("Pair Correlation Function")
        plt.legend()
        plt.xlim(0, 4)
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "pair_correlation.png", dpi=150)
        plt.close()

    # Compute fit to Montgomery
    montgomery_at_bins = 1 - (np.sin(np.pi * bin_centers + 1e-10) / (np.pi * bin_centers + 1e-10)) ** 2
    mse = np.mean((hist - montgomery_at_bins) ** 2)

    return {
        "mse_to_montgomery": float(mse),
        "bin_centers": bin_centers.tolist(),
        "observed_correlation": hist.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare spectrum to zeta zeros")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="research/spectral_analysis/results",
        help="Directory with eigenvalues",
    )
    parser.add_argument(
        "--n-zeta-zeros",
        type=int,
        default=100,
        help="Number of zeta zeros to use",
    )
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    output_dir = results_dir

    # Load eigenvalues
    eigenvalues_file = results_dir / "eigenvalues.npy"
    if not eigenvalues_file.exists():
        print(f"Error: {eigenvalues_file} not found")
        print("Run 02_compute_spectrum.py first")
        return

    print(f"Loading eigenvalues from {eigenvalues_file}")
    eigenvalues = np.load(eigenvalues_file)
    print(f"Loaded {len(eigenvalues)} eigenvalues")

    # Load zeta zeros
    print(f"\nLoading first {args.n_zeta_zeros} Riemann zeta zeros...")
    zeta_zeros = download_zeta_zeros(args.n_zeta_zeros)
    print(f"Loaded {len(zeta_zeros)} zeta zeros")
    print(f"Range: [{zeta_zeros.min():.4f}, {zeta_zeros.max():.4f}]")

    # Compare spectra
    print("\nComparing spectra...")
    comparison_results = compare_spectra(eigenvalues, zeta_zeros, output_dir)

    print("\n=== Comparison Results ===")
    print(f"Pearson correlation (spacings): r = {comparison_results['pearson_spacings']['r']:.4f}")
    print(f"  p-value: {comparison_results['pearson_spacings']['p']:.4e}")
    print(f"Spearman correlation (spacings): rho = {comparison_results['spearman_spacings']['rho']:.4f}")
    print(f"  p-value: {comparison_results['spearman_spacings']['p']:.4e}")
    print(f"KS test: D = {comparison_results['ks_test']['statistic']:.4f}")
    print(f"  p-value: {comparison_results['ks_test']['p_value']:.4e}")
    print(f"CDF correlation: r = {comparison_results['cdf_correlation']:.4f}")

    # Compute pair correlation
    print("\nComputing pair correlation function...")
    spacings = np.load(results_dir / "spacings.npy")
    pair_corr_results = compute_pair_correlation(spacings, output_dir)
    comparison_results["pair_correlation"] = pair_corr_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"zeta_comparison_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nSaved comparison results to {results_file}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    r = comparison_results["pearson_spacings"]["r"]
    ks = comparison_results["ks_test"]["statistic"]

    if r > 0.7 and ks < 0.1:
        print("\n*** STRONG CORRELATION DETECTED ***")
        print("The eigenvalue spacings show strong correlation with zeta zeros!")
        print("This is potentially significant and warrants further investigation.")
        print("Consider:")
        print("  1. Increasing the number of eigenvalues computed")
        print("  2. Testing with different kernel bandwidths")
        print("  3. Formal mathematical analysis of the connection")
        print("  4. Consultation with number theorists")
    elif r > 0.3:
        print("\n** MODERATE CORRELATION DETECTED **")
        print("There is a notable correlation between eigenvalue spacings and zeta zeros.")
        print("This is suggestive but not conclusive.")
        print("Further analysis recommended.")
    else:
        print("\nNo strong correlation detected.")
        print("The eigenvalue spectrum does not closely match zeta zeros.")
        print("This could mean:")
        print("  1. The connection requires different mathematical formulation")
        print("  2. Different kernel/Laplacian construction needed")
        print("  3. The connection may not exist in this form")

    print("\n" + "=" * 60)

    return comparison_results


if __name__ == "__main__":
    main()
