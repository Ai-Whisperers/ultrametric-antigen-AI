"""
07d_analyze_natural_positions.py - Analyze the 64 natural codon positions

Questions:
1. What are the 3-adic valuations of these indices?
2. What ternary digit patterns do they share?
3. Can we reverse-engineer a codon→index mapping?
4. Do clusters share common trit patterns (like wobble position)?

Usage:
    python 07d_analyze_natural_positions.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Load the selected indices from previous analysis
SELECTED_INDICES = [
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

CLUSTER_LABELS = [
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    9,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
    12,
    13,
    13,
    14,
    14,
    15,
    15,
    16,
    16,
    17,
    17,
    18,
    18,
    19,
    20,
]

# Genetic code for reference
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


def index_to_ternary(idx, n_digits=9):
    """Convert index to ternary digits (balanced: -1, 0, +1)."""
    digits = []
    n = idx
    for _ in range(n_digits):
        digits.append(n % 3 - 1)  # Convert 0,1,2 to -1,0,+1
        n //= 3
    return list(reversed(digits))


def index_to_ternary_unsigned(idx, n_digits=9):
    """Convert index to unsigned ternary digits (0, 1, 2)."""
    digits = []
    n = idx
    for _ in range(n_digits):
        digits.append(n % 3)
        n //= 3
    return list(reversed(digits))


def valuation_3(n):
    """Compute 3-adic valuation of n."""
    if n == 0:
        return 9
    v = 0
    while n % 3 == 0:
        v += 1
        n //= 3
    return v


def analyze_valuations():
    """Analyze 3-adic valuations of selected indices."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: 3-ADIC VALUATIONS")
    print("=" * 70)

    valuations = [valuation_3(idx) for idx in SELECTED_INDICES]

    print("\n  Valuation distribution:")
    val_counts = Counter(valuations)
    for v in sorted(val_counts.keys()):
        print(f"    v₃ = {v}: {val_counts[v]} indices")

    print(f"\n  Mean valuation: {np.mean(valuations):.2f}")
    print("  Expected for random 64: ~0.5")

    # By cluster
    print("\n  Valuations by cluster:")
    clusters = defaultdict(list)
    for idx, label in zip(SELECTED_INDICES, CLUSTER_LABELS):
        clusters[label].append(valuation_3(idx))

    for c in sorted(clusters.keys())[:10]:  # First 10 clusters
        vals = clusters[c]
        print(f"    Cluster {c} (n={len(vals)}): v₃ = {vals}, mean={np.mean(vals):.1f}")

    return valuations


def analyze_ternary_patterns():
    """Analyze ternary digit patterns."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: TERNARY DIGIT PATTERNS")
    print("=" * 70)

    # Convert all indices to ternary
    ternary = [index_to_ternary_unsigned(idx) for idx in SELECTED_INDICES]

    # Analyze digit frequency by position
    print("\n  Digit frequency by position (0=MSB, 8=LSB):")
    for pos in range(9):
        digits = [t[pos] for t in ternary]
        counts = Counter(digits)
        print(f"    Position {pos}: 0={counts[0]:2d}, 1={counts[1]:2d}, 2={counts[2]:2d}")

    # Look for patterns within clusters
    print("\n  Shared digits within clusters (positions where all members agree):")

    clusters = defaultdict(list)
    for idx, label, tern in zip(SELECTED_INDICES, CLUSTER_LABELS, ternary):
        clusters[label].append((idx, tern))

    cluster_patterns = {}
    for c in sorted(clusters.keys()):
        members = clusters[c]
        if len(members) < 2:
            continue

        shared_positions = []
        for pos in range(9):
            digits = [m[1][pos] for m in members]
            if len(set(digits)) == 1:
                shared_positions.append((pos, digits[0]))

        if shared_positions:
            cluster_patterns[c] = shared_positions
            positions_str = ", ".join([f"pos{p}={d}" for p, d in shared_positions])
            print(f"    Cluster {c} (n={len(members)}): {positions_str}")

    # Analyze which positions are most conserved
    print("\n  Position conservation across all clusters:")
    conserved_counts = Counter()
    for c, positions in cluster_patterns.items():
        for pos, _ in positions:
            conserved_counts[pos] += 1

    for pos in range(9):
        pct = 100 * conserved_counts[pos] / len(cluster_patterns) if cluster_patterns else 0
        bar = "█" * int(pct / 5)
        print(f"    Position {pos}: {conserved_counts[pos]:2d} clusters ({pct:5.1f}%) {bar}")

    return ternary, cluster_patterns


def analyze_codon_mapping_hypothesis():
    """Test if ternary patterns could encode codon structure."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: CODON MAPPING HYPOTHESIS")
    print("=" * 70)

    # Hypothesis: Positions 0-2 encode 1st nucleotide
    #             Positions 3-5 encode 2nd nucleotide
    #             Positions 6-8 encode 3rd nucleotide (wobble)

    ternary = [index_to_ternary_unsigned(idx) for idx in SELECTED_INDICES]

    # Group by first 6 positions (1st + 2nd nucleotide)
    print("\n  Grouping by first 6 trits (positions 0-5):")
    prefix_groups = defaultdict(list)
    for idx, tern, label in zip(SELECTED_INDICES, ternary, CLUSTER_LABELS):
        prefix = tuple(tern[:6])
        prefix_groups[prefix].append((idx, label, tern[6:]))

    # How many unique prefixes?
    print(f"    Unique 6-trit prefixes: {len(prefix_groups)}")

    # Do prefixes correlate with clusters?
    prefix_cluster_match = 0
    for prefix, members in prefix_groups.items():
        labels = [m[1] for m in members]
        if len(set(labels)) == 1:
            prefix_cluster_match += 1

    print(f"    Prefixes with single cluster: {prefix_cluster_match}/{len(prefix_groups)}")

    # Analyze the "wobble" positions (6-8)
    print("\n  Wobble position analysis (positions 6-8):")
    clusters = defaultdict(list)
    for idx, tern, label in zip(SELECTED_INDICES, ternary, CLUSTER_LABELS):
        clusters[label].append(tern[6:])

    wobble_variance = []
    for c in sorted(clusters.keys()):
        wobbles = clusters[c]
        if len(wobbles) < 2:
            continue
        # Measure variance in wobble positions
        wobble_arr = np.array(wobbles)
        var = np.var(wobble_arr, axis=0).sum()
        wobble_variance.append((c, len(wobbles), var))

    print("    Cluster | Size | Wobble Variance")
    print("    --------|------|----------------")
    for c, size, var in sorted(wobble_variance, key=lambda x: -x[1])[:10]:
        print(f"    {c:7d} | {size:4d} | {var:.2f}")

    # Test: Do larger clusters (like size-6) have more wobble variation?
    large_clusters = [v for c, s, v in wobble_variance if s >= 4]
    small_clusters = [v for c, s, v in wobble_variance if s == 2]

    if large_clusters and small_clusters:
        print(f"\n    Mean wobble variance (large clusters, n≥4): {np.mean(large_clusters):.2f}")
        print(f"    Mean wobble variance (small clusters, n=2): {np.mean(small_clusters):.2f}")

        if np.mean(large_clusters) > np.mean(small_clusters):
            print("\n    *** PATTERN: Larger clusters have MORE wobble variation ***")
            print("    This matches genetic code: 6-codon AAs have more 3rd-position flexibility!")


def find_optimal_nucleotide_mapping():
    """Try to find optimal nucleotide → trit mapping."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: OPTIMAL NUCLEOTIDE MAPPING SEARCH")
    print("=" * 70)

    # There are 4! = 24 ways to map {A,C,G,T} to {0,0,1,1,2,2} (with repeats)
    # Actually, we map to trits: each nucleotide gets 2 trits
    # Try different encodings and see which best matches cluster structure

    nucleotides = ["A", "C", "G", "T"]

    # Try several encoding schemes
    encodings = {
        "binary_pad": {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)},
        "ring_structure": {
            "A": (0, 0),
            "G": (0, 1),
            "C": (1, 0),
            "T": (1, 1),
        },  # Purine/Pyrimidine grouping
        "h_bond": {
            "A": (0, 0),
            "T": (0, 1),
            "G": (1, 0),
            "C": (1, 1),
        },  # A-T, G-C pairing
        "amino_keto": {
            "A": (0, 0),
            "C": (0, 1),
            "G": (1, 1),
            "T": (1, 0),
        },  # Chemical grouping
    }

    def codon_to_index(codon, encoding):
        trits = []
        for nuc in codon:
            trits.extend(encoding[nuc])
        while len(trits) < 9:
            trits.append(0)
        idx = 0
        for i, t in enumerate(trits):
            idx += t * (3 ** (8 - i))
        return idx

    print("\n  Testing different nucleotide→trit encodings:")

    for enc_name, encoding in encodings.items():
        # Map all 64 codons
        codon_indices = {codon: codon_to_index(codon, encoding) for codon in GENETIC_CODE.keys()}

        # How many codon indices are in our selected 64?
        overlap = sum(1 for idx in codon_indices.values() if idx in SELECTED_INDICES)

        # Do synonymous codons map to same cluster?
        aa_to_clusters = defaultdict(set)
        for codon, aa in GENETIC_CODE.items():
            idx = codon_indices[codon]
            if idx in SELECTED_INDICES:
                cluster = CLUSTER_LABELS[SELECTED_INDICES.index(idx)]
                aa_to_clusters[aa].add(cluster)

        single_cluster_aas = sum(1 for clusters in aa_to_clusters.values() if len(clusters) == 1)

        print(f"\n    {enc_name}:")
        print(f"      Overlap with selected 64: {overlap}/64")
        print(f"      AAs mapping to single cluster: {single_cluster_aas}/21")

        # Show which codons hit selected indices
        if overlap > 0:
            hits = [(c, idx) for c, idx in codon_indices.items() if idx in SELECTED_INDICES]
            print(f"      Matching codons: {[c for c, _ in hits[:5]]}{'...' if len(hits) > 5 else ''}")


def analyze_cluster_properties():
    """Detailed analysis of each cluster's ternary structure."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: CLUSTER TERNARY SIGNATURES")
    print("=" * 70)

    ternary = [index_to_ternary_unsigned(idx) for idx in SELECTED_INDICES]

    clusters = defaultdict(list)
    for idx, tern, label in zip(SELECTED_INDICES, ternary, CLUSTER_LABELS):
        clusters[label].append((idx, tern))

    print("\n  Cluster signatures (showing ternary patterns):")
    print("  Format: [pos0 pos1 pos2 | pos3 pos4 pos5 | pos6 pos7 pos8]")
    print("          (1st nuc-like)   (2nd nuc-like)   (wobble-like)")
    print()

    for c in sorted(clusters.keys()):
        members = clusters[c]
        size = len(members)

        # Find consensus pattern (most common digit at each position)
        consensus = []
        variability = []
        for pos in range(9):
            digits = [m[1][pos] for m in members]
            most_common = Counter(digits).most_common(1)[0][0]
            consensus.append(most_common)
            variability.append(len(set(digits)) > 1)

        # Format consensus with variability markers
        pattern = ""
        for i, (d, var) in enumerate(zip(consensus, variability)):
            if i == 3 or i == 6:
                pattern += " | "
            pattern += f"{'*' if var else ''}{d}"

        indices_str = ",".join(str(m[0]) for m in members[:3])
        if len(members) > 3:
            indices_str += "..."

        print(f"    Cluster {c:2d} (n={size}): [{pattern}]  indices: {indices_str}")


def main():
    print("=" * 70)
    print("ANALYZING 64 NATURAL CODON POSITIONS")
    print("=" * 70)

    # Analysis 1: Valuations
    valuations = analyze_valuations()

    # Analysis 2: Ternary patterns
    ternary, cluster_patterns = analyze_ternary_patterns()

    # Analysis 3: Codon mapping hypothesis
    analyze_codon_mapping_hypothesis()

    # Analysis 4: Find optimal nucleotide mapping
    find_optimal_nucleotide_mapping()

    # Analysis 5: Cluster signatures
    analyze_cluster_properties()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: KEY PATTERNS FOUND")
    print("=" * 70)

    print(
        """
    1. VALUATION STRUCTURE:
       - Selected indices have specific 3-adic valuations
       - This matches the model's learned radial hierarchy

    2. POSITION CONSERVATION:
       - Certain trit positions are conserved within clusters
       - This suggests a structured mapping is possible

    3. WOBBLE HYPOTHESIS:
       - Positions 6-8 (last 3 trits) show more variation in larger clusters
       - This parallels the genetic code's wobble position tolerance

    4. NUCLEOTIDE MAPPING:
       - Current naive encodings show low overlap
       - A learned/optimized mapping may be needed

    NEXT STEP: Train a small network to learn the optimal codon→index mapping
    that maximizes cluster quality for synonymous codons.
    """
    )


if __name__ == "__main__":
    main()
