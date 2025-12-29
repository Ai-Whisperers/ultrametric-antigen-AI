# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""
Structural Features Module

Implements structure-derived features for HIV sequence analysis.
Based on:
- Sander et al. (2007) - Structural descriptors of gp120 V3 loop
- Fouchier et al. (1992) - 11/25 rule for tropism
- Thielen et al. (2010) - V2 loop features

Features:
- Net charge calculation
- V3 crown/stem analysis
- V2 loop features
- Hydrophobicity profiles
- Secondary structure prediction
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ============================================================================
# AMINO ACID PROPERTIES
# ============================================================================

# Charge at physiological pH
AMINO_ACID_CHARGE = {
    "R": 1.0,   # Arginine - positive
    "K": 1.0,   # Lysine - positive
    "H": 0.5,   # Histidine - partially positive at pH 7
    "D": -1.0,  # Aspartic acid - negative
    "E": -1.0,  # Glutamic acid - negative
    "A": 0.0, "C": 0.0, "F": 0.0, "G": 0.0, "I": 0.0,
    "L": 0.0, "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0,
    "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
    "X": 0.0, "-": 0.0, "*": 0.0,
}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
    "X": 0.0, "-": 0.0, "*": 0.0,
}

# Molecular weight (Da)
MOLECULAR_WEIGHT = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121,
    "E": 147, "Q": 146, "G": 75, "H": 155, "I": 131,
    "L": 131, "K": 146, "M": 149, "F": 165, "P": 115,
    "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117,
    "X": 110, "-": 0, "*": 0,
}

# Polarity (Grantham)
POLARITY = {
    "L": 0.0, "I": 0.0, "F": 0.0, "W": 0.0, "C": 0.0,
    "M": 0.0, "V": 0.0, "Y": 1.0, "P": 0.0, "A": 0.0,
    "T": 1.0, "G": 0.0, "S": 1.0, "H": 1.0, "Q": 1.0,
    "N": 1.0, "E": 1.0, "D": 1.0, "K": 1.0, "R": 1.0,
    "X": 0.5, "-": 0.0, "*": 0.0,
}

# Volume (normalized)
VOLUME = {
    "G": 0.0, "A": 0.17, "S": 0.23, "C": 0.36, "D": 0.42,
    "P": 0.42, "N": 0.48, "T": 0.48, "E": 0.59, "V": 0.53,
    "Q": 0.65, "H": 0.66, "M": 0.68, "I": 0.68, "L": 0.68,
    "K": 0.71, "R": 0.88, "F": 0.76, "Y": 0.83, "W": 1.0,
    "X": 0.5, "-": 0.0, "*": 0.0,
}


# ============================================================================
# V3 LOOP ANALYSIS
# ============================================================================

# V3 loop regions (standard 35 amino acid length)
V3_REGIONS = {
    "stem_n": (0, 9),      # N-terminal stem
    "crown": (10, 24),      # Crown region (most variable)
    "stem_c": (25, 35),     # C-terminal stem
    "tip": (14, 18),        # Tip of crown (GPG motif area)
}

# Key positions for tropism (Fouchier 11/25 rule, 0-indexed as 10/24)
TROPISM_KEY_POSITIONS = {
    11: "classic_11",  # Position 11 (0-indexed: 10)
    25: "classic_25",  # Position 25 (0-indexed: 24)
    22: "novel_22",    # Our novel finding - position 22
}


def calculate_net_charge(sequence: str) -> float:
    """
    Calculate net charge of a peptide sequence.

    Based on Fouchier et al. (1992):
    - V3 net charge >= 5 often indicates X4 tropism
    - Basic amino acids (R, K) at positions 11 and 25 predict X4

    Args:
        sequence: Amino acid sequence

    Returns:
        float: Net charge at physiological pH
    """
    return sum(AMINO_ACID_CHARGE.get(aa.upper(), 0.0) for aa in sequence)


def calculate_hydrophobicity(sequence: str) -> dict:
    """
    Calculate hydrophobicity features of a sequence.

    Returns:
        dict: {
            'mean': float,
            'std': float,
            'max': float,
            'min': float,
            'profile': list[float]
        }
    """
    profile = [HYDROPHOBICITY.get(aa.upper(), 0.0) for aa in sequence]

    if not profile:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "profile": []}

    return {
        "mean": np.mean(profile),
        "std": np.std(profile),
        "max": np.max(profile),
        "min": np.min(profile),
        "profile": profile,
    }


def analyze_v3_structure(v3_sequence: str) -> dict:
    """
    Analyze V3 loop structural features.

    Based on Sander et al. (2007) and Cormier & Dragic (2002):
    - Crown and stem have different roles in tropism
    - Crown is highly variable, stem is conserved
    - Tip region contacts coreceptor

    Args:
        v3_sequence: V3 loop sequence (typically 35 amino acids)

    Returns:
        dict: Comprehensive structural analysis
    """
    seq = v3_sequence.upper()
    length = len(seq)

    # Adjust region boundaries for non-standard lengths
    scale = length / 35.0

    results = {
        "length": length,
        "total_charge": calculate_net_charge(seq),
        "regions": {},
    }

    # Analyze each region
    for region_name, (start, end) in V3_REGIONS.items():
        adj_start = int(start * scale)
        adj_end = min(int(end * scale), length)
        region_seq = seq[adj_start:adj_end]

        if region_seq:
            results["regions"][region_name] = {
                "sequence": region_seq,
                "length": len(region_seq),
                "charge": calculate_net_charge(region_seq),
                "hydrophobicity": calculate_hydrophobicity(region_seq)["mean"],
                "basic_count": sum(1 for aa in region_seq if aa in "RKH"),
                "acidic_count": sum(1 for aa in region_seq if aa in "DE"),
            }

    # Key position analysis
    results["key_positions"] = {}
    for pos, name in TROPISM_KEY_POSITIONS.items():
        idx = pos - 1  # Convert to 0-indexed
        if idx < length:
            aa = seq[idx]
            results["key_positions"][name] = {
                "position": pos,
                "amino_acid": aa,
                "charge": AMINO_ACID_CHARGE.get(aa, 0.0),
                "is_basic": aa in "RKH",
            }

    # 11/25 rule prediction
    pos11_basic = results["key_positions"].get("classic_11", {}).get("is_basic", False)
    pos25_basic = results["key_positions"].get("classic_25", {}).get("is_basic", False)
    results["rule_11_25_x4"] = pos11_basic or pos25_basic

    # Charge-based prediction
    results["charge_rule_x4"] = results["total_charge"] >= 5

    # Crown charge (most predictive region)
    crown_data = results["regions"].get("crown", {})
    results["crown_charge"] = crown_data.get("charge", 0.0)

    return results


def calculate_v3_features_for_ml(v3_sequence: str) -> np.ndarray:
    """
    Extract V3 features as a numpy array for ML models.

    Features (20-dimensional):
    - Total charge, crown charge, stem charges
    - Hydrophobicity: mean, std, crown mean
    - Key position charges (11, 22, 25)
    - Region-specific basic/acidic counts
    - Length, GPG motif present

    Args:
        v3_sequence: V3 loop sequence

    Returns:
        np.ndarray: Feature vector (20,)
    """
    analysis = analyze_v3_structure(v3_sequence)

    features = [
        # Charge features
        analysis["total_charge"],
        analysis["crown_charge"],
        analysis["regions"].get("stem_n", {}).get("charge", 0.0),
        analysis["regions"].get("stem_c", {}).get("charge", 0.0),

        # Hydrophobicity
        analysis["regions"].get("crown", {}).get("hydrophobicity", 0.0),
        analysis["regions"].get("stem_n", {}).get("hydrophobicity", 0.0),
        analysis["regions"].get("stem_c", {}).get("hydrophobicity", 0.0),

        # Key positions
        analysis["key_positions"].get("classic_11", {}).get("charge", 0.0),
        analysis["key_positions"].get("novel_22", {}).get("charge", 0.0),
        analysis["key_positions"].get("classic_25", {}).get("charge", 0.0),

        # Basic/acidic counts
        analysis["regions"].get("crown", {}).get("basic_count", 0),
        analysis["regions"].get("crown", {}).get("acidic_count", 0),
        analysis["regions"].get("stem_n", {}).get("basic_count", 0),
        analysis["regions"].get("stem_c", {}).get("basic_count", 0),

        # Binary features
        float(analysis["rule_11_25_x4"]),
        float(analysis["charge_rule_x4"]),

        # Length and motif
        analysis["length"],
        float("GPG" in v3_sequence.upper()),
        float("GPGR" in v3_sequence.upper()),
        float("GPGQ" in v3_sequence.upper()),
    ]

    return np.array(features, dtype=np.float32)


# ============================================================================
# V2 LOOP ANALYSIS
# ============================================================================

def analyze_v2_structure(v2_sequence: str) -> dict:
    """
    Analyze V2 loop structural features.

    Based on Thielen et al. (2010):
    - V2 loop length correlates with tropism
    - Certain V2 positions add predictive power
    - Glycosylation sites in V2 affect antibody access

    Args:
        v2_sequence: V2 loop sequence

    Returns:
        dict: V2 structural analysis
    """
    seq = v2_sequence.upper()
    length = len(seq)

    # Find potential N-glycosylation sites (NX[ST] where X != P)
    glycan_sites = []
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST":
            glycan_sites.append(i + 1)  # 1-indexed

    results = {
        "length": length,
        "charge": calculate_net_charge(seq),
        "hydrophobicity": calculate_hydrophobicity(seq),
        "glycan_sites": glycan_sites,
        "glycan_count": len(glycan_sites),
        "basic_count": sum(1 for aa in seq if aa in "RKH"),
        "acidic_count": sum(1 for aa in seq if aa in "DE"),
        "proline_count": sum(1 for aa in seq if aa == "P"),
    }

    # V2 length categories (from literature)
    if length < 35:
        results["length_category"] = "short"
    elif length < 45:
        results["length_category"] = "medium"
    else:
        results["length_category"] = "long"

    return results


def calculate_v2_features_for_ml(v2_sequence: str) -> np.ndarray:
    """
    Extract V2 features as a numpy array for ML models.

    Features (10-dimensional):
    - Length, charge, hydrophobicity mean/std
    - Glycan count, basic count, acidic count
    - Proline count, length category encoded

    Args:
        v2_sequence: V2 loop sequence

    Returns:
        np.ndarray: Feature vector (10,)
    """
    analysis = analyze_v2_structure(v2_sequence)

    length_encoding = {
        "short": 0.0,
        "medium": 0.5,
        "long": 1.0,
    }

    features = [
        analysis["length"],
        analysis["charge"],
        analysis["hydrophobicity"]["mean"],
        analysis["hydrophobicity"]["std"],
        analysis["glycan_count"],
        analysis["basic_count"],
        analysis["acidic_count"],
        analysis["proline_count"],
        length_encoding.get(analysis["length_category"], 0.5),
        float(analysis["glycan_count"] >= 3),  # High glycosylation
    ]

    return np.array(features, dtype=np.float32)


# ============================================================================
# COMBINED V2V3 ANALYSIS
# ============================================================================

def analyze_v2v3_combined(
    v2_sequence: Optional[str],
    v3_sequence: str
) -> dict:
    """
    Combined V2V3 analysis for enhanced tropism prediction.

    Based on Thielen et al. (2010):
    - V2 + V3 together improve prediction by 3-5%

    Args:
        v2_sequence: V2 loop sequence (optional)
        v3_sequence: V3 loop sequence

    Returns:
        dict: Combined analysis
    """
    v3_analysis = analyze_v3_structure(v3_sequence)

    results = {
        "v3": v3_analysis,
        "v2": None,
        "combined_charge": v3_analysis["total_charge"],
        "combined_glycan_count": 0,
    }

    if v2_sequence:
        v2_analysis = analyze_v2_structure(v2_sequence)
        results["v2"] = v2_analysis
        results["combined_charge"] += v2_analysis["charge"]
        results["combined_glycan_count"] = v2_analysis["glycan_count"]

    # Enhanced tropism prediction
    results["enhanced_x4_prediction"] = (
        results["combined_charge"] >= 6 or
        v3_analysis["rule_11_25_x4"] or
        v3_analysis["crown_charge"] >= 4
    )

    return results


def calculate_v2v3_features_for_ml(
    v2_sequence: Optional[str],
    v3_sequence: str
) -> np.ndarray:
    """
    Combined V2V3 feature vector for ML.

    Total: 30 features (20 V3 + 10 V2)

    Args:
        v2_sequence: V2 loop sequence (optional, zeros if None)
        v3_sequence: V3 loop sequence

    Returns:
        np.ndarray: Combined feature vector (30,)
    """
    v3_features = calculate_v3_features_for_ml(v3_sequence)

    if v2_sequence:
        v2_features = calculate_v2_features_for_ml(v2_sequence)
    else:
        v2_features = np.zeros(10, dtype=np.float32)

    return np.concatenate([v3_features, v2_features])


# ============================================================================
# GLYCOSYLATION ANALYSIS
# ============================================================================

def find_glycosylation_sites(sequence: str) -> list[dict]:
    """
    Find potential N-linked glycosylation sites (PNGS).

    Pattern: N-X-[ST] where X is not P

    Args:
        sequence: Amino acid sequence

    Returns:
        list[dict]: List of glycosylation sites with details
    """
    seq = sequence.upper()
    sites = []

    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST":
            sites.append({
                "position": i + 1,  # 1-indexed
                "motif": seq[i:i + 3],
                "context": seq[max(0, i - 2):min(len(seq), i + 5)],
            })

    return sites


def calculate_glycan_density(sequence: str, window_size: int = 20) -> np.ndarray:
    """
    Calculate sliding window glycan density.

    Args:
        sequence: Amino acid sequence
        window_size: Size of sliding window

    Returns:
        np.ndarray: Glycan density at each position
    """
    sites = find_glycosylation_sites(sequence)
    site_positions = [s["position"] - 1 for s in sites]  # 0-indexed

    density = np.zeros(len(sequence))

    for pos in site_positions:
        start = max(0, pos - window_size // 2)
        end = min(len(sequence), pos + window_size // 2)
        density[start:end] += 1

    # Normalize
    if density.max() > 0:
        density = density / density.max()

    return density


# ============================================================================
# HILL COEFFICIENT CALCULATION
# ============================================================================

def calculate_hill_coefficient(ic50: float, ic80: float) -> float:
    """
    Calculate Hill coefficient from IC50/IC80 ratio.

    Based on Gilbert et al. (2022):
    - Hill coefficient indicates binding cooperativity
    - Higher Hill = steeper dose-response = better therapeutic
    - Typical range for bnAbs: 1-3

    Formula: n = log(4) / log(IC80/IC50)
    (Derived from: IC80/IC50 = 4^(1/n))

    Args:
        ic50: 50% inhibitory concentration
        ic80: 80% inhibitory concentration

    Returns:
        float: Hill coefficient (typically 1-3)
    """
    if ic50 <= 0 or ic80 <= 0:
        return np.nan

    if ic80 <= ic50:
        # Invalid: IC80 should be greater than IC50
        return np.nan

    ratio = ic80 / ic50

    if ratio <= 1:
        return np.nan

    # n = log(4) / log(ratio)
    hill = np.log(4) / np.log(ratio)

    # Clamp to reasonable range
    return np.clip(hill, 0.1, 10.0)


def analyze_neutralization_curve(
    ic50: float,
    ic80: float,
    ic90: Optional[float] = None
) -> dict:
    """
    Analyze neutralization curve characteristics.

    Args:
        ic50: 50% inhibitory concentration
        ic80: 80% inhibitory concentration
        ic90: 90% inhibitory concentration (optional)

    Returns:
        dict: Curve analysis including Hill coefficient
    """
    hill = calculate_hill_coefficient(ic50, ic80)

    results = {
        "ic50": ic50,
        "ic80": ic80,
        "ic90": ic90,
        "hill_coefficient": hill,
        "potency_category": "unknown",
        "cooperativity": "unknown",
    }

    # Categorize potency
    if ic50 < 0.1:
        results["potency_category"] = "very_potent"
    elif ic50 < 1.0:
        results["potency_category"] = "potent"
    elif ic50 < 10.0:
        results["potency_category"] = "moderate"
    else:
        results["potency_category"] = "weak"

    # Categorize cooperativity
    if not np.isnan(hill):
        if hill < 1.0:
            results["cooperativity"] = "negative"
        elif hill < 1.5:
            results["cooperativity"] = "non_cooperative"
        elif hill < 2.5:
            results["cooperativity"] = "cooperative"
        else:
            results["cooperativity"] = "highly_cooperative"

    return results


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def extract_all_structural_features(
    v3_sequence: str,
    v2_sequence: Optional[str] = None,
    include_glycans: bool = True
) -> dict:
    """
    Extract all structural features for a sequence.

    Args:
        v3_sequence: V3 loop sequence
        v2_sequence: V2 loop sequence (optional)
        include_glycans: Whether to include glycosylation analysis

    Returns:
        dict: All structural features
    """
    results = {
        "v3_analysis": analyze_v3_structure(v3_sequence),
        "v3_features": calculate_v3_features_for_ml(v3_sequence).tolist(),
    }

    if v2_sequence:
        results["v2_analysis"] = analyze_v2_structure(v2_sequence)
        results["v2_features"] = calculate_v2_features_for_ml(v2_sequence).tolist()
        results["combined"] = analyze_v2v3_combined(v2_sequence, v3_sequence)
        results["combined_features"] = calculate_v2v3_features_for_ml(
            v2_sequence, v3_sequence
        ).tolist()

    if include_glycans:
        results["v3_glycans"] = find_glycosylation_sites(v3_sequence)
        if v2_sequence:
            results["v2_glycans"] = find_glycosylation_sites(v2_sequence)

    return results


if __name__ == "__main__":
    # Example usage
    v3_example = "CTRPNNNTRKSIHIGPGRAFYATGDIIGDIRQAHC"
    v2_example = "CSFNITTSIRNKVQKEYALFYKLDVVPIDNDNTS"

    print("V3 Analysis:")
    print("-" * 50)
    v3_result = analyze_v3_structure(v3_example)
    print(f"Length: {v3_result['length']}")
    print(f"Total charge: {v3_result['total_charge']}")
    print(f"Crown charge: {v3_result['crown_charge']}")
    print(f"11/25 rule X4: {v3_result['rule_11_25_x4']}")
    print(f"Charge rule X4: {v3_result['charge_rule_x4']}")

    print("\nV2 Analysis:")
    print("-" * 50)
    v2_result = analyze_v2_structure(v2_example)
    print(f"Length: {v2_result['length']}")
    print(f"Charge: {v2_result['charge']}")
    print(f"Glycan sites: {v2_result['glycan_count']}")

    print("\nCombined V2V3:")
    print("-" * 50)
    combined = analyze_v2v3_combined(v2_example, v3_example)
    print(f"Combined charge: {combined['combined_charge']}")
    print(f"Enhanced X4 prediction: {combined['enhanced_x4_prediction']}")

    print("\nML Features:")
    print("-" * 50)
    features = calculate_v2v3_features_for_ml(v2_example, v3_example)
    print(f"Feature vector shape: {features.shape}")
    print(f"Features: {features}")
