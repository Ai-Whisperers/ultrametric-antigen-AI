"""
Amino acid physical and chemical properties.

Properties used for encoding and biological interpretation.
"""

# Amino acid properties dictionary
# Properties: hydrophobicity, volume, charge, polarity, aromaticity
AMINO_ACID_PROPERTIES: dict[str, dict[str, float]] = {
    "A": {"hydrophobicity": 1.8, "volume": 88.6, "charge": 0, "polarity": 0.0, "aromaticity": 0},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "charge": 1, "polarity": 52.0, "aromaticity": 0},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "charge": 0, "polarity": 3.38, "aromaticity": 0},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "charge": -1, "polarity": 40.7, "aromaticity": 0},
    "C": {"hydrophobicity": 2.5, "volume": 108.5, "charge": 0, "polarity": 1.48, "aromaticity": 0},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "charge": 0, "polarity": 3.53, "aromaticity": 0},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "charge": -1, "polarity": 49.9, "aromaticity": 0},
    "G": {"hydrophobicity": -0.4, "volume": 60.1, "charge": 0, "polarity": 0.0, "aromaticity": 0},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "charge": 0.5, "polarity": 51.6, "aromaticity": 1},
    "I": {"hydrophobicity": 4.5, "volume": 166.7, "charge": 0, "polarity": 0.0, "aromaticity": 0},
    "L": {"hydrophobicity": 3.8, "volume": 166.7, "charge": 0, "polarity": 0.0, "aromaticity": 0},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "charge": 1, "polarity": 49.5, "aromaticity": 0},
    "M": {"hydrophobicity": 1.9, "volume": 162.9, "charge": 0, "polarity": 1.43, "aromaticity": 0},
    "F": {"hydrophobicity": 2.8, "volume": 189.9, "charge": 0, "polarity": 0.0, "aromaticity": 1},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "charge": 0, "polarity": 0.0, "aromaticity": 0},
    "S": {"hydrophobicity": -0.8, "volume": 89.0, "charge": 0, "polarity": 1.67, "aromaticity": 0},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "charge": 0, "polarity": 1.66, "aromaticity": 0},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "charge": 0, "polarity": 2.1, "aromaticity": 1},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "charge": 0, "polarity": 1.61, "aromaticity": 1},
    "V": {"hydrophobicity": 4.2, "volume": 140.0, "charge": 0, "polarity": 0.0, "aromaticity": 0},
    "*": {"hydrophobicity": 0.0, "volume": 0.0, "charge": 0, "polarity": 0.0, "aromaticity": 0},  # Stop
}

# Property ranges for normalization
PROPERTY_RANGES: dict[str, tuple[float, float]] = {
    "hydrophobicity": (-4.5, 4.5),
    "volume": (60.1, 227.8),
    "charge": (-1, 1),
    "polarity": (0.0, 52.0),
    "aromaticity": (0, 1),
}


def get_aa_vector(aa: str, normalize: bool = True) -> tuple[float, ...]:
    """
    Get property vector for amino acid.

    Args:
        aa: Single-letter amino acid code
        normalize: Whether to normalize to [0, 1]

    Returns:
        Tuple of (hydrophobicity, volume, charge, polarity, aromaticity)
    """
    props = AMINO_ACID_PROPERTIES.get(aa.upper(), AMINO_ACID_PROPERTIES["A"])

    values = (
        props["hydrophobicity"],
        props["volume"],
        props["charge"],
        props["polarity"],
        props["aromaticity"],
    )

    if normalize:
        normalized = []
        for val, prop_name in zip(values, PROPERTY_RANGES.keys()):
            min_val, max_val = PROPERTY_RANGES[prop_name]
            norm = (val - min_val) / (max_val - min_val + 1e-7)
            normalized.append(norm)
        return tuple(normalized)

    return values


def get_property(aa: str, property_name: str) -> float:
    """
    Get single property for amino acid.

    Args:
        aa: Single-letter amino acid code
        property_name: One of hydrophobicity, volume, charge, polarity, aromaticity

    Returns:
        Property value
    """
    props = AMINO_ACID_PROPERTIES.get(aa.upper())
    if props is None:
        return 0.0
    return props.get(property_name, 0.0)


def property_distance(aa1: str, aa2: str) -> float:
    """
    Calculate Euclidean distance between amino acids in property space.

    Args:
        aa1: First amino acid
        aa2: Second amino acid

    Returns:
        Distance in normalized property space
    """
    v1 = get_aa_vector(aa1, normalize=True)
    v2 = get_aa_vector(aa2, normalize=True)

    return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5


def is_similar(aa1: str, aa2: str, threshold: float = 0.3) -> bool:
    """
    Check if two amino acids are biochemically similar.

    Args:
        aa1: First amino acid
        aa2: Second amino acid
        threshold: Distance threshold for similarity

    Returns:
        True if similar
    """
    return property_distance(aa1, aa2) < threshold


# Amino acid groupings
HYDROPHOBIC = frozenset("AILMFVPWG")
POLAR = frozenset("STYNQ")
CHARGED_POSITIVE = frozenset("RKH")
CHARGED_NEGATIVE = frozenset("DE")
AROMATIC = frozenset("FWY")
SMALL = frozenset("GASTC")
