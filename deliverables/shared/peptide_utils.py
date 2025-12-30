# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Shared peptide utility functions.

This module consolidates commonly used peptide analysis functions
to eliminate code duplication across bioinformatics scripts.

Functions:
    compute_peptide_properties: Calculate biophysical properties
    compute_ml_features: Generate feature vector for ML models
    compute_amino_acid_composition: Get AA composition fractions
    decode_latent_to_sequence: Convert latent vector to sequence
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .constants import AMINO_ACIDS, CHARGES, HYDROPHOBICITY, VOLUMES

# Combined AA properties dict for backward compatibility
AA_PROPERTIES = {}
for aa in AMINO_ACIDS:
    AA_PROPERTIES[aa] = {
        "charge": CHARGES.get(aa, 0),
        "hydrophobicity": HYDROPHOBICITY.get(aa, 0),
        "volume": VOLUMES.get(aa, 100),
    }


def compute_peptide_properties(sequence: str) -> dict:
    """Compute biophysical properties of peptide sequence.

    Args:
        sequence: Amino acid sequence (single-letter code)

    Returns:
        Dictionary with:
            - net_charge: Total charge at pH 7.4
            - hydrophobicity: Mean Kyte-Doolittle hydrophobicity
            - length: Sequence length
            - hydrophobic_ratio: Fraction of hydrophobic residues
            - cationic_ratio: Fraction of cationic residues (K, R, H)
    """
    if not sequence:
        return {
            "net_charge": 0,
            "hydrophobicity": 0,
            "length": 0,
            "hydrophobic_ratio": 0,
            "cationic_ratio": 0,
        }

    total_charge = 0
    total_hydro = 0
    valid_count = 0

    for aa in sequence:
        if aa in AA_PROPERTIES:
            total_charge += AA_PROPERTIES[aa]["charge"]
            total_hydro += AA_PROPERTIES[aa]["hydrophobicity"]
            valid_count += 1

    seq_len = len(sequence)
    hydrophobic_aa = set("AILMFVW")
    cationic_aa = set("KRH")

    hydro_ratio = sum(1 for aa in sequence if aa in hydrophobic_aa) / max(seq_len, 1)
    cationic_ratio = sum(1 for aa in sequence if aa in cationic_aa) / max(seq_len, 1)

    if valid_count == 0:
        return {
            "net_charge": 0,
            "hydrophobicity": 0,
            "length": seq_len,
            "hydrophobic_ratio": hydro_ratio,
            "cationic_ratio": cationic_ratio,
        }

    return {
        "net_charge": total_charge,
        "hydrophobicity": total_hydro / valid_count,
        "length": seq_len,
        "hydrophobic_ratio": hydro_ratio,
        "cationic_ratio": cationic_ratio,
    }


def compute_amino_acid_composition(sequence: str) -> np.ndarray:
    """Compute amino acid composition as frequency vector.

    Args:
        sequence: Amino acid sequence

    Returns:
        Array of 20 floats representing AA frequencies (sums to 1.0)
    """
    aa_list = list(AMINO_ACIDS)
    aa_comp = np.zeros(20)
    seq_len = len(sequence)

    if seq_len > 0:
        for i, aa in enumerate(aa_list):
            aa_comp[i] = sequence.count(aa) / seq_len

    return aa_comp


def compute_ml_features(sequence: str) -> np.ndarray:
    """Compute features for ML model prediction.

    Generates a 25-dimensional feature vector suitable for
    trained activity/property prediction models.

    Args:
        sequence: Amino acid sequence

    Returns:
        Feature array: [length, charge, hydro, hydro_ratio, cationic_ratio, aa_comp[20]]
    """
    props = compute_peptide_properties(sequence)
    aa_comp = compute_amino_acid_composition(sequence)

    # Combine features: [length, charge, hydro, hydro_ratio, cationic_ratio, aa_comp...]
    features = np.concatenate([
        np.array([
            props["length"],
            props["net_charge"],
            props["hydrophobicity"],
            props["hydrophobic_ratio"],
            props["cationic_ratio"],
        ]),
        aa_comp,
    ])

    return features


def decode_latent_to_sequence(
    z: np.ndarray,
    length: int = 20,
    seed: Optional[int] = None,
    use_vae: bool = True,
) -> str:
    """Decode latent vector to peptide sequence.

    Attempts to use the real VAE decoder if available, otherwise
    falls back to a heuristic property-based decoder.

    Args:
        z: Latent vector (at least 2 dimensions used)
        length: Target sequence length (only used in heuristic mode)
        seed: Random seed for heuristic mode
        use_vae: If True, try to use the real VAE service

    Returns:
        Generated amino acid sequence
    """
    # Try to use real VAE service
    if use_vae:
        try:
            from .vae_service import get_vae_service
            vae = get_vae_service()
            if vae.is_real:
                return vae.decode_latent(z)
            # VAE service exists but no model - use its mock decoder
            return vae.decode_latent(z)
        except ImportError:
            pass  # Fall through to heuristic decoder

    # Heuristic decoder (fallback)
    return _heuristic_decode(z, length, seed)


def _heuristic_decode(
    z: np.ndarray,
    length: int = 20,
    seed: Optional[int] = None,
) -> str:
    """Heuristic property-based sequence generator.

    WARNING: This is NOT a real VAE decoder. It generates sequences
    based on property preferences encoded in the latent vector.

    Args:
        z: Latent vector
        length: Target sequence length
        seed: Random seed

    Returns:
        Generated amino acid sequence
    """
    if seed is None:
        seed = int(abs(z[0] * 1000)) % (2**31)
    np.random.seed(seed)

    # Map latent dimensions to sequence properties
    charge_preference = np.tanh(z[0]) if len(z) > 0 else 0
    hydro_preference = np.tanh(z[1]) if len(z) > 1 else 0

    # Build amino acid probability distribution
    aa_list = list(AA_PROPERTIES.keys())
    probs = np.zeros(len(aa_list))

    for i, aa in enumerate(aa_list):
        props = AA_PROPERTIES[aa]
        # Score based on charge preference
        charge_score = 1 - abs(props["charge"] - charge_preference)
        # Score based on hydrophobicity preference (normalize to ~-1 to 1 range)
        hydro_score = 1 - abs(props["hydrophobicity"] / 2 - hydro_preference)
        probs[i] = charge_score + hydro_score + 0.1  # Base probability

    # Ensure non-negative probabilities
    probs = np.clip(probs, 0.01, None)
    probs = probs / probs.sum()

    # Generate sequence
    sequence = "".join(np.random.choice(aa_list, size=length, p=probs))

    return sequence


def decode_latent_with_vae(z: np.ndarray) -> tuple[str, bool]:
    """Decode latent vector using VAE service with status indicator.

    Args:
        z: Latent vector

    Returns:
        Tuple of (sequence, used_real_vae)
    """
    try:
        from .vae_service import get_vae_service
        vae = get_vae_service()
        sequence = vae.decode_latent(z)
        return sequence, vae.is_real
    except ImportError:
        return _heuristic_decode(z), False


def compute_physicochemical_descriptors(sequence: str) -> dict:
    """Compute extended physicochemical descriptors.

    Args:
        sequence: Amino acid sequence

    Returns:
        Dictionary with extended properties including:
        - Basic properties (charge, hydrophobicity, etc.)
        - Aromaticity (fraction of F, W, Y)
        - Instability index estimate
        - Aliphatic index
    """
    props = compute_peptide_properties(sequence)
    seq_len = len(sequence) if sequence else 1

    # Aromatic residues
    aromatic_aa = set("FWY")
    aromaticity = sum(1 for aa in sequence if aa in aromatic_aa) / seq_len

    # Aliphatic residues (weighted)
    aliphatic_index = (
        sequence.count("A") * 1.0
        + sequence.count("V") * 2.9
        + sequence.count("I") * 3.9
        + sequence.count("L") * 3.9
    ) / seq_len * 100 if seq_len > 0 else 0

    # Tiny, small, large residue fractions
    tiny_aa = set("AGS")
    small_aa = set("ACDGNPSTV")
    large_aa = set("FHKMRWY")

    tiny_ratio = sum(1 for aa in sequence if aa in tiny_aa) / seq_len
    small_ratio = sum(1 for aa in sequence if aa in small_aa) / seq_len
    large_ratio = sum(1 for aa in sequence if aa in large_aa) / seq_len

    # Polar vs nonpolar
    polar_aa = set("DEHKNQRST")
    polar_ratio = sum(1 for aa in sequence if aa in polar_aa) / seq_len

    return {
        **props,
        "aromaticity": aromaticity,
        "aliphatic_index": aliphatic_index,
        "tiny_ratio": tiny_ratio,
        "small_ratio": small_ratio,
        "large_ratio": large_ratio,
        "polar_ratio": polar_ratio,
    }


def validate_sequence(sequence: str, allow_ambiguous: bool = False) -> tuple[bool, str]:
    """Validate amino acid sequence.

    Args:
        sequence: Sequence to validate
        allow_ambiguous: If True, allow X (unknown) and B, Z (ambiguous)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sequence:
        return False, "Empty sequence"

    valid_aa = set(AMINO_ACIDS)
    if allow_ambiguous:
        valid_aa.update("XBZ")

    invalid_chars = [aa for aa in sequence if aa not in valid_aa]
    if invalid_chars:
        return False, f"Invalid amino acids: {set(invalid_chars)}"

    return True, ""
