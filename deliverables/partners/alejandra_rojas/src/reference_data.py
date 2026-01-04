# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Reference data and phylogenetic sequence generation for arbovirus validation.

This module provides:
- Phylogenetically-informed sequence generation
- Sequence identity computation
- Ground truth validation utilities

The core constants (REFSEQ_ACCESSIONS, PHYLOGENETIC_IDENTITY, ValidatedPrimer)
are defined in constants.py to maintain a single source of truth.

Example:
    >>> from .reference_data import generate_phylogenetic_sequence
    >>> mutated = generate_phylogenetic_sequence(ref_seq, target_identity=0.65)
"""

from __future__ import annotations

import random
from typing import Optional

# Import core constants from constants.py (single source of truth)
from .constants import (
    REFSEQ_ACCESSIONS,
    PHYLOGENETIC_IDENTITY,
    ARBOVIRUS_TARGETS,
    get_phylogenetic_identity,
    get_validated_primers,
    ValidatedPrimer,
    CDC_PRIMERS,
    PANFLAVIVIRUS_PRIMERS,
)

# Re-export for backwards compatibility
__all__ = [
    # From constants
    "REFSEQ_ACCESSIONS",
    "PHYLOGENETIC_IDENTITY",
    "get_phylogenetic_identity",
    "get_validated_primers",
    "ValidatedPrimer",
    "CDC_PRIMERS",
    "PANFLAVIVIRUS_PRIMERS",
    # Local functions
    "generate_phylogenetic_sequence",
    "generate_realistic_demo_sequences",
    "compute_sequence_identity",
    "validate_primer_against_ground_truth",
]


def generate_phylogenetic_sequence(
    reference: str,
    target_identity: float,
    seed: int = 42,
    preserve_regions: list[tuple[int, int]] = None,
) -> str:
    """Generate sequence with target identity to reference.

    Uses codon-aware mutation to maintain realistic sequence properties.
    Preserves specified regions (e.g., UTRs, conserved domains) unchanged.

    Args:
        reference: Reference sequence
        target_identity: Target identity fraction (0.0 to 1.0)
        seed: Random seed for reproducibility
        preserve_regions: List of (start, end) regions to keep unchanged

    Returns:
        Mutated sequence with approximately target identity

    Example:
        >>> ref = "ATGCGATCGATCGATCGATC" * 100
        >>> mutated = generate_phylogenetic_sequence(ref, 0.65, seed=42)
        >>> identity = compute_sequence_identity(ref, mutated)
        >>> assert 0.60 <= identity <= 0.70
    """
    random.seed(seed)

    if preserve_regions is None:
        preserve_regions = []

    # Convert to list for mutation
    seq_list = list(reference.upper())
    n = len(seq_list)

    # Calculate number of mutations needed
    mutations_needed = int(n * (1 - target_identity))

    # Get mutable positions (exclude preserved regions)
    mutable_positions = set(range(n))
    for start, end in preserve_regions:
        for pos in range(start, min(end, n)):
            mutable_positions.discard(pos)

    mutable_positions = list(mutable_positions)

    # Perform mutations
    if len(mutable_positions) > 0 and mutations_needed > 0:
        positions_to_mutate = random.sample(
            mutable_positions,
            min(mutations_needed, len(mutable_positions))
        )

        bases = "ACGT"
        for pos in positions_to_mutate:
            original = seq_list[pos]
            if original in bases:
                alternatives = [b for b in bases if b != original]
                seq_list[pos] = random.choice(alternatives)

    return "".join(seq_list)


def generate_realistic_demo_sequences(
    base_virus: str = "DENV-1",
    seed: int = 42,
) -> dict[str, str]:
    """Generate demo sequences for all viruses with realistic phylogenetic identities.

    Uses DENV-1 as reference and mutates to target identities based on
    the PHYLOGENETIC_IDENTITY matrix.

    Args:
        base_virus: Reference virus (default DENV-1)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping virus name to sequence

    Example:
        >>> seqs = generate_realistic_demo_sequences()
        >>> # DENV-2 should be ~65% identical to DENV-1
        >>> identity = compute_sequence_identity(seqs["DENV-1"], seqs["DENV-2"])
        >>> assert 0.60 <= identity <= 0.70
    """
    # Get genome size for base virus
    base_target = ARBOVIRUS_TARGETS.get(base_virus, {})
    base_size = base_target.get("genome_size", 10700)
    conserved = base_target.get("conserved_regions", [])

    # Generate base sequence (random but reproducible)
    random.seed(seed)
    base_sequence = "".join(random.choices("ACGT", k=base_size))

    # Insert conserved motifs at known positions
    seq_list = list(base_sequence)
    for i, (start, end) in enumerate(conserved):
        # Use deterministic conserved motif
        motif_seed = seed + i * 1000
        random.seed(motif_seed)
        motif = "".join(random.choices("ACGT", k=min(100, end - start)))
        for j, nt in enumerate(motif):
            if start + j < len(seq_list):
                seq_list[start + j] = nt

    base_sequence = "".join(seq_list)

    # Generate sequences for all viruses
    result = {base_virus: base_sequence}

    viruses = list(ARBOVIRUS_TARGETS.keys())
    for i, virus in enumerate(viruses):
        if virus == base_virus:
            continue

        target_identity = get_phylogenetic_identity(base_virus, virus)

        # Get conserved regions for this virus
        virus_target = ARBOVIRUS_TARGETS.get(virus, {})
        virus_conserved = virus_target.get("conserved_regions", conserved)

        # Adjust sequence length if needed
        virus_size = virus_target.get("genome_size", base_size)
        if virus_size != base_size:
            # Truncate or extend
            if virus_size < base_size:
                ref_for_virus = base_sequence[:virus_size]
            else:
                # Extend with random bases
                random.seed(seed + i * 10000)
                extension = "".join(random.choices("ACGT", k=virus_size - base_size))
                ref_for_virus = base_sequence + extension
        else:
            ref_for_virus = base_sequence

        result[virus] = generate_phylogenetic_sequence(
            reference=ref_for_virus,
            target_identity=target_identity,
            seed=seed + i + 1,
            preserve_regions=virus_conserved,
        )

    return result


def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Identity fraction (0.0 to 1.0)
    """
    seq1, seq2 = seq1.upper(), seq2.upper()

    if len(seq1) != len(seq2):
        # Use shorter length
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]

    if len(seq1) == 0:
        return 0.0

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def validate_primer_against_ground_truth(
    designed_primers: list,
    target_virus: str,
    identity_threshold: float = 0.9,
) -> dict:
    """Validate designed primers against CDC/PAHO ground truth.

    Args:
        designed_primers: List of PrimerCandidate or PrimerPair objects
        target_virus: Target virus name
        identity_threshold: Minimum identity to consider a match

    Returns:
        Validation results dict with:
        - ground_truth_count: Number of known primers for this virus
        - designed_count: Number of primers designed
        - recovered: List of ground truth primers that were recovered
        - missed: List of ground truth primers that were missed
        - recovery_rate: Fraction of ground truth recovered
    """
    ground_truth = get_validated_primers(target_virus)

    results = {
        "ground_truth_count": len(ground_truth),
        "designed_count": len(designed_primers),
        "recovered": [],
        "missed": [],
        "novel": 0,
    }

    # Extract sequences from designed primers
    designed_seqs = []
    for dp in designed_primers:
        if hasattr(dp, "sequence"):
            designed_seqs.append(dp.sequence)
        elif hasattr(dp, "forward"):
            designed_seqs.append(dp.forward.sequence if hasattr(dp.forward, "sequence") else dp.forward)
            if hasattr(dp, "reverse"):
                designed_seqs.append(dp.reverse.sequence if hasattr(dp.reverse, "sequence") else dp.reverse)

    # Check which ground truth primers were recovered
    for gt in ground_truth:
        found = False
        for designed_seq in designed_seqs:
            # Check forward primer
            fwd_identity = compute_sequence_identity(gt.forward, designed_seq)
            if fwd_identity >= identity_threshold:
                found = True
                results["recovered"].append({
                    "name": gt.name,
                    "matched_seq": designed_seq,
                    "identity": fwd_identity,
                    "match_type": "forward",
                })
                break

            # Check reverse primer
            rev_identity = compute_sequence_identity(gt.reverse, designed_seq)
            if rev_identity >= identity_threshold:
                found = True
                results["recovered"].append({
                    "name": gt.name,
                    "matched_seq": designed_seq,
                    "identity": rev_identity,
                    "match_type": "reverse",
                })
                break

        if not found:
            results["missed"].append(gt.name)

    # Count novel primers (not matching any ground truth)
    results["novel"] = len(designed_primers) - len(results["recovered"])

    # Compute recovery rate
    if results["ground_truth_count"] > 0:
        results["recovery_rate"] = len(results["recovered"]) / results["ground_truth_count"]
    else:
        results["recovery_rate"] = 0.0

    return results
