# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Primer3 Integration for PCR Primer Design.

Provides utilities for designing PCR primers for peptide-encoding sequences.
Can interface with Primer3 if installed, or use built-in heuristics.

Usage:
    from shared.primer_design import PrimerDesigner

    designer = PrimerDesigner()
    primers = designer.design_primers("ATGGGCAAGTTCAAA...")
    print(f"Forward: {primers['forward']}")
    print(f"Reverse: {primers['reverse']}")
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import CODON_TABLE, PRIMER_DEFAULTS, IUPAC_AMBIGUOUS


@dataclass
class PrimerResult:
    """Container for primer design results."""

    forward: str
    reverse: str
    forward_tm: float
    reverse_tm: float
    forward_gc: float
    reverse_gc: float
    product_size: int
    penalty: float = 0.0
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def reverse_complement(seq: str) -> str:
    """Get reverse complement of DNA sequence.

    Args:
        seq: DNA sequence (ACGT)

    Returns:
        Reverse complement sequence
    """
    complement = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(complement.get(base, "N") for base in reversed(seq.upper()))


def calculate_tm(seq: str) -> float:
    """Calculate melting temperature using nearest-neighbor method.

    Simplified Wallace rule for short oligonucleotides.

    Args:
        seq: Primer sequence

    Returns:
        Estimated Tm in Celsius
    """
    seq = seq.upper()
    gc_count = seq.count("G") + seq.count("C")
    at_count = seq.count("A") + seq.count("T")

    if len(seq) < 14:
        # Wallace rule for short oligos
        tm = 2 * at_count + 4 * gc_count
    else:
        # Simplified nearest-neighbor approximation
        tm = 64.9 + 41 * (gc_count - 16.4) / len(seq)

    return tm


def calculate_gc(seq: str) -> float:
    """Calculate GC content as percentage.

    Args:
        seq: DNA sequence

    Returns:
        GC percentage (0-100)
    """
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return 100 * gc / len(seq) if len(seq) > 0 else 0


def check_self_complementarity(seq: str, max_run: int = 4) -> int:
    """Check for self-complementary runs (hairpins).

    Args:
        seq: Primer sequence
        max_run: Maximum allowed complementary run

    Returns:
        Longest self-complementary run found
    """
    seq = seq.upper()
    rev_comp = reverse_complement(seq)
    longest = 0

    for i in range(len(seq)):
        for j in range(len(rev_comp)):
            run = 0
            while (i + run < len(seq) and j + run < len(rev_comp) and
                   seq[i + run] == rev_comp[j + run]):
                run += 1
            longest = max(longest, run)

    return longest


def check_homopolymer(seq: str) -> int:
    """Find longest homopolymer run in sequence.

    Args:
        seq: DNA sequence

    Returns:
        Length of longest single-base run
    """
    if not seq:
        return 0

    seq = seq.upper()
    max_run = 1
    current_run = 1

    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    return max_run


class PrimerDesigner:
    """Design PCR primers for DNA sequences.

    Can use Primer3 if installed, otherwise uses built-in heuristics.
    """

    def __init__(
        self,
        min_length: int = 18,
        max_length: int = 25,
        min_gc: float = 40,
        max_gc: float = 60,
        min_tm: float = 55,
        max_tm: float = 65,
        max_homopolymer: int = 4,
    ):
        """Initialize primer designer.

        Args:
            min_length: Minimum primer length
            max_length: Maximum primer length
            min_gc: Minimum GC content (%)
            max_gc: Maximum GC content (%)
            min_tm: Minimum melting temperature (C)
            max_tm: Maximum melting temperature (C)
            max_homopolymer: Maximum homopolymer run allowed
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_gc = min_gc
        self.max_gc = max_gc
        self.min_tm = min_tm
        self.max_tm = max_tm
        self.max_homopolymer = max_homopolymer

        # Check if Primer3 is available
        self._primer3_available = self._check_primer3()

    def _check_primer3(self) -> bool:
        """Check if Primer3 is installed and available."""
        try:
            result = subprocess.run(
                ["primer3_core", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def design_primers(
        self,
        sequence: str,
        target_start: Optional[int] = None,
        target_length: Optional[int] = None,
        product_size_range: tuple[int, int] = (100, 500),
    ) -> PrimerResult:
        """Design primers for a DNA sequence.

        Args:
            sequence: Template DNA sequence
            target_start: Start position of target region (optional)
            target_length: Length of target region (optional)
            product_size_range: Desired product size range

        Returns:
            PrimerResult with designed primers
        """
        sequence = sequence.upper().replace(" ", "").replace("\n", "")

        if len(sequence) < product_size_range[0]:
            # Sequence too short, design primers at ends
            return self._design_simple_primers(sequence)

        if self._primer3_available:
            try:
                return self._design_with_primer3(
                    sequence, target_start, target_length, product_size_range
                )
            except Exception:
                pass  # Fall back to heuristic method

        return self._design_heuristic_primers(
            sequence, target_start, target_length, product_size_range
        )

    def _design_simple_primers(self, sequence: str) -> PrimerResult:
        """Design simple primers at sequence ends."""
        forward = sequence[:min(self.max_length, len(sequence))]
        reverse_seq = sequence[-min(self.max_length, len(sequence)):]
        reverse = reverse_complement(reverse_seq)

        return PrimerResult(
            forward=forward,
            reverse=reverse,
            forward_tm=calculate_tm(forward),
            reverse_tm=calculate_tm(reverse),
            forward_gc=calculate_gc(forward),
            reverse_gc=calculate_gc(reverse),
            product_size=len(sequence),
            warnings=["Sequence shorter than minimum product size"],
        )

    def _design_heuristic_primers(
        self,
        sequence: str,
        target_start: Optional[int],
        target_length: Optional[int],
        product_size_range: tuple[int, int],
    ) -> PrimerResult:
        """Design primers using heuristic approach."""
        warnings = []

        # Define search regions
        if target_start is not None and target_length is not None:
            # Design around target
            forward_region_end = target_start
            reverse_region_start = target_start + target_length
        else:
            # Design for full sequence
            forward_region_end = min(100, len(sequence) // 3)
            reverse_region_start = max(len(sequence) - 100, 2 * len(sequence) // 3)

        # Find best forward primer
        best_forward = None
        best_forward_score = float("inf")

        for start in range(0, forward_region_end):
            for length in range(self.min_length, self.max_length + 1):
                if start + length > len(sequence):
                    break
                primer = sequence[start : start + length]
                score = self._score_primer(primer)
                if score < best_forward_score:
                    best_forward_score = score
                    best_forward = (primer, start)

        # Find best reverse primer
        best_reverse = None
        best_reverse_score = float("inf")

        for end in range(len(sequence), reverse_region_start, -1):
            for length in range(self.min_length, self.max_length + 1):
                start = end - length
                if start < 0:
                    break
                primer_template = sequence[start:end]
                primer = reverse_complement(primer_template)
                score = self._score_primer(primer)
                if score < best_reverse_score:
                    best_reverse_score = score
                    best_reverse = (primer, start)

        if best_forward is None or best_reverse is None:
            warnings.append("Could not find suitable primers")
            forward = sequence[: self.min_length]
            reverse = reverse_complement(sequence[-self.min_length :])
        else:
            forward = best_forward[0]
            forward_pos = best_forward[1]
            reverse = best_reverse[0]
            reverse_pos = best_reverse[1]

        forward_tm = calculate_tm(forward)
        reverse_tm = calculate_tm(reverse)
        forward_gc = calculate_gc(forward)
        reverse_gc = calculate_gc(reverse)

        # Calculate product size
        if best_forward and best_reverse:
            product_size = best_reverse[1] + len(reverse) - best_forward[1]
        else:
            product_size = len(sequence)

        # Add warnings for suboptimal primers
        if forward_tm < self.min_tm or forward_tm > self.max_tm:
            warnings.append(f"Forward Tm ({forward_tm:.1f}) outside target range")
        if reverse_tm < self.min_tm or reverse_tm > self.max_tm:
            warnings.append(f"Reverse Tm ({reverse_tm:.1f}) outside target range")
        if abs(forward_tm - reverse_tm) > 5:
            warnings.append(f"Tm difference ({abs(forward_tm - reverse_tm):.1f}) > 5C")

        return PrimerResult(
            forward=forward,
            reverse=reverse,
            forward_tm=forward_tm,
            reverse_tm=reverse_tm,
            forward_gc=forward_gc,
            reverse_gc=reverse_gc,
            product_size=product_size,
            penalty=best_forward_score + best_reverse_score,
            warnings=warnings,
        )

    def _score_primer(self, primer: str) -> float:
        """Score a primer (lower is better).

        Args:
            primer: Primer sequence

        Returns:
            Penalty score
        """
        score = 0.0

        # Tm penalty
        tm = calculate_tm(primer)
        if tm < self.min_tm:
            score += (self.min_tm - tm) * 2
        elif tm > self.max_tm:
            score += (tm - self.max_tm) * 2

        # GC penalty
        gc = calculate_gc(primer)
        if gc < self.min_gc:
            score += (self.min_gc - gc) * 0.5
        elif gc > self.max_gc:
            score += (gc - self.max_gc) * 0.5

        # Homopolymer penalty
        homo_run = check_homopolymer(primer)
        if homo_run > self.max_homopolymer:
            score += (homo_run - self.max_homopolymer) * 5

        # Self-complementarity penalty
        self_comp = check_self_complementarity(primer)
        if self_comp > 4:
            score += (self_comp - 4) * 3

        # 3' end penalty (should end with G or C)
        if primer[-1] not in "GC":
            score += 1

        return score

    def _design_with_primer3(
        self,
        sequence: str,
        target_start: Optional[int],
        target_length: Optional[int],
        product_size_range: tuple[int, int],
    ) -> PrimerResult:
        """Design primers using Primer3."""
        # Create Primer3 input file
        input_data = f"""SEQUENCE_ID=template
SEQUENCE_TEMPLATE={sequence}
PRIMER_TASK=generic
PRIMER_PICK_LEFT_PRIMER=1
PRIMER_PICK_RIGHT_PRIMER=1
PRIMER_OPT_SIZE=20
PRIMER_MIN_SIZE={self.min_length}
PRIMER_MAX_SIZE={self.max_length}
PRIMER_MIN_TM={self.min_tm}
PRIMER_MAX_TM={self.max_tm}
PRIMER_MIN_GC={self.min_gc}
PRIMER_MAX_GC={self.max_gc}
PRIMER_PRODUCT_SIZE_RANGE={product_size_range[0]}-{product_size_range[1]}
PRIMER_NUM_RETURN=1
=
"""

        if target_start is not None and target_length is not None:
            input_data = input_data.replace(
                "PRIMER_TASK=generic",
                f"PRIMER_TASK=generic\nSEQUENCE_TARGET={target_start},{target_length}",
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(input_data)
            input_file = f.name

        try:
            result = subprocess.run(
                ["primer3_core", input_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout
            Path(input_file).unlink()

            # Parse Primer3 output
            return self._parse_primer3_output(output)

        except Exception as e:
            Path(input_file).unlink(missing_ok=True)
            raise RuntimeError(f"Primer3 failed: {e}")

    def _parse_primer3_output(self, output: str) -> PrimerResult:
        """Parse Primer3 output."""
        data = {}
        for line in output.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                data[key] = value

        forward = data.get("PRIMER_LEFT_0_SEQUENCE", "")
        reverse = data.get("PRIMER_RIGHT_0_SEQUENCE", "")
        forward_tm = float(data.get("PRIMER_LEFT_0_TM", 0))
        reverse_tm = float(data.get("PRIMER_RIGHT_0_TM", 0))
        forward_gc = float(data.get("PRIMER_LEFT_0_GC_PERCENT", 0))
        reverse_gc = float(data.get("PRIMER_RIGHT_0_GC_PERCENT", 0))
        product_size = int(data.get("PRIMER_PAIR_0_PRODUCT_SIZE", 0))
        penalty = float(data.get("PRIMER_PAIR_0_PENALTY", 0))

        warnings = []
        if "PRIMER_ERROR" in data:
            warnings.append(data["PRIMER_ERROR"])
        if "PRIMER_WARNING" in data:
            warnings.append(data["PRIMER_WARNING"])

        return PrimerResult(
            forward=forward,
            reverse=reverse,
            forward_tm=forward_tm,
            reverse_tm=reverse_tm,
            forward_gc=forward_gc,
            reverse_gc=reverse_gc,
            product_size=product_size,
            penalty=penalty,
            warnings=warnings,
        )

    def peptide_to_dna(
        self,
        peptide: str,
        codon_optimization: str = "ecoli",
    ) -> str:
        """Convert peptide sequence to DNA using optimized codons.

        Args:
            peptide: Amino acid sequence
            codon_optimization: Optimization target ('ecoli', 'human', 'yeast')

        Returns:
            DNA sequence encoding the peptide
        """
        # Codon usage tables (simplified)
        ecoli_codons = {
            "A": "GCG", "C": "TGC", "D": "GAT", "E": "GAA", "F": "TTT",
            "G": "GGC", "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTG",
            "M": "ATG", "N": "AAC", "P": "CCG", "Q": "CAG", "R": "CGT",
            "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAT",
        }

        human_codons = {
            "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
            "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
            "M": "ATG", "N": "AAC", "P": "CCC", "Q": "CAG", "R": "AGG",
            "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC",
        }

        codons = ecoli_codons if codon_optimization == "ecoli" else human_codons

        dna = []
        for aa in peptide.upper():
            codon = codons.get(aa)
            if codon:
                dna.append(codon)
            else:
                dna.append("NNN")  # Unknown amino acid

        return "".join(dna)

    def design_for_peptide(
        self,
        peptide: str,
        codon_optimization: str = "ecoli",
        add_start_codon: bool = True,
        add_stop_codon: bool = True,
    ) -> PrimerResult:
        """Design primers to clone a peptide sequence.

        Args:
            peptide: Amino acid sequence
            codon_optimization: Target organism for codon usage
            add_start_codon: Add ATG start codon
            add_stop_codon: Add TAA stop codon

        Returns:
            PrimerResult with designed primers
        """
        dna = self.peptide_to_dna(peptide, codon_optimization)

        if add_start_codon and not peptide.startswith("M"):
            dna = "ATG" + dna

        if add_stop_codon:
            dna = dna + "TAA"

        return self.design_primers(dna)


def get_primer_designer(**kwargs) -> PrimerDesigner:
    """Get a primer designer instance.

    Args:
        **kwargs: Arguments passed to PrimerDesigner

    Returns:
        Configured PrimerDesigner
    """
    return PrimerDesigner(**kwargs)
