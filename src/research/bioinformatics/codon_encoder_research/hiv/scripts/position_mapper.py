# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
HIV Position Mapper

Maps between different HIV position numbering systems:
- HXB2 nucleotide positions (1-9719)
- HXB2 amino acid positions by protein
- Stanford HIVDB position columns
- LANL epitope positions

Reference: HXB2 (K03455.1)

Key Regions:
- Gag: nt 790-2292 (aa 1-500)
  - p17: nt 790-1185 (aa 1-132)
  - p24: nt 1186-1878 (aa 133-363)
  - p7: nt 1921-2085 (aa 378-433)
  - p6: nt 2134-2292 (aa 449-500)
- Pol: nt 2085-5096 (aa 1-1003)
  - Protease (PR): nt 2253-2549 (aa 1-99)
  - RT: nt 2550-3869 (aa 1-440)
  - RNase H: nt 3870-4229 (aa 441-560)
  - Integrase (IN): nt 4230-5096 (aa 1-288)
- Env: nt 6225-8795 (aa 1-856)
  - gp120: nt 6225-7758 (aa 1-511)
    - V1: aa 131-156
    - V2: aa 157-196
    - V3: aa 296-331
    - V4: aa 385-418
    - V5: aa 460-469
  - gp41: nt 7759-8795 (aa 512-856)
    - MPER: aa 660-683
- Accessory proteins:
  - Vif: nt 5041-5619
  - Vpr: nt 5559-5850
  - Tat: nt 5831-6045 + 8379-8469 (spliced)
  - Rev: nt 5970-6045 + 8379-8653 (spliced)
  - Vpu: nt 6062-6310
  - Nef: nt 8797-9417
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# HXB2 REFERENCE COORDINATES
# ============================================================================


@dataclass
class GeneRegion:
    """Represents an HIV gene or protein region."""

    name: str
    nt_start: int
    nt_end: int
    aa_start: int
    aa_end: int
    parent: Optional[str] = None


# HXB2 gene coordinates (1-indexed)
HXB2_REGIONS = {
    # Main proteins
    "gag": GeneRegion("gag", 790, 2292, 1, 500),
    "pol": GeneRegion("pol", 2085, 5096, 1, 1003),
    "env": GeneRegion("env", 6225, 8795, 1, 856),
    # Gag subproteins
    "p17": GeneRegion("p17", 790, 1185, 1, 132, parent="gag"),
    "p24": GeneRegion("p24", 1186, 1878, 133, 363, parent="gag"),
    "p7": GeneRegion("p7", 1921, 2085, 378, 433, parent="gag"),
    "p6": GeneRegion("p6", 2134, 2292, 449, 500, parent="gag"),
    # Pol subproteins
    "pr": GeneRegion("pr", 2253, 2549, 1, 99, parent="pol"),
    "protease": GeneRegion("protease", 2253, 2549, 1, 99, parent="pol"),
    "rt": GeneRegion("rt", 2550, 3869, 1, 440, parent="pol"),
    "rnase": GeneRegion("rnase", 3870, 4229, 441, 560, parent="pol"),
    "in": GeneRegion("in", 4230, 5096, 1, 288, parent="pol"),
    "integrase": GeneRegion("integrase", 4230, 5096, 1, 288, parent="pol"),
    # Env subproteins
    "gp120": GeneRegion("gp120", 6225, 7758, 1, 511, parent="env"),
    "gp41": GeneRegion("gp41", 7759, 8795, 512, 856, parent="env"),
    # Accessory proteins
    "vif": GeneRegion("vif", 5041, 5619, 1, 192),
    "vpr": GeneRegion("vpr", 5559, 5850, 1, 96),
    "vpu": GeneRegion("vpu", 6062, 6310, 1, 82),
    "nef": GeneRegion("nef", 8797, 9417, 1, 206),
}

# V-loop positions in gp120 (amino acid positions)
V_LOOPS = {
    "V1": (131, 156),
    "V2": (157, 196),
    "V3": (296, 331),
    "V4": (385, 418),
    "V5": (460, 469),
}

# Key epitope regions
EPITOPE_REGIONS = {
    "CD4_binding_site": (365, 371),  # CD4bs in gp120
    "MPER": (660, 683),  # Membrane proximal external region in gp41
    "gp41_fusion_peptide": (512, 527),
}


# ============================================================================
# POSITION CONVERSION FUNCTIONS
# ============================================================================


def get_region(name: str) -> Optional[GeneRegion]:
    """Get gene region by name (case-insensitive)."""
    return HXB2_REGIONS.get(name.lower())


def hxb2_nt_to_aa(nt_pos: int, region: str) -> Optional[int]:
    """
    Convert HXB2 nucleotide position to amino acid position.

    Args:
        nt_pos: Nucleotide position (1-indexed)
        region: Gene/protein name (e.g., 'gag', 'pr', 'gp120')

    Returns:
        Amino acid position within the specified region, or None if out of range
    """
    gene = get_region(region)
    if gene is None:
        return None

    if nt_pos < gene.nt_start or nt_pos > gene.nt_end:
        return None

    # Calculate codon position (1-indexed)
    offset = nt_pos - gene.nt_start
    aa_offset = offset // 3
    aa_pos = gene.aa_start + aa_offset

    return aa_pos


def hxb2_aa_to_nt(aa_pos: int, region: str, codon_pos: int = 1) -> Optional[int]:
    """
    Convert amino acid position to HXB2 nucleotide position.

    Args:
        aa_pos: Amino acid position within the region
        region: Gene/protein name
        codon_pos: Position within codon (1, 2, or 3)

    Returns:
        Nucleotide position in HXB2, or None if out of range
    """
    gene = get_region(region)
    if gene is None:
        return None

    if aa_pos < gene.aa_start or aa_pos > gene.aa_end:
        return None

    aa_offset = aa_pos - gene.aa_start
    nt_pos = gene.nt_start + (aa_offset * 3) + (codon_pos - 1)

    return nt_pos


def protein_pos_to_hxb2(pos: int, protein: str) -> Optional[int]:
    """
    Convert protein-relative position to HXB2 amino acid position.

    For subproteins (PR, RT, IN), converts local position to absolute HXB2 position.

    Args:
        pos: Position within the protein (1-indexed)
        protein: Protein name (e.g., 'pr', 'rt', 'gp120')

    Returns:
        HXB2 amino acid position
    """
    gene = get_region(protein)
    if gene is None:
        return None

    # For subproteins, aa_start is the offset
    return gene.aa_start + pos - 1


def hxb2_to_protein_pos(hxb2_pos: int, protein: str) -> Optional[int]:
    """
    Convert HXB2 amino acid position to protein-relative position.

    Args:
        hxb2_pos: HXB2 amino acid position
        protein: Protein name

    Returns:
        Position within the protein (1-indexed)
    """
    gene = get_region(protein)
    if gene is None:
        return None

    if hxb2_pos < gene.aa_start or hxb2_pos > gene.aa_end:
        return None

    return hxb2_pos - gene.aa_start + 1


def get_v_loop_positions(loop: str = "V3") -> tuple[int, int]:
    """Get start and end positions of a V-loop in gp120."""
    return V_LOOPS.get(loop.upper(), (0, 0))


def is_in_v_loop(pos: int, loop: Optional[str] = None) -> bool | str:
    """
    Check if a gp120 position is in a variable loop.

    Args:
        pos: Position in gp120
        loop: Specific loop to check (V1-V5) or None to check all

    Returns:
        If loop specified: True/False
        If loop is None: Name of loop or False
    """
    if loop:
        start, end = V_LOOPS.get(loop.upper(), (0, 0))
        return start <= pos <= end

    for loop_name, (start, end) in V_LOOPS.items():
        if start <= pos <= end:
            return loop_name
    return False


# ============================================================================
# MUTATION PARSING
# ============================================================================


def parse_mutation(mut_string: str) -> Optional[dict]:
    """
    Parse a mutation string into components.

    Formats supported:
    - D30N (amino acid)
    - D30N/I (multiple mutants)
    - 30N (position + mutant only)
    - RT:K103N (protein-prefixed)

    Returns:
        Dictionary with 'protein', 'position', 'wild_type', 'mutant' keys
        or None if parsing fails
    """
    mut_string = mut_string.strip()

    # Check for protein prefix
    protein = None
    if ":" in mut_string:
        protein, mut_string = mut_string.split(":", 1)
        protein = protein.strip()
        mut_string = mut_string.strip()

    # Standard format: D30N
    pattern = r"^([A-Z*])?(\d+)([A-Z*/]+)$"
    match = re.match(pattern, mut_string)

    if match:
        wild_type = match.group(1)
        position = int(match.group(2))
        mutant = match.group(3)

        # Handle multiple mutants (D30N/I)
        mutants = mutant.replace("/", "")

        return {
            "protein": protein,
            "position": position,
            "wild_type": wild_type,
            "mutant": mutants,
            "mutant_list": list(mutants),
        }

    return None


def parse_mutation_list(mut_list: str, delimiter: str = ",") -> list[dict]:
    """
    Parse a comma-separated list of mutations.

    Args:
        mut_list: String like "D30N, M46I, R57G"
        delimiter: Separator character

    Returns:
        List of parsed mutation dictionaries
    """
    if not mut_list or str(mut_list).lower() == "nan":
        return []

    mutations = []
    for mut in str(mut_list).split(delimiter):
        mut = mut.strip()
        if mut:
            parsed = parse_mutation(mut)
            if parsed:
                mutations.append(parsed)

    return mutations


# ============================================================================
# STANFORD HIVDB POSITION MAPPING
# ============================================================================


def stanford_col_to_position(col_name: str) -> tuple[str, int]:
    """
    Convert Stanford column name to protein and position.

    Args:
        col_name: Column like 'P30', 'RT103', 'IN155'

    Returns:
        Tuple of (protein, position)
    """
    # Protease: P1-P99
    if col_name.startswith("P") and col_name[1:].isdigit():
        return ("PR", int(col_name[1:]))

    # RT: RT1-RT560
    if col_name.startswith("RT") and col_name[2:].isdigit():
        return ("RT", int(col_name[2:]))

    # Integrase: IN1-IN288
    if col_name.startswith("IN") and col_name[2:].isdigit():
        return ("IN", int(col_name[2:]))

    return (None, 0)


def position_to_stanford_col(protein: str, position: int) -> str:
    """
    Convert protein and position to Stanford column name.

    Args:
        protein: 'PR', 'RT', or 'IN'
        position: Position in the protein

    Returns:
        Column name like 'P30', 'RT103', 'IN155'
    """
    protein = protein.upper()
    if protein in ("PR", "PROTEASE"):
        return f"P{position}"
    elif protein in ("RT", "REVERSE TRANSCRIPTASE"):
        return f"RT{position}"
    elif protein in ("IN", "INTEGRASE"):
        return f"IN{position}"
    else:
        return f"{protein}{position}"


# ============================================================================
# EPITOPE POSITION UTILITIES
# ============================================================================


def epitope_to_positions(epitope: str, start_pos: int) -> list[int]:
    """
    Convert epitope sequence to list of positions.

    Args:
        epitope: Amino acid sequence
        start_pos: Starting HXB2 position

    Returns:
        List of HXB2 positions for each residue
    """
    return list(range(start_pos, start_pos + len(epitope)))


def positions_overlap(pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
    """Check if two position ranges overlap."""
    return pos1[0] <= pos2[1] and pos2[0] <= pos1[1]


def epitope_contains_position(epitope_start: int, epitope_end: int, position: int) -> bool:
    """Check if an epitope contains a specific position."""
    return epitope_start <= position <= epitope_end


# ============================================================================
# CODON EXTRACTION
# ============================================================================


def extract_codon_at_position(sequence: str, position: int, region: str = "env") -> Optional[str]:
    """
    Extract the codon at a given amino acid position.

    Args:
        sequence: Nucleotide sequence (aligned to HXB2)
        position: Amino acid position in the region
        region: Gene/protein name

    Returns:
        3-letter codon string or None
    """
    gene = get_region(region)
    if gene is None:
        return None

    # Calculate nucleotide offset
    aa_offset = position - gene.aa_start
    nt_offset = aa_offset * 3

    if nt_offset < 0 or nt_offset + 3 > len(sequence):
        return None

    return sequence[nt_offset : nt_offset + 3].upper()


def sequence_to_codons(sequence: str) -> list[str]:
    """
    Split a nucleotide sequence into codons.

    Args:
        sequence: Nucleotide sequence

    Returns:
        List of 3-letter codon strings
    """
    sequence = sequence.upper().replace(" ", "").replace("\n", "")
    codons = []

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i : i + 3]
        if len(codon) == 3:
            codons.append(codon)

    return codons


def codons_to_amino_acids(codons: list[str]) -> str:
    """Convert list of codons to amino acid sequence."""
    codon_table = {
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

    aa_seq = []
    for codon in codons:
        codon = codon.upper().replace("U", "T")  # Handle RNA
        aa = codon_table.get(codon, "X")  # X for unknown
        aa_seq.append(aa)

    return "".join(aa_seq)


# ============================================================================
# CROSS-DATASET POSITION ALIGNMENT
# ============================================================================


def align_stanford_to_ctl(stanford_protein: str, stanford_pos: int, ctl_protein: str, ctl_start: int) -> bool:
    """
    Check if a Stanford resistance position overlaps with a CTL epitope.

    Args:
        stanford_protein: 'PR', 'RT', or 'IN'
        stanford_pos: Position in Stanford data
        ctl_protein: Protein from CTL database
        ctl_start: HXB2 start position of epitope

    Returns:
        True if positions could overlap
    """
    # Map Stanford to HXB2
    stanford_hxb2 = protein_pos_to_hxb2(stanford_pos, stanford_protein.lower())
    if stanford_hxb2 is None:
        return False

    # Map CTL protein to HXB2 region
    ctl_region = get_region(ctl_protein.lower())
    if ctl_region is None:
        return False

    # Check if Stanford position falls within CTL region
    ctl_hxb2 = ctl_region.aa_start + ctl_start - 1
    return stanford_hxb2 == ctl_hxb2


def find_overlapping_epitopes(
    mutation_protein: str, mutation_pos: int, epitopes: list[dict], epitope_length: int = 9
) -> list[dict]:
    """
    Find CTL epitopes that contain a given mutation position.

    Args:
        mutation_protein: Protein containing mutation
        mutation_pos: Position of mutation
        epitopes: List of epitope dictionaries with 'Protein', 'HXB2_start' keys
        epitope_length: Assumed epitope length for those without end position

    Returns:
        List of overlapping epitope dictionaries
    """
    overlapping = []

    for epitope in epitopes:
        epi_protein = epitope.get("Protein", "")
        epi_start = epitope.get("HXB2_start")
        epi_end = epitope.get("HXB2_end")

        if epi_start is None:
            continue

        # Estimate end if not provided
        if epi_end is None:
            epi_length = len(epitope.get("Epitope", "")) or epitope_length
            epi_end = epi_start + epi_length - 1

        # Check protein match
        if mutation_protein.lower() not in epi_protein.lower():
            continue

        # Check position overlap
        if epi_start <= mutation_pos <= epi_end:
            overlapping.append(epitope)

    return overlapping


# ============================================================================
# MAIN - TEST FUNCTIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HIV Position Mapper - Test Suite")
    print("=" * 60)

    # Test region lookup
    print("\n1. Region Lookup:")
    for region in ["pr", "rt", "in", "gp120", "gp41"]:
        r = get_region(region)
        if r:
            print(f"   {region.upper()}: nt {r.nt_start}-{r.nt_end}, aa {r.aa_start}-{r.aa_end}")

    # Test position conversion
    print("\n2. Position Conversion:")
    pr_pos = 30
    hxb2 = protein_pos_to_hxb2(pr_pos, "pr")
    print(f"   PR position {pr_pos} -> HXB2 aa {hxb2}")

    rt_pos = 103
    hxb2 = protein_pos_to_hxb2(rt_pos, "rt")
    print(f"   RT position {rt_pos} -> HXB2 aa {hxb2}")

    # Test mutation parsing
    print("\n3. Mutation Parsing:")
    mutations = ["D30N", "M46I", "RT:K103N", "L90M/V"]
    for mut in mutations:
        parsed = parse_mutation(mut)
        print(f"   {mut}: {parsed}")

    # Test V-loop detection
    print("\n4. V-Loop Detection:")
    for pos in [150, 200, 300, 400, 465]:
        loop = is_in_v_loop(pos)
        print(f"   Position {pos}: {loop if loop else 'Not in V-loop'}")

    # Test Stanford column mapping
    print("\n5. Stanford Column Mapping:")
    for col in ["P30", "RT103", "IN155"]:
        protein, pos = stanford_col_to_position(col)
        print(f"   {col}: {protein} position {pos}")

    print("\n" + "=" * 60)
