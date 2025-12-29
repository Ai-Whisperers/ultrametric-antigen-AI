"""
Protein value objects - represent HIV proteins.

Covers all major HIV proteins with their properties.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class HIVProtein(Enum):
    """HIV protein enumeration."""

    # Structural proteins (Gag)
    MA = "ma"      # Matrix (p17)
    CA = "ca"      # Capsid (p24)
    NC = "nc"      # Nucleocapsid (p7)
    P6 = "p6"      # p6

    # Enzymes (Pol)
    PR = "pr"      # Protease
    RT = "rt"      # Reverse Transcriptase
    IN = "in"      # Integrase

    # Envelope
    GP120 = "gp120"  # Surface glycoprotein
    GP41 = "gp41"    # Transmembrane glycoprotein

    # Regulatory
    TAT = "tat"    # Trans-activator
    REV = "rev"    # Regulator of expression

    # Accessory
    NEF = "nef"    # Negative factor
    VIF = "vif"    # Viral infectivity factor
    VPR = "vpr"    # Viral protein R
    VPU = "vpu"    # Viral protein U


# Protein metadata
PROTEIN_INFO: dict[str, dict] = {
    "ma": {"length": 132, "parent": "gag", "function": "Membrane targeting"},
    "ca": {"length": 231, "parent": "gag", "function": "Core structure"},
    "nc": {"length": 55, "parent": "gag", "function": "RNA packaging"},
    "p6": {"length": 52, "parent": "gag", "function": "Virion release"},
    "pr": {"length": 99, "parent": "pol", "function": "Polyprotein cleavage"},
    "rt": {"length": 560, "parent": "pol", "function": "RNA to DNA"},
    "in": {"length": 288, "parent": "pol", "function": "DNA integration"},
    "gp120": {"length": 511, "parent": "env", "function": "Receptor binding"},
    "gp41": {"length": 345, "parent": "env", "function": "Membrane fusion"},
    "tat": {"length": 101, "parent": None, "function": "Transcription activation"},
    "rev": {"length": 116, "parent": None, "function": "RNA export"},
    "nef": {"length": 206, "parent": None, "function": "Immune evasion"},
    "vif": {"length": 192, "parent": None, "function": "APOBEC3 antagonist"},
    "vpr": {"length": 96, "parent": None, "function": "Cell cycle arrest"},
    "vpu": {"length": 81, "parent": None, "function": "CD4/BST2 antagonist"},
}


@dataclass(frozen=True, slots=True)
class Protein:
    """
    Represents an HIV protein.

    Attributes:
        name: Protein name/abbreviation
        hiv_protein: HIVProtein enum value (if known)
        length: Protein length in amino acids
        function: Brief functional description
    """

    name: str
    hiv_protein: Optional[HIVProtein] = None
    length: Optional[int] = None
    function: Optional[str] = None

    def __post_init__(self) -> None:
        name_lower = self.name.lower()

        # Try to match to HIVProtein enum
        if self.hiv_protein is None:
            try:
                hiv_prot = HIVProtein(name_lower)
                object.__setattr__(self, "hiv_protein", hiv_prot)
            except ValueError:
                pass

        # Get metadata from info dict
        if name_lower in PROTEIN_INFO:
            info = PROTEIN_INFO[name_lower]
            if self.length is None:
                object.__setattr__(self, "length", info["length"])
            if self.function is None:
                object.__setattr__(self, "function", info["function"])

    @classmethod
    def from_name(cls, name: str) -> "Protein":
        """Create Protein from name."""
        return cls(name=name)

    @property
    def is_enzyme(self) -> bool:
        """Check if protein is an enzyme."""
        return self.name.lower() in ("pr", "rt", "in")

    @property
    def is_structural(self) -> bool:
        """Check if protein is structural."""
        return self.name.lower() in ("ma", "ca", "nc", "p6", "gp120", "gp41")

    @property
    def is_accessory(self) -> bool:
        """Check if protein is accessory."""
        return self.name.lower() in ("nef", "vif", "vpr", "vpu")

    def __str__(self) -> str:
        return self.name.upper()

    def __repr__(self) -> str:
        return f"Protein({self.name})"
