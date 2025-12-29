"""
Drug value objects - represent antiretroviral drugs.

Covers the main HIV drug classes: PI, NRTI, NNRTI, INSTI.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DrugClass(Enum):
    """HIV drug classes."""

    PI = "pi"       # Protease Inhibitors
    NRTI = "nrti"   # Nucleoside Reverse Transcriptase Inhibitors
    NNRTI = "nnrti" # Non-Nucleoside Reverse Transcriptase Inhibitors
    INSTI = "insti" # Integrase Strand Transfer Inhibitors
    EI = "ei"       # Entry Inhibitors
    MI = "mi"       # Maturation Inhibitors


# Drug abbreviation to full name mapping
DRUG_NAMES: dict[str, str] = {
    # Protease Inhibitors
    "FPV": "Fosamprenavir",
    "ATV": "Atazanavir",
    "IDV": "Indinavir",
    "LPV": "Lopinavir",
    "NFV": "Nelfinavir",
    "SQV": "Saquinavir",
    "TPV": "Tipranavir",
    "DRV": "Darunavir",
    # NRTIs
    "ABC": "Abacavir",
    "AZT": "Zidovudine",
    "D4T": "Stavudine",
    "DDI": "Didanosine",
    "FTC": "Emtricitabine",
    "3TC": "Lamivudine",
    "TDF": "Tenofovir",
    "TAF": "Tenofovir Alafenamide",
    # NNRTIs
    "DOR": "Doravirine",
    "EFV": "Efavirenz",
    "ETR": "Etravirine",
    "NVP": "Nevirapine",
    "RPV": "Rilpivirine",
    # INSTIs
    "BIC": "Bictegravir",
    "CAB": "Cabotegravir",
    "DTG": "Dolutegravir",
    "EVG": "Elvitegravir",
    "RAL": "Raltegravir",
}

# Drug to class mapping
DRUG_CLASSES: dict[str, DrugClass] = {
    # PI
    "FPV": DrugClass.PI, "ATV": DrugClass.PI, "IDV": DrugClass.PI,
    "LPV": DrugClass.PI, "NFV": DrugClass.PI, "SQV": DrugClass.PI,
    "TPV": DrugClass.PI, "DRV": DrugClass.PI,
    # NRTI
    "ABC": DrugClass.NRTI, "AZT": DrugClass.NRTI, "D4T": DrugClass.NRTI,
    "DDI": DrugClass.NRTI, "FTC": DrugClass.NRTI, "3TC": DrugClass.NRTI,
    "TDF": DrugClass.NRTI, "TAF": DrugClass.NRTI,
    # NNRTI
    "DOR": DrugClass.NNRTI, "EFV": DrugClass.NNRTI, "ETR": DrugClass.NNRTI,
    "NVP": DrugClass.NNRTI, "RPV": DrugClass.NNRTI,
    # INSTI
    "BIC": DrugClass.INSTI, "CAB": DrugClass.INSTI, "DTG": DrugClass.INSTI,
    "EVG": DrugClass.INSTI, "RAL": DrugClass.INSTI,
}


@dataclass(frozen=True, slots=True)
class Drug:
    """
    Represents an antiretroviral drug.

    Attributes:
        abbreviation: Standard abbreviation (e.g., 'DRV')
        name: Full drug name
        drug_class: Drug class (PI, NRTI, etc.)
    """

    abbreviation: str
    name: Optional[str] = None
    drug_class: Optional[DrugClass] = None

    def __post_init__(self) -> None:
        abbrev = self.abbreviation.upper()
        object.__setattr__(self, "abbreviation", abbrev)

        if self.name is None:
            object.__setattr__(self, "name", DRUG_NAMES.get(abbrev, abbrev))

        if self.drug_class is None:
            object.__setattr__(self, "drug_class", DRUG_CLASSES.get(abbrev))

    @classmethod
    def from_abbreviation(cls, abbrev: str) -> "Drug":
        """Create Drug from abbreviation."""
        return cls(abbreviation=abbrev.upper())

    @property
    def target_protein(self) -> str:
        """Get the target protein for this drug."""
        if self.drug_class == DrugClass.PI:
            return "PR"
        elif self.drug_class in (DrugClass.NRTI, DrugClass.NNRTI):
            return "RT"
        elif self.drug_class == DrugClass.INSTI:
            return "IN"
        return "unknown"

    def __str__(self) -> str:
        return self.abbreviation

    def __repr__(self) -> str:
        return f"Drug({self.abbreviation}, {self.drug_class.value if self.drug_class else 'unknown'})"
