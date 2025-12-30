# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Constants and reference data for HIV clinical decision support.

This module contains drug databases, mutation lists, and clinical
reference values used for resistance interpretation and treatment
recommendations.
"""

from __future__ import annotations

# WHO-recommended first-line drugs by class
FIRST_LINE_DRUGS: dict[str, list[str]] = {
    "NRTI": ["TDF", "TAF", "ABC", "3TC", "FTC"],
    "NNRTI": ["EFV", "NVP", "DOR"],
    "INSTI": ["DTG", "RAL", "EVG", "BIC"],
}

# WHO-recommended first-line regimens
FIRST_LINE_REGIMENS: list[dict] = [
    {"name": "TDF/3TC/DTG", "drugs": ["TDF", "3TC", "DTG"], "preferred": True},
    {"name": "TDF/FTC/DTG", "drugs": ["TDF", "FTC", "DTG"], "preferred": True},
    {"name": "TAF/FTC/DTG", "drugs": ["TAF", "FTC", "DTG"], "preferred": False},
    {"name": "TDF/3TC/EFV", "drugs": ["TDF", "3TC", "EFV"], "preferred": False},
    {"name": "ABC/3TC/DTG", "drugs": ["ABC", "3TC", "DTG"], "preferred": False},
]

# Known transmitted drug resistance (TDR) mutations by class
# Based on WHO surveillance drug resistance mutations (SDRM) list
TDR_MUTATIONS: dict[str, dict[str, dict]] = {
    # NRTI mutations
    "NRTI": {
        "M184V": {"drugs": ["3TC", "FTC"], "level": "high", "prevalence": 5.2},
        "M184I": {"drugs": ["3TC", "FTC"], "level": "high", "prevalence": 0.8},
        "K65R": {"drugs": ["TDF", "ABC"], "level": "moderate", "prevalence": 2.1},
        "K70R": {"drugs": ["AZT", "D4T", "TDF"], "level": "moderate", "prevalence": 1.5},
        "K70E": {"drugs": ["TDF", "ABC"], "level": "moderate", "prevalence": 0.3},
        "L74V": {"drugs": ["ABC", "DDI"], "level": "high", "prevalence": 0.6},
        "L74I": {"drugs": ["ABC", "DDI"], "level": "moderate", "prevalence": 0.2},
        "Y115F": {"drugs": ["ABC"], "level": "moderate", "prevalence": 0.4},
        "T215F": {"drugs": ["AZT", "D4T"], "level": "high", "prevalence": 1.8},
        "T215Y": {"drugs": ["AZT", "D4T"], "level": "high", "prevalence": 2.3},
        "K219Q": {"drugs": ["AZT"], "level": "moderate", "prevalence": 0.9},
        "K219E": {"drugs": ["AZT"], "level": "moderate", "prevalence": 0.5},
        "T69ins": {"drugs": ["All NRTIs"], "level": "high", "prevalence": 0.1},
        "Q151M": {"drugs": ["All NRTIs"], "level": "high", "prevalence": 0.1},
    },
    # NNRTI mutations
    "NNRTI": {
        "K103N": {"drugs": ["EFV", "NVP"], "level": "high", "prevalence": 4.8},
        "K103S": {"drugs": ["EFV", "NVP"], "level": "moderate", "prevalence": 0.5},
        "Y181C": {"drugs": ["NVP", "EFV", "RPV"], "level": "high", "prevalence": 2.1},
        "Y181I": {"drugs": ["NVP", "RPV"], "level": "high", "prevalence": 0.3},
        "Y181V": {"drugs": ["NVP", "EFV"], "level": "high", "prevalence": 0.2},
        "G190A": {"drugs": ["NVP", "EFV"], "level": "high", "prevalence": 1.9},
        "G190S": {"drugs": ["NVP", "EFV"], "level": "high", "prevalence": 0.4},
        "K101E": {"drugs": ["NVP", "EFV"], "level": "moderate", "prevalence": 0.8},
        "K101P": {"drugs": ["NVP", "EFV", "RPV"], "level": "high", "prevalence": 0.3},
        "V106A": {"drugs": ["NVP"], "level": "high", "prevalence": 0.6},
        "V106M": {"drugs": ["EFV"], "level": "moderate", "prevalence": 0.3},
        "E138A": {"drugs": ["RPV"], "level": "moderate", "prevalence": 1.2},
        "E138K": {"drugs": ["RPV"], "level": "high", "prevalence": 0.5},
        "V179D": {"drugs": ["EFV", "NVP"], "level": "low", "prevalence": 0.8},
        "Y188L": {"drugs": ["EFV", "NVP"], "level": "high", "prevalence": 0.4},
        "H221Y": {"drugs": ["RPV"], "level": "moderate", "prevalence": 0.3},
    },
    # INSTI mutations (rare in TDR but increasing)
    "INSTI": {
        "N155H": {"drugs": ["RAL", "EVG"], "level": "high", "prevalence": 0.2},
        "Q148H": {"drugs": ["RAL", "EVG", "DTG"], "level": "high", "prevalence": 0.1},
        "Q148R": {"drugs": ["RAL", "EVG", "DTG"], "level": "high", "prevalence": 0.1},
        "Q148K": {"drugs": ["RAL", "EVG"], "level": "high", "prevalence": 0.05},
        "Y143R": {"drugs": ["RAL"], "level": "high", "prevalence": 0.1},
        "Y143C": {"drugs": ["RAL"], "level": "high", "prevalence": 0.05},
        "G140S": {"drugs": ["RAL", "EVG"], "level": "moderate", "prevalence": 0.05},
        "E92Q": {"drugs": ["RAL", "EVG"], "level": "moderate", "prevalence": 0.1},
        "T66I": {"drugs": ["EVG"], "level": "moderate", "prevalence": 0.05},
        "S147G": {"drugs": ["CAB"], "level": "moderate", "prevalence": 0.02},
    },
    # PI mutations
    "PI": {
        "M46I": {"drugs": ["IDV", "NFV", "LPV"], "level": "moderate", "prevalence": 1.5},
        "M46L": {"drugs": ["IDV", "NFV", "LPV"], "level": "moderate", "prevalence": 0.8},
        "I50L": {"drugs": ["ATV"], "level": "high", "prevalence": 0.3},
        "I50V": {"drugs": ["DRV", "LPV"], "level": "moderate", "prevalence": 0.2},
        "I54V": {"drugs": ["All PIs"], "level": "moderate", "prevalence": 0.5},
        "L76V": {"drugs": ["LPV", "DRV"], "level": "moderate", "prevalence": 0.3},
        "V82A": {"drugs": ["IDV", "LPV"], "level": "moderate", "prevalence": 0.8},
        "I84V": {"drugs": ["All PIs"], "level": "high", "prevalence": 0.4},
        "N88S": {"drugs": ["NFV", "ATV"], "level": "high", "prevalence": 0.3},
        "L90M": {"drugs": ["NFV", "SQV"], "level": "high", "prevalence": 1.2},
    },
}

# Long-acting injectable drugs and their resistance profiles
LA_DRUGS: dict[str, dict] = {
    "CAB": {
        "name": "Cabotegravir",
        "class": "INSTI",
        "half_life_days": 25,
        "administration": "IM gluteal",
        "mutations": {
            "Q148H": {"fold_change": 5.0, "level": "high"},
            "Q148R": {"fold_change": 3.0, "level": "moderate"},
            "Q148K": {"fold_change": 4.0, "level": "high"},
            "N155H": {"fold_change": 2.5, "level": "moderate"},
            "G140S": {"fold_change": 2.0, "level": "low"},
            "E138K": {"fold_change": 1.5, "level": "low"},
            "S147G": {"fold_change": 1.8, "level": "low"},
            "G140A": {"fold_change": 1.8, "level": "low"},
        },
    },
    "RPV": {
        "name": "Rilpivirine",
        "class": "NNRTI",
        "half_life_days": 45,
        "administration": "IM gluteal",
        "mutations": {
            "E138K": {"fold_change": 3.0, "level": "high"},
            "E138A": {"fold_change": 2.5, "level": "moderate"},
            "E138G": {"fold_change": 2.0, "level": "moderate"},
            "K101E": {"fold_change": 2.0, "level": "moderate"},
            "K101P": {"fold_change": 4.0, "level": "high"},
            "Y181C": {"fold_change": 5.0, "level": "high"},
            "Y181I": {"fold_change": 4.0, "level": "high"},
            "Y181V": {"fold_change": 3.5, "level": "high"},
            "H221Y": {"fold_change": 2.5, "level": "moderate"},
            "F227C": {"fold_change": 3.0, "level": "high"},
            "M230L": {"fold_change": 2.0, "level": "moderate"},
        },
    },
}

# BMI impact on pharmacokinetics
BMI_CATEGORIES: dict[str, dict] = {
    "underweight": {"range": (0, 18.5), "pk_adjustment": 1.15},
    "normal": {"range": (18.5, 25), "pk_adjustment": 1.0},
    "overweight": {"range": (25, 30), "pk_adjustment": 0.95},
    "obese_1": {"range": (30, 35), "pk_adjustment": 0.85},
    "obese_2": {"range": (35, 40), "pk_adjustment": 0.75},
    "obese_3": {"range": (40, 100), "pk_adjustment": 0.60},
}

# WHO Surveillance Drug Resistance Mutations (SDRM)
# Used to identify transmitted drug resistance (TDR)

WHO_SDRM_NRTI: set[str] = {
    "M41L",
    "K65R",
    "K65N",
    "D67N",
    "D67G",
    "D67E",
    "T69ins",
    "K70R",
    "K70E",
    "L74V",
    "L74I",
    "Y115F",
    "Q151M",
    "M184V",
    "M184I",
    "L210W",
    "T215Y",
    "T215F",
    "T215I",
    "T215S",
    "T215C",
    "T215D",
    "T215E",
    "K219Q",
    "K219E",
    "K219N",
    "K219R",
}

WHO_SDRM_NNRTI: set[str] = {
    "L100I",
    "K101E",
    "K101P",
    "K103N",
    "K103S",
    "V106M",
    "V106A",
    "E138A",
    "E138G",
    "E138K",
    "E138Q",
    "E138R",
    "V179L",
    "Y181C",
    "Y181I",
    "Y181V",
    "Y188L",
    "Y188C",
    "Y188H",
    "G190A",
    "G190S",
    "G190E",
    "H221Y",
    "P225H",
    "F227C",
    "M230I",
    "M230L",
}

WHO_SDRM_INSTI: set[str] = {
    "T66I",
    "T66A",
    "T66K",
    "E92Q",
    "E92G",
    "G118R",
    "F121Y",
    "G140S",
    "G140A",
    "G140C",
    "Y143R",
    "Y143C",
    "Y143H",
    "S147G",
    "Q148H",
    "Q148K",
    "Q148R",
    "N155H",
    "N155S",
    "R263K",
}

WHO_SDRM_PI: set[str] = {
    "D30N",
    "V32I",
    "M46I",
    "M46L",
    "I47V",
    "I47A",
    "G48V",
    "G48M",
    "I50L",
    "I50V",
    "I54V",
    "I54L",
    "I54M",
    "L76V",
    "V82A",
    "V82F",
    "V82T",
    "V82S",
    "V82L",
    "I84V",
    "N88S",
    "L90M",
}

# Drug class descriptions
HIV_DRUG_CLASSES: dict[str, str] = {
    "NRTI": "Nucleoside/Nucleotide Reverse Transcriptase Inhibitor",
    "NNRTI": "Non-Nucleoside Reverse Transcriptase Inhibitor",
    "PI": "Protease Inhibitor",
    "INSTI": "Integrase Strand Transfer Inhibitor",
    "EI": "Entry Inhibitor",
    "FI": "Fusion Inhibitor",
}

# Drug full names
DRUG_NAMES: dict[str, str] = {
    # NRTIs
    "3TC": "Lamivudine",
    "ABC": "Abacavir",
    "AZT": "Zidovudine",
    "D4T": "Stavudine",
    "DDI": "Didanosine",
    "FTC": "Emtricitabine",
    "TAF": "Tenofovir alafenamide",
    "TDF": "Tenofovir disoproxil fumarate",
    # NNRTIs
    "DOR": "Doravirine",
    "EFV": "Efavirenz",
    "ETR": "Etravirine",
    "NVP": "Nevirapine",
    "RPV": "Rilpivirine",
    # PIs
    "ATV": "Atazanavir",
    "DRV": "Darunavir",
    "LPV": "Lopinavir",
    "NFV": "Nelfinavir",
    "SQV": "Saquinavir",
    # INSTIs
    "BIC": "Bictegravir",
    "CAB": "Cabotegravir",
    "DTG": "Dolutegravir",
    "EVG": "Elvitegravir",
    "RAL": "Raltegravir",
}

# Reference sequence positions (HXB2)
HXB2_RT_LENGTH = 560
HXB2_PR_LENGTH = 99
HXB2_IN_LENGTH = 288
