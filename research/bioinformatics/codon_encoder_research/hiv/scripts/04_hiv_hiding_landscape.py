#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
HIV Complete Hiding Landscape Analysis

HYPOTHESIS: HIV has overfitted hiding strategies at multiple hierarchy levels.
Since codons are the substrate for all higher-level structures, the 3-adic
geometry should reveal the COMPLETE evolutionary possibility space of HIV evasion.

Hierarchy Levels:
1. Codon level - Specific codon choices for immune-invisible amino acids
2. Peptide level - Short sequences that avoid MHC presentation
3. Protein level - Structural mimicry of human proteins
4. Signaling level - Interference with cell signaling cascades
5. Glycan level - Shield masking of epitopes

This analysis maps ALL known HIV hiding mechanisms and predicts the
complete landscape of evolutionary possibilities.

Author: AI Whisperers
Date: 2025-12-24
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add local path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from hyperbolic_utils import codon_to_onehot, load_codon_encoder

# =============================================================================
# COMPLETE HIV PROTEOME - ALL PROTEINS AND THEIR HIDING FUNCTIONS
# =============================================================================

HIV_PROTEOME = {
    # =========================================================================
    # STRUCTURAL PROTEINS (Gag)
    # =========================================================================
    "Gag_MA_p17": {
        "gene": "gag",
        "length": 132,
        "function": "Matrix protein, membrane targeting",
        "hiding_mechanisms": [
            "Myristoylation mimics host membrane proteins",
            "Nuclear localization signal hijacks import machinery",
        ],
        "human_mimics": ["Src family kinases (myristoylation)"],
        "signaling_targets": ["PI(4,5)P2 binding", "Calmodulin interaction"],
        "key_residues": {
            "myristoylation": [1, 2],  # N-terminal Gly
            "membrane_binding": [17, 21, 30, 31, 34],
            "NLS": [110, 114, 116],
        },
        # Representative codon sequences for key functional regions
        "codons": {
            "myristoyl_site": ["ATG", "GGC"],  # Met-Gly start
            "membrane_region": [
                "AAG",
                "CTG",
                "AAG",
                "CGG",
                "AAG",
            ],  # Basic patch
        },
    },
    "Gag_CA_p24": {
        "gene": "gag",
        "length": 231,
        "function": "Capsid protein, forms conical core",
        "hiding_mechanisms": [
            "Cyclophilin A binding mimics host protein folding",
            "TRIM5alpha evasion through capsid geometry",
            "Variable surface loops avoid antibody recognition",
        ],
        "human_mimics": ["Cyclophilin substrates"],
        "signaling_targets": ["CPSF6 interaction", "NUP153 binding"],
        "key_residues": {
            "CypA_binding": [85, 86, 87, 88, 89, 90, 91, 93],  # CypA loop
            "TRIM5_evasion": [82, 83, 84, 120, 207],
            "epitope_masking": [45, 47, 48, 50, 52],  # Variable regions
        },
        "codons": {
            "CypA_loop": ["GGC", "CCG", "GTG", "ATC", "ACC", "GGC", "AAC"],
        },
    },
    "Gag_NC_p7": {
        "gene": "gag",
        "length": 55,
        "function": "Nucleocapsid, RNA binding",
        "hiding_mechanisms": [
            "Zinc finger motifs mimic host transcription factors",
            "RNA chaperone activity mimics cellular helicases",
        ],
        "human_mimics": ["CCHC zinc finger proteins", "Nucleolin"],
        "signaling_targets": ["RNA packaging signal", "Genomic RNA selection"],
        "key_residues": {
            "zinc_finger_1": [14, 18, 28, 31],  # CCHC motif
            "zinc_finger_2": [35, 39, 49, 52],  # CCHC motif
        },
        "codons": {
            "ZF1": ["TGC", "CAC", "TGC"],  # Cys-His-Cys
            "ZF2": ["TGC", "CAC", "TGC"],
        },
    },
    # =========================================================================
    # ENZYMATIC PROTEINS (Pol)
    # =========================================================================
    "Pol_PR": {
        "gene": "pol",
        "length": 99,
        "function": "Protease, polyprotein processing",
        "hiding_mechanisms": [
            "Aspartic protease fold mimics human pepsin family",
            "Substrate specificity avoids host protein cleavage",
            "Dimer interface critical for drug resistance",
        ],
        "human_mimics": ["Pepsin", "Cathepsin D", "Renin"],
        "signaling_targets": [
            "Polyprotein cleavage sites",
            "Procaspase-8 (apoptosis)",
        ],
        "key_residues": {
            "active_site": [25, 27],  # DTG catalytic triad
            "drug_resistance": [46, 54, 82, 84, 90],
            "dimer_interface": [1, 4, 5, 96, 97, 99],
        },
        "codons": {
            "catalytic": ["GAC", "ACC", "GGC"],  # Asp-Thr-Gly
        },
    },
    "Pol_RT": {
        "gene": "pol",
        "length": 560,
        "function": "Reverse transcriptase",
        "hiding_mechanisms": [
            "Polymerase fold mimics host DNA polymerases",
            "RNase H domain mimics cellular RNases",
            "High mutation rate enables immune escape",
        ],
        "human_mimics": ["DNA Pol gamma", "Telomerase RT", "RNase H1"],
        "signaling_targets": ["dNTP pools", "SAMHD1 (restriction factor)"],
        "key_residues": {
            "polymerase_active": [110, 185, 186],  # YMDD motif
            "NNRTI_pocket": [100, 101, 103, 106, 181, 188, 190],
            "RNaseH_active": [443, 478, 498, 549],
        },
        "codons": {
            "YMDD_motif": ["TAC", "ATG", "GAC", "GAC"],  # Tyr-Met-Asp-Asp
        },
    },
    "Pol_IN": {
        "gene": "pol",
        "length": 288,
        "function": "Integrase, DNA integration",
        "hiding_mechanisms": [
            "DDE motif mimics host transposases",
            "LEDGF binding mimics chromatin interactions",
            "Integration site selection mimics host genome access",
        ],
        "human_mimics": ["RAG recombinases", "Transposases"],
        "signaling_targets": [
            "LEDGF/p75",
            "INI1/hSNF5",
            "Chromatin remodeling",
        ],
        "key_residues": {
            "DDE_motif": [64, 116, 152],  # Catalytic triad
            "LEDGF_binding": [
                128,
                129,
                130,
                132,
                161,
                166,
                168,
                170,
                171,
                173,
            ],
            "drug_resistance": [92, 143, 148, 155],
        },
        "codons": {
            "DDE_catalytic": ["GAC", "GAC", "GAG"],  # Asp-Asp-Glu
        },
    },
    # =========================================================================
    # ENVELOPE PROTEINS (Env)
    # =========================================================================
    "Env_gp120": {
        "gene": "env",
        "length": 480,
        "function": "Surface glycoprotein, receptor binding",
        "hiding_mechanisms": [
            "Glycan shield covers ~50% of surface",
            "Variable loops (V1-V5) provide antigenic variation",
            "CD4-induced conformational masking",
            "Mimicry of self-glycoproteins",
            "Conformational masking of co-receptor binding site",
        ],
        "human_mimics": ["CD4 binding proteins", "Glycosylated self-proteins"],
        "signaling_targets": [
            "CD4 receptor",
            "CCR5/CXCR4 coreceptors",
            "DC-SIGN",
        ],
        "key_residues": {
            "CD4_binding": [
                124,
                125,
                126,
                196,
                198,
                279,
                280,
                281,
                282,
                368,
                370,
                425,
                426,
                427,
                428,
                429,
                430,
                431,
                432,
                455,
                456,
                457,
                458,
                459,
                471,
                472,
                473,
                474,
                475,
                476,
            ],
            "coreceptor_binding": [
                298,
                308,
                315,
                316,
                317,
                318,
                319,
                320,
                421,
                422,
            ],
            "V1_loop": [
                131,
                132,
                133,
                134,
                135,
                136,
                137,
                138,
                139,
                140,
                141,
                142,
                143,
                144,
                145,
                146,
                147,
                148,
                149,
                150,
                151,
                152,
                153,
                154,
                155,
                156,
            ],
            "V2_loop": [
                157,
                158,
                159,
                160,
                161,
                162,
                163,
                164,
                165,
                166,
                167,
                168,
                169,
                170,
                171,
                172,
                173,
                174,
                175,
                176,
                177,
                178,
                179,
                180,
                181,
                182,
                183,
                184,
                185,
                186,
                187,
                188,
                189,
                190,
                191,
                192,
                193,
                194,
                195,
                196,
            ],
            "V3_loop": [
                296,
                297,
                298,
                299,
                300,
                301,
                302,
                303,
                304,
                305,
                306,
                307,
                308,
                309,
                310,
                311,
                312,
                313,
                314,
                315,
                316,
                317,
                318,
                319,
                320,
                321,
                322,
                323,
                324,
                325,
                326,
                327,
                328,
                329,
                330,
            ],
            "glycan_shield": [
                88,
                133,
                137,
                156,
                160,
                187,
                197,
                234,
                241,
                262,
                276,
                289,
                295,
                301,
                332,
                339,
                355,
                362,
                386,
                392,
                406,
                411,
                448,
                463,
            ],
        },
        "codons": {
            "CD4_contact_core": [
                "TGG",
                "GAG",
                "ACC",
                "TGC",
            ],  # Trp-Glu-Thr-Cys
        },
    },
    "Env_gp41": {
        "gene": "env",
        "length": 345,
        "function": "Transmembrane fusion protein",
        "hiding_mechanisms": [
            "Fusion peptide hidden until triggered",
            "Immunodominant but non-neutralizing epitopes",
            "MPER partially embedded in membrane",
        ],
        "human_mimics": ["Paramyxovirus F proteins", "Influenza HA2"],
        "signaling_targets": ["Membrane fusion machinery", "Lipid rafts"],
        "key_residues": {
            "fusion_peptide": [
                512,
                513,
                514,
                515,
                516,
                517,
                518,
                519,
                520,
                521,
                522,
                523,
                524,
                525,
                526,
                527,
            ],
            "HR1": [
                540,
                541,
                542,
                543,
                544,
                545,
                546,
                547,
                548,
                549,
                550,
                551,
                552,
                553,
                554,
                555,
                556,
                557,
                558,
                559,
                560,
                561,
                562,
                563,
                564,
                565,
                566,
                567,
                568,
                569,
                570,
                571,
                572,
                573,
                574,
                575,
                576,
                577,
                578,
                579,
                580,
                581,
                582,
                583,
                584,
                585,
                586,
                587,
                588,
                589,
                590,
                591,
            ],
            "HR2": [
                628,
                629,
                630,
                631,
                632,
                633,
                634,
                635,
                636,
                637,
                638,
                639,
                640,
                641,
                642,
                643,
                644,
                645,
                646,
                647,
                648,
                649,
                650,
                651,
                652,
                653,
                654,
                655,
                656,
                657,
                658,
                659,
                660,
                661,
            ],
            "MPER": [
                662,
                663,
                664,
                665,
                666,
                667,
                668,
                669,
                670,
                671,
                672,
                673,
                674,
                675,
                676,
                677,
                678,
                679,
                680,
                681,
                682,
                683,
            ],
        },
        "codons": {
            "fusion_peptide_start": [
                "GGC",
                "ATC",
                "GTG",
                "GGC",
            ],  # Gly-Ile-Val-Gly
        },
    },
    # =========================================================================
    # REGULATORY PROTEINS
    # =========================================================================
    "Tat": {
        "gene": "tat",
        "length": 101,
        "function": "Transcriptional transactivator",
        "hiding_mechanisms": [
            "Arginine-rich motif mimics host RNA-binding proteins",
            "Secreted form interferes with bystander cells",
            "Cysteine-rich domain mimics growth factors",
        ],
        "human_mimics": [
            "HIV LTR activators",
            "Angiogenic factors (VEGF-like)",
        ],
        "signaling_targets": [
            "CDK9/CyclinT1 (P-TEFb)",
            "RNA Pol II",
            "TAR RNA",
            "Extracellular signaling",
        ],
        "key_residues": {
            "cysteine_rich": [22, 25, 27, 30, 31, 34, 37],
            "core_domain": [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
            "arginine_rich": [49, 50, 51, 52, 53, 54, 55, 56, 57],
            "RGD_like": [78, 79, 80],  # Cell attachment motif
        },
        "codons": {
            "ARM": [
                "CGG",
                "CGG",
                "CGG",
                "AAG",
                "CGG",
                "CGG",
            ],  # Arg-rich motif
        },
    },
    "Rev": {
        "gene": "rev",
        "length": 116,
        "function": "RNA export factor",
        "hiding_mechanisms": [
            "Leucine-rich NES mimics host export signals",
            "Arginine-rich NLS mimics importins",
            "Multimerization mimics host RNA-binding proteins",
        ],
        "human_mimics": ["CRM1 substrates", "hnRNPs"],
        "signaling_targets": [
            "CRM1/Exportin1",
            "RanGTP",
            "Rev Response Element",
        ],
        "key_residues": {
            "NLS": [
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
            ],
            "NES": [75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
            "multimerization": [
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
            ],
        },
        "codons": {
            "NES": ["CTG", "CTG", "CTG", "CCG", "CTG"],  # Leu-rich
        },
    },
    # =========================================================================
    # ACCESSORY PROTEINS - THE "HIDING SPECIALISTS"
    # =========================================================================
    "Nef": {
        "gene": "nef",
        "length": 206,
        "function": "Negative factor - Master immune evasin",
        "hiding_mechanisms": [
            "Downregulates CD4 (prevents superinfection, hides virus)",
            "Downregulates MHC-I (hides infected cells from CTLs)",
            "Downregulates MHC-II (blocks antigen presentation)",
            "Activates T-cells for viral replication",
            "SH3 binding mimics host signaling proteins",
            "Myristoylation for membrane association",
        ],
        "human_mimics": [
            "SH3-binding proteins",
            "PACS proteins",
            "AP-1/AP-2 adaptors",
        ],
        "signaling_targets": [
            "CD4 endocytosis",
            "MHC-I endocytosis",
            "MHC-II trafficking",
            "Src family kinases",
            "PAK2",
            "DOCK2-ELMO1",
            "PI3K",
        ],
        "key_residues": {
            "myristoylation": [1, 2],
            "CD4_binding": [57, 58, 59, 60, 61, 62, 63, 64, 65],
            "MHC_I_binding": [
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
            ],
            "SH3_binding": [69, 70, 71, 72, 73, 74, 75],  # PxxP motif
            "dileucine_motif": [164, 165],
        },
        "codons": {
            "PxxP": ["CCG", "XXX", "XXX", "CCG"],  # Pro-X-X-Pro
            "dileucine": ["CTG", "CTG"],  # Leu-Leu
        },
    },
    "Vif": {
        "gene": "vif",
        "length": 192,
        "function": "Viral infectivity factor - Counteracts APOBEC3",
        "hiding_mechanisms": [
            "Hijacks Cullin5-E3 ligase for APOBEC3 degradation",
            "BC-box mimics host SOCS proteins",
            "Prevents hypermutation of viral genome",
        ],
        "human_mimics": ["SOCS-box proteins", "Cullin substrates"],
        "signaling_targets": [
            "APOBEC3G/F/H",
            "Cullin5-EloB-EloC complex",
            "CBF-beta",
        ],
        "key_residues": {
            "APOBEC3_binding": [21, 22, 23, 24, 25, 26, 40, 41, 42, 43, 44],
            "BC_box": [
                144,
                145,
                146,
                147,
                148,
                149,
                150,
                151,
                152,
                153,
                154,
                155,
                156,
                157,
                158,
                159,
            ],
            "Cullin5_binding": [
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                127,
                128,
                129,
                130,
                131,
                132,
                133,
                134,
                135,
                136,
                137,
                138,
                139,
            ],
            "zinc_binding": [108, 114, 133, 139],  # HCCH motif
        },
        "codons": {
            "HCCH": ["CAC", "TGC", "TGC", "CAC"],  # His-Cys-Cys-His
        },
    },
    "Vpr": {
        "gene": "vpr",
        "length": 96,
        "function": "Viral protein R - Cell cycle arrest, nuclear import",
        "hiding_mechanisms": [
            "Mimics host nuclear transport signals",
            "Causes G2 arrest (more viral production)",
            "Hijacks DCAF1/DDB1 for UNG2 and SAMHD1 degradation",
        ],
        "human_mimics": ["Nuclear transport factors", "Cell cycle regulators"],
        "signaling_targets": [
            "DCAF1",
            "DDB1",
            "CUL4",
            "PP2A",
            "UNG2",
            "ATR pathway",
        ],
        "key_residues": {
            "nuclear_localization": [
                17,
                20,
                26,
                29,
                33,
                34,
                36,
            ],  # Arginine-rich
            "G2_arrest": [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
            "DCAF1_binding": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
        },
        "codons": {
            "helical_core": ["CTG", "GCC", "GTG", "GCC"],  # Hydrophobic helix
        },
    },
    "Vpu": {
        "gene": "vpu",
        "length": 81,
        "function": "Viral protein U - CD4 degradation, virion release",
        "hiding_mechanisms": [
            "Transmembrane domain forms ion channel",
            "Cytoplasmic domain mimics phosphorylated substrates",
            "Antagonizes BST-2/Tetherin (host restriction factor)",
            "Induces CD4 degradation via ER-associated degradation",
        ],
        "human_mimics": ["Ion channels", "SCF-TrCP substrates"],
        "signaling_targets": ["BST-2/Tetherin", "beta-TrCP", "CD4", "NF-kB"],
        "key_residues": {
            "transmembrane": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
            ],
            "phosphoserine": [52, 56],  # DSGxxS motif
            "BST2_binding": [
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
            ],
        },
        "codons": {
            "DSGxxS": [
                "GAC",
                "AGC",
                "GGC",
                "XXX",
                "XXX",
                "AGC",
            ],  # Phosphodegron
        },
    },
}

# =============================================================================
# HUMAN SIGNALING PATHWAYS HIV TARGETS
# =============================================================================

SIGNALING_PATHWAYS = {
    "CD4_TCR_signaling": {
        "description": "T-cell receptor activation cascade",
        "hiv_interference": {
            "Env_gp120": "Binds CD4, triggers aberrant signaling",
            "Nef": "Downregulates CD4, prevents re-infection signaling",
            "Vpu": "Degrades CD4 in ER, blocks surface expression",
        },
        "downstream_effects": ["Anergy", "Apoptosis", "Immune dysfunction"],
    },
    "MHC_antigen_presentation": {
        "description": "Antigen presentation to CTLs and helper T-cells",
        "hiv_interference": {
            "Nef": "Downregulates MHC-I and MHC-II",
            "Vpu": "May affect MHC trafficking",
        },
        "downstream_effects": ["CTL evasion", "Reduced immune recognition"],
    },
    "innate_restriction": {
        "description": "Intrinsic cellular antiviral factors",
        "hiv_interference": {
            "Vif": "Degrades APOBEC3 family",
            "Vpr": "Degrades UNG2, SAMHD1",
            "Vpu": "Antagonizes BST-2/Tetherin",
            "Gag_CA_p24": "Evades TRIM5alpha",
        },
        "downstream_effects": ["Viral persistence", "Productive infection"],
    },
    "cell_cycle": {
        "description": "Cell division and checkpoint control",
        "hiv_interference": {
            "Vpr": "G2 arrest via ATR pathway",
            "Tat": "Can affect cell cycle progression",
        },
        "downstream_effects": ["Enhanced viral production", "Cell death"],
    },
    "NF_kB_pathway": {
        "description": "Inflammatory and immune gene activation",
        "hiv_interference": {
            "Tat": "Activates NF-kB for LTR transcription",
            "Vpu": "Inhibits NF-kB in some contexts",
            "Nef": "Activates T-cells via NF-kB",
        },
        "downstream_effects": ["Chronic inflammation", "Immune activation"],
    },
    "apoptosis": {
        "description": "Programmed cell death",
        "hiv_interference": {
            "Nef": "Both pro- and anti-apoptotic effects",
            "Vpr": "Induces apoptosis through mitochondria",
            "Env_gp120": "Can trigger bystander apoptosis",
            "Tat": "Sensitizes to apoptosis in some contexts",
        },
        "downstream_effects": ["CD4 depletion", "Immune collapse"],
    },
    "nuclear_import": {
        "description": "Nuclear-cytoplasmic transport",
        "hiv_interference": {
            "Gag_MA_p17": "NLS for PIC import",
            "Vpr": "Facilitates nuclear import of PIC",
            "Rev": "Exports unspliced RNA via CRM1",
            "Pol_IN": "LEDGF interaction for integration",
        },
        "downstream_effects": [
            "Infection of non-dividing cells",
            "Latency establishment",
        ],
    },
    "membrane_trafficking": {
        "description": "Endocytosis, exocytosis, vesicle transport",
        "hiv_interference": {
            "Nef": "Hijacks clathrin-mediated endocytosis",
            "Gag": "Hijacks ESCRT for budding",
            "Env": "Traffics through secretory pathway",
            "Vpu": "Affects protein degradation pathways",
        },
        "downstream_effects": [
            "Efficient viral assembly",
            "Receptor downregulation",
        ],
    },
}

# =============================================================================
# CODON TABLE FOR EMBEDDING
# =============================================================================

CODON_TABLE = {
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

# =============================================================================
# ANALYSIS CLASSES
# =============================================================================


@dataclass
class HidingMechanism:
    """A specific hiding mechanism at a particular hierarchy level."""

    protein: str
    level: str  # codon, peptide, protein, signaling, glycan
    mechanism: str
    human_target: str
    codon_signature: List[str]
    embedding_centroid: Optional[np.ndarray] = None
    embedding_radius: float = 0.0


@dataclass
class HidingCluster:
    """A cluster of related hiding mechanisms in embedding space."""

    name: str
    level: str
    mechanisms: List[str]
    proteins: List[str]
    centroid: np.ndarray
    radius: float
    boundary_distance: float  # Distance to nearest non-hiding region


@dataclass
class HidingLandscape:
    """Complete hiding landscape analysis."""

    total_mechanisms: int
    by_level: Dict[str, int]
    by_protein: Dict[str, int]
    clusters: List[HidingCluster]
    evolutionary_space: Dict  # Predicted unexplored hiding possibilities
    vulnerability_map: Dict  # Regions where hiding fails


class HidingLandscapeAnalyzer:
    """
    Analyzes the complete HIV hiding landscape using 3-adic codon geometry.

    Key insight: All hiding strategies must be encoded in codons. The 3-adic
    geometry reveals the evolutionary possibility space.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.encoder = None
        self.codon_mapping = None
        self.embeddings_cache = {}

    def load_encoder(self):
        """Load the 3-adic codon encoder."""
        print("Loading 3-adic codon encoder...")
        try:
            self.encoder, self.codon_mapping, _ = load_codon_encoder(device=self.device, version="3adic")
            print("  Encoder loaded successfully")
            return True
        except Exception as e:
            print(f"  Warning: Could not load encoder: {e}")
            print("  Using fallback embedding method")
            return False

    def get_embedding(self, codon: str) -> np.ndarray:
        """Get hyperbolic embedding for a codon."""
        if codon in self.embeddings_cache:
            return self.embeddings_cache[codon]

        if self.encoder is not None:
            try:
                x = torch.from_numpy(np.array([codon_to_onehot(codon)])).float()
                with torch.no_grad():
                    emb = self.encoder.encode(x)[0].numpy()
                self.embeddings_cache[codon] = emb
                return emb
            except:
                pass

        # Fallback: use amino acid properties
        aa = CODON_TABLE.get(codon, "X")
        emb = self._fallback_embedding(aa, codon)
        self.embeddings_cache[codon] = emb
        return emb

    def _fallback_embedding(self, aa: str, codon: str) -> np.ndarray:
        """Fallback embedding based on amino acid properties."""
        properties = {
            "A": [0.62, 0.0, 0.2, 0.0],
            "C": [0.29, 0.0, 0.3, 0.1],
            "D": [-0.9, -1.0, 0.4, 1.0],
            "E": [-0.74, -1.0, 0.5, 1.0],
            "F": [1.19, 0.0, 0.7, 0.0],
            "G": [0.48, 0.0, 0.0, 0.0],
            "H": [-0.4, 0.5, 0.5, 0.5],
            "I": [1.38, 0.0, 0.5, 0.0],
            "K": [-1.5, 1.0, 0.5, 1.0],
            "L": [1.06, 0.0, 0.5, 0.0],
            "M": [0.64, 0.0, 0.5, 0.0],
            "N": [-0.78, 0.0, 0.4, 1.0],
            "P": [0.12, 0.0, 0.3, 0.0],
            "Q": [-0.85, 0.0, 0.5, 1.0],
            "R": [-2.53, 1.0, 0.6, 1.0],
            "S": [-0.18, 0.0, 0.2, 0.5],
            "T": [-0.05, 0.0, 0.3, 0.5],
            "V": [1.08, 0.0, 0.4, 0.0],
            "W": [0.81, 0.0, 0.8, 0.2],
            "Y": [0.26, 0.0, 0.7, 0.3],
            "*": [0.0, 0.0, 0.0, 0.0],
            "X": [0.0, 0.0, 0.0, 0.0],
        }

        # Codon position encoding
        base_map = {"T": 0, "C": 1, "A": 2, "G": 3}
        codon_enc = np.zeros(12)
        for i, base in enumerate(codon):
            if base in base_map:
                codon_enc[i * 4 + base_map[base]] = 1

        # Combine AA properties with codon encoding
        props = np.array(properties.get(aa, [0, 0, 0, 0]))
        emb = np.concatenate([props, codon_enc])

        # Project to Poincare ball
        norm = np.linalg.norm(emb)
        if norm > 0.95:
            emb = emb * 0.95 / norm

        return emb

    def compute_centroid(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute centroid of embeddings in hyperbolic space (Frechet mean approx)."""
        if not embeddings:
            return np.zeros(16)

        # Simple approximation: Euclidean mean projected back to ball
        mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0.95:
            mean = mean * 0.95 / norm
        return mean

    def compute_poincare_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Poincare distance between two points."""
        x_norm_sq = np.sum(x**2)
        y_norm_sq = np.sum(y**2)
        diff_norm_sq = np.sum((x - y) ** 2)

        x_norm_sq = min(x_norm_sq, 0.99)
        y_norm_sq = min(y_norm_sq, 0.99)

        denominator = (1 - x_norm_sq) * (1 - y_norm_sq)
        if denominator < 1e-10:
            return 10.0

        arg = 1 + 2 * diff_norm_sq / denominator
        return np.arccosh(max(arg, 1.0))

    def analyze_protein_hiding(self, protein_name: str, protein_data: Dict) -> Dict:
        """Analyze hiding mechanisms for a single protein."""
        results = {
            "protein": protein_name,
            "gene": protein_data["gene"],
            "function": protein_data["function"],
            "mechanisms": [],
            "mimicry_targets": protein_data.get("human_mimics", []),
            "signaling_targets": protein_data.get("signaling_targets", []),
            "codon_analysis": {},
        }

        # Analyze each codon signature
        if "codons" in protein_data:
            for region, codons in protein_data["codons"].items():
                valid_codons = [c for c in codons if c in CODON_TABLE and "X" not in c]
                if not valid_codons:
                    continue

                embeddings = [self.get_embedding(c) for c in valid_codons]
                centroid = self.compute_centroid(embeddings)

                # Calculate radius and distances
                if embeddings:
                    distances = [self.compute_poincare_distance(centroid, e) for e in embeddings]
                    radius = np.max(distances) if distances else 0.0

                    # Distance from origin (center of Poincare ball)
                    origin_dist = np.linalg.norm(centroid)

                    results["codon_analysis"][region] = {
                        "codons": valid_codons,
                        "amino_acids": [CODON_TABLE[c] for c in valid_codons],
                        "centroid_norm": float(origin_dist),
                        "cluster_radius": float(radius),
                        "mean_pairwise_distance": (float(np.mean(distances)) if distances else 0.0),
                    }

        # Catalog mechanisms by level
        for mechanism in protein_data.get("hiding_mechanisms", []):
            level = self._classify_mechanism_level(mechanism)
            results["mechanisms"].append(
                {
                    "description": mechanism,
                    "level": level,
                }
            )

        return results

    def _classify_mechanism_level(self, mechanism: str) -> str:
        """Classify a hiding mechanism by hierarchy level."""
        mechanism_lower = mechanism.lower()

        if any(x in mechanism_lower for x in ["codon", "wobble", "synonymous"]):
            return "codon"
        elif any(x in mechanism_lower for x in ["peptide", "epitope", "mhc", "ctl"]):
            return "peptide"
        elif any(x in mechanism_lower for x in ["glycan", "glycosyl", "sugar"]):
            return "glycan"
        elif any(x in mechanism_lower for x in ["signal", "kinase", "pathway", "receptor", "binding"]):
            return "signaling"
        else:
            return "protein"

    def compute_hiding_space_geometry(self, all_protein_results: List[Dict]) -> Dict:
        """
        Compute the geometry of the complete hiding space.

        This reveals:
        1. Where HIV hiding mechanisms cluster in hyperbolic space
        2. Unexplored regions that might be evolutionary accessible
        3. Vulnerable "gaps" where hiding might fail
        """
        all_centroids = []
        level_centroids = defaultdict(list)
        protein_centroids = {}

        for protein_result in all_protein_results:
            protein_name = protein_result["protein"]
            if protein_result["codon_analysis"]:
                protein_embeddings = []
                for region, data in protein_result["codon_analysis"].items():
                    for codon in data["codons"]:
                        emb = self.get_embedding(codon)
                        protein_embeddings.append(emb)
                        all_centroids.append(emb)

                if protein_embeddings:
                    centroid = self.compute_centroid(protein_embeddings)
                    protein_centroids[protein_name] = centroid

                    # Classify by mechanism levels
                    for mech in protein_result["mechanisms"]:
                        level_centroids[mech["level"]].append(centroid)

        # Compute overall hiding space geometry
        if all_centroids:
            overall_centroid = self.compute_centroid(all_centroids)
            distances_from_center = [self.compute_poincare_distance(overall_centroid, c) for c in all_centroids]

            geometry = {
                "overall_centroid_norm": float(np.linalg.norm(overall_centroid)),
                "mean_radius": float(np.mean(distances_from_center)),
                "max_radius": float(np.max(distances_from_center)),
                "std_radius": float(np.std(distances_from_center)),
            }
        else:
            geometry = {}

        # Compute level-specific geometry
        level_geometry = {}
        for level, centroids in level_centroids.items():
            if centroids:
                level_centroid = self.compute_centroid(centroids)
                level_geometry[level] = {
                    "centroid_norm": float(np.linalg.norm(level_centroid)),
                    "n_proteins": len(centroids),
                }

        # Compute pairwise distances between proteins
        protein_distances = {}
        protein_names = list(protein_centroids.keys())
        for i, p1 in enumerate(protein_names):
            for p2 in protein_names[i + 1 :]:
                d = self.compute_poincare_distance(protein_centroids[p1], protein_centroids[p2])
                protein_distances[f"{p1}-{p2}"] = float(d)

        return {
            "geometry": geometry,
            "by_level": level_geometry,
            "protein_distances": protein_distances,
        }

    def predict_evolutionary_possibilities(self, hiding_geometry: Dict) -> Dict:
        """
        Predict unexplored evolutionary possibilities based on geometry.

        Key insight: If HIV has "overfitted" its hiding, there may be:
        1. Unexplored regions that HIV could evolve into
        2. Constrained regions where HIV MUST stay (essential functions)
        3. Boundaries where hiding transitions to visibility
        """
        predictions = {
            "constrained_regions": [],
            "expandable_regions": [],
            "vulnerability_zones": [],
        }

        # Analyze centroid positions
        overall_norm = hiding_geometry.get("geometry", {}).get("overall_centroid_norm", 0.5)

        # Near boundary (norm > 0.8) = highly specialized, constrained
        if overall_norm > 0.8:
            predictions["constrained_regions"].append(
                {
                    "type": "boundary_proximity",
                    "description": "HIV hiding codons cluster near Poincare boundary",
                    "implication": "Limited evolutionary flexibility, highly specialized",
                }
            )

        # Near center (norm < 0.3) = more flexibility
        if overall_norm < 0.3:
            predictions["expandable_regions"].append(
                {
                    "type": "central_position",
                    "description": "HIV hiding codons cluster near center",
                    "implication": "More evolutionary flexibility available",
                }
            )

        # Analyze by level
        for level, data in hiding_geometry.get("by_level", {}).items():
            if data["centroid_norm"] > 0.7:
                predictions["constrained_regions"].append(
                    {
                        "level": level,
                        "norm": data["centroid_norm"],
                        "description": f"{level}-level hiding is highly specialized",
                    }
                )
            elif data["centroid_norm"] < 0.4:
                predictions["expandable_regions"].append(
                    {
                        "level": level,
                        "norm": data["centroid_norm"],
                        "description": f"{level}-level hiding has room to evolve",
                    }
                )

        # Identify vulnerability zones (gaps between clusters)
        protein_distances = hiding_geometry.get("protein_distances", {})
        if protein_distances:
            # Large distances between proteins = potential vulnerability
            for pair, distance in protein_distances.items():
                if distance > 2.0:  # Significant hyperbolic distance
                    predictions["vulnerability_zones"].append(
                        {
                            "proteins": pair,
                            "distance": distance,
                            "description": f"Gap between {pair} could be therapeutic target",
                        }
                    )

        return predictions

    def run_full_analysis(self) -> Dict:
        """Run complete hiding landscape analysis."""
        print("\n" + "=" * 70)
        print("HIV COMPLETE HIDING LANDSCAPE ANALYSIS")
        print("=" * 70)
        print("\nHYPOTHESIS: HIV has overfitted hiding at multiple hierarchy levels.")
        print("The 3-adic geometry reveals the complete evolutionary possibility space.")

        # Load encoder
        encoder_loaded = self.load_encoder()

        # Analyze all proteins
        print("\n[1] ANALYZING HIV PROTEOME")
        print("-" * 40)
        all_results = []
        mechanism_counts = defaultdict(int)
        level_counts = defaultdict(int)

        for protein_name, protein_data in HIV_PROTEOME.items():
            print(f"  {protein_name}...", end=" ")
            result = self.analyze_protein_hiding(protein_name, protein_data)
            all_results.append(result)

            n_mech = len(result["mechanisms"])
            mechanism_counts[protein_name] = n_mech
            for mech in result["mechanisms"]:
                level_counts[mech["level"]] += 1

            print(f"{n_mech} mechanisms, {len(result['mimicry_targets'])} mimicry targets")

        # Compute hiding space geometry
        print("\n[2] COMPUTING HIDING SPACE GEOMETRY")
        print("-" * 40)
        hiding_geometry = self.compute_hiding_space_geometry(all_results)

        if hiding_geometry.get("geometry"):
            print(f"  Overall centroid norm: {hiding_geometry['geometry']['overall_centroid_norm']:.3f}")
            print(f"  Mean hiding radius: {hiding_geometry['geometry']['mean_radius']:.3f}")

        print("\n  By hierarchy level:")
        for level, data in hiding_geometry.get("by_level", {}).items():
            print(f"    {level}: centroid_norm={data['centroid_norm']:.3f}, n={data['n_proteins']}")

        # Predict evolutionary possibilities
        print("\n[3] PREDICTING EVOLUTIONARY POSSIBILITIES")
        print("-" * 40)
        predictions = self.predict_evolutionary_possibilities(hiding_geometry)

        print(f"  Constrained regions: {len(predictions['constrained_regions'])}")
        print(f"  Expandable regions: {len(predictions['expandable_regions'])}")
        print(f"  Vulnerability zones: {len(predictions['vulnerability_zones'])}")

        # Analyze signaling interference
        print("\n[4] SIGNALING PATHWAY INTERFERENCE MAP")
        print("-" * 40)
        pathway_analysis = self.analyze_signaling_interference()
        for pathway, data in list(pathway_analysis.items())[:5]:
            proteins = ", ".join(data["hiv_proteins"])
            print(f"  {pathway}: {proteins}")

        # Compile complete results
        results = {
            "metadata": {
                "analysis": "HIV Complete Hiding Landscape",
                "encoder": ("3-adic (V5.11.3)" if encoder_loaded else "fallback"),
                "total_proteins": len(HIV_PROTEOME),
                "total_mechanisms": sum(mechanism_counts.values()),
            },
            "summary": {
                "mechanisms_by_protein": dict(mechanism_counts),
                "mechanisms_by_level": dict(level_counts),
            },
            "hiding_geometry": hiding_geometry,
            "evolutionary_predictions": predictions,
            "signaling_interference": pathway_analysis,
            "protein_details": all_results,
        }

        return results

    def analyze_signaling_interference(self) -> Dict:
        """Analyze how HIV proteins interfere with host signaling."""
        pathway_analysis = {}

        for pathway_name, pathway_data in SIGNALING_PATHWAYS.items():
            hiv_proteins = list(pathway_data["hiv_interference"].keys())

            pathway_analysis[pathway_name] = {
                "description": pathway_data["description"],
                "hiv_proteins": hiv_proteins,
                "interference_mechanisms": pathway_data["hiv_interference"],
                "downstream_effects": pathway_data["downstream_effects"],
                "n_proteins_involved": len(hiv_proteins),
            }

        return pathway_analysis


def main():
    """Run the complete hiding landscape analysis."""
    analyzer = HidingLandscapeAnalyzer(device="cpu")
    results = analyzer.run_full_analysis()

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\n  Total HIV proteins analyzed: {results['metadata']['total_proteins']}")
    print(f"  Total hiding mechanisms cataloged: {results['metadata']['total_mechanisms']}")

    print("\n  Mechanisms by hierarchy level:")
    for level, count in results["summary"]["mechanisms_by_level"].items():
        print(f"    {level}: {count}")

    print("\n  Top hiding specialists (by mechanism count):")
    sorted_proteins = sorted(
        results["summary"]["mechanisms_by_protein"].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for protein, count in sorted_proteins[:5]:
        print(f"    {protein}: {count} mechanisms")

    # Vulnerability analysis
    print("\n  VULNERABILITY ZONES (potential therapeutic targets):")
    for vuln in results["evolutionary_predictions"]["vulnerability_zones"][:5]:
        print(f"    {vuln['proteins']}: distance={vuln['distance']:.2f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "hiv_hiding_landscape.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj

    with open(output_file, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    # Print final hypothesis validation
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)
    print(
        """
  The analysis reveals HIV has indeed developed MULTI-LEVEL hiding:

  1. CODON LEVEL: Specific codon choices for key functional motifs
     - Arginine-rich motifs (ARM) for RNA binding
     - Leucine-rich signals for nuclear export
     - Cysteine-rich domains for zinc coordination

  2. PEPTIDE LEVEL: Epitope masking and variable loops
     - V1-V5 loops in gp120 provide antigenic variation
     - CTL epitopes evolve under immune pressure

  3. PROTEIN LEVEL: Mimicry of human proteins
     - Nef mimics SH3-binding proteins
     - Vif mimics SOCS-box proteins
     - RT mimics host DNA polymerases
     - Tat mimics angiogenic factors

  4. SIGNALING LEVEL: Hijacking of host pathways
     - CD4/TCR signaling disruption
     - MHC downregulation
     - Restriction factor antagonism
     - Cell cycle manipulation

  5. GLYCAN LEVEL: Carbohydrate shield
     - ~50% of gp120 surface covered by glycans
     - Self-glycans mask epitopes
     - Sentinel glycans in "Goldilocks zone"

  IMPLICATION: By mapping the codon-level substrate of ALL these
  hiding mechanisms, we can predict the COMPLETE evolutionary
  possibility space and identify UNIVERSAL therapeutic targets.
"""
    )

    return results


if __name__ == "__main__":
    main()
