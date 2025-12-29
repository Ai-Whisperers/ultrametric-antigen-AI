"""
Tau Protein Database: Sequences and Phosphorylation Sites

Reference: Human MAPT (Microtubule-Associated Protein Tau)
UniProt: P10636
Isoform: 2N4R (441 amino acids) - longest adult brain isoform

Phosphorylation site data compiled from:
- PhosphoSitePlus
- Alzheimer's disease literature
- Cryo-EM structural studies
"""

# Human Tau 2N4R (441 amino acids)
# UniProt P10636-8
TAU_2N4R_SEQUENCE = (
    (
        "MAEPRQEFEV MEDHAGTYGL GDRKDQGGYT MHQDQEGDTD AGLKESPLQT"  # 1-50
        "PTEDGSEEPG SETSDAKSTP TAEDVTAPLV DEGAPGKQAA AQPHTEIPEG"  # 51-100
        "TTAEEAGIGD TPSLEDEAAG HVTQARMVSK SKDGTGSDDK KAKGADGKTK"  # 101-150
        "IATPRGAAPP GQKGQANATR IPAKTPPAPK TPPSSGEPPK SGDRSGYSSP"  # 151-200
        "GSPGTPGSRS RTPSLPTPPT REPKKVAVVR TPPKSPSSAK SRLQTAPVPM"  # 201-250
        "PDLKNVKSKI GSTENLKHQP GGGKVQIINK KLDLSNVQSK CGSKDNIKHV"  # 251-300
        "PGGGSVQIVY KPVDLSKVTS KCGSLGNIHH KPGGGQVEVK SEKLDFKDRV"  # 301-350
        "QSKIGSLDNI THVPGGGNKK IETHKLTFRE NAKAKTDHGA EIVYKSPVVS"  # 351-400
        "GDTSPRHLSN VSSTGSIDMV DSPQLATLAD EVSASLAKQG L"  # 401-441
    )
    .replace(" ", "")
    .replace("\n", "")
)

# Remove any whitespace artifacts
TAU_2N4R_SEQUENCE = "".join(TAU_2N4R_SEQUENCE.split())

# Domain boundaries (1-indexed)
TAU_DOMAINS = {
    "N_terminal": (1, 150),
    "N1_insert": (45, 73),
    "N2_insert": (74, 102),
    "Proline_rich": (151, 243),
    "PRR1": (151, 198),
    "PRR2": (199, 243),
    "MTBR": (244, 368),
    "R1": (244, 274),
    "R2": (275, 305),
    "R3": (306, 336),
    "R4": (337, 368),
    "C_terminal": (369, 441),
}

# Comprehensive phosphorylation site database
# Format: position: {aa, domain, epitope, pathological_stage, kinases, notes}
TAU_PHOSPHO_SITES = {
    # N-terminal region
    18: {
        "aa": "Y",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["Fyn"],
        "notes": "Fyn kinase substrate",
    },
    29: {
        "aa": "T",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "Casein kinase 1 site",
    },
    39: {
        "aa": "S",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "GSK3 site",
    },
    46: {
        "aa": "T",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "early",
        "kinases": ["GSK3", "CDK5"],
        "notes": "N-terminal projection",
    },
    50: {
        "aa": "T",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    52: {
        "aa": "S",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    56: {
        "aa": "S",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    61: {
        "aa": "T",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    63: {
        "aa": "S",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    64: {
        "aa": "S",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    68: {
        "aa": "S",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    69: {
        "aa": "T",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    71: {
        "aa": "T",
        "domain": "N_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    # Proline-rich region - heavily phosphorylated in AD
    153: {
        "aa": "T",
        "domain": "PRR1",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CDK5"],
        "notes": "",
    },
    175: {
        "aa": "T",
        "domain": "PRR1",
        "epitope": "AT270",
        "stage": "early",
        "kinases": ["GSK3", "CDK5"],
        "notes": "AT270 epitope region",
    },
    181: {
        "aa": "T",
        "domain": "PRR1",
        "epitope": "AT270",
        "stage": "early",
        "kinases": ["GSK3", "CDK5", "CK1"],
        "notes": "CSF biomarker, AT270",
    },
    184: {
        "aa": "S",
        "domain": "PRR1",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    185: {
        "aa": "S",
        "domain": "PRR1",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    191: {
        "aa": "Y",
        "domain": "PRR1",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["Fyn", "Src"],
        "notes": "Tyrosine kinase site",
    },
    195: {
        "aa": "S",
        "domain": "PRR1",
        "epitope": "CP27",
        "stage": "early",
        "kinases": ["GSK3", "CDK5"],
        "notes": "",
    },
    198: {
        "aa": "S",
        "domain": "PRR1",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["PKA"],
        "notes": "",
    },
    199: {
        "aa": "S",
        "domain": "PRR2",
        "epitope": "AT100",
        "stage": "early",
        "kinases": ["GSK3"],
        "notes": "Part of AT100 epitope",
    },
    202: {
        "aa": "S",
        "domain": "PRR2",
        "epitope": "AT8/CP13",
        "stage": "early",
        "kinases": ["GSK3", "CDK5", "MARK"],
        "notes": "Major AT8 component",
    },
    205: {
        "aa": "T",
        "domain": "PRR2",
        "epitope": "AT8",
        "stage": "early",
        "kinases": ["GSK3", "CDK5"],
        "notes": "Major AT8 component",
    },
    208: {
        "aa": "S",
        "domain": "PRR2",
        "epitope": "AT8",
        "stage": "early",
        "kinases": ["CK1"],
        "notes": "Extended AT8",
    },
    210: {
        "aa": "T",
        "domain": "PRR2",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    212: {
        "aa": "S",
        "domain": "PRR2",
        "epitope": "AT100",
        "stage": "mid",
        "kinases": ["GSK3"],
        "notes": "Part of AT100",
    },
    214: {
        "aa": "S",
        "domain": "PRR2",
        "epitope": "AT100",
        "stage": "mid",
        "kinases": ["GSK3", "CK1"],
        "notes": "Part of AT100",
    },
    217: {
        "aa": "T",
        "domain": "PRR2",
        "epitope": None,
        "stage": "early",
        "kinases": ["GSK3", "CDK5"],
        "notes": "CSF biomarker (pT217)",
    },
    231: {
        "aa": "T",
        "domain": "PRR2",
        "epitope": "TG3/AT180",
        "stage": "early",
        "kinases": ["GSK3", "CDK5"],
        "notes": "Conformational change marker",
    },
    235: {
        "aa": "S",
        "domain": "PRR2",
        "epitope": "AT180",
        "stage": "early",
        "kinases": ["GSK3"],
        "notes": "Part of AT180",
    },
    237: {
        "aa": "S",
        "domain": "PRR2",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    238: {
        "aa": "T",
        "domain": "PRR2",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    # MTBR region - critical for microtubule binding and aggregation
    258: {
        "aa": "S",
        "domain": "R1",
        "epitope": None,
        "stage": "mid",
        "kinases": ["MARK", "PKA"],
        "notes": "Near KXGS motif",
    },
    262: {
        "aa": "S",
        "domain": "R1",
        "epitope": "12E8",
        "stage": "mid",
        "kinases": ["MARK", "PKA", "BRSK"],
        "notes": "Major MT detachment site",
    },
    263: {
        "aa": "T",
        "domain": "R1",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    285: {
        "aa": "S",
        "domain": "R2",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    289: {
        "aa": "S",
        "domain": "R2",
        "epitope": None,
        "stage": "mid",
        "kinases": ["GSK3", "CDK5"],
        "notes": "",
    },
    293: {
        "aa": "S",
        "domain": "R2",
        "epitope": None,
        "stage": "mid",
        "kinases": ["MARK"],
        "notes": "KXGS motif in R2",
    },
    305: {
        "aa": "S",
        "domain": "R2",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    316: {
        "aa": "S",
        "domain": "R3",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    320: {
        "aa": "S",
        "domain": "R3",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    324: {
        "aa": "S",
        "domain": "R3",
        "epitope": None,
        "stage": "mid",
        "kinases": ["MARK"],
        "notes": "KXGS motif in R3",
    },
    352: {
        "aa": "S",
        "domain": "R4",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    356: {
        "aa": "S",
        "domain": "R4",
        "epitope": None,
        "stage": "mid",
        "kinases": ["MARK"],
        "notes": "KXGS motif in R4",
    },
    # C-terminal region
    394: {
        "aa": "S",
        "domain": "C_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["GSK3"],
        "notes": "",
    },
    396: {
        "aa": "S",
        "domain": "C_terminal",
        "epitope": "PHF-1",
        "stage": "late",
        "kinases": ["GSK3", "CDK5"],
        "notes": "Major PHF-1 component",
    },
    400: {
        "aa": "S",
        "domain": "C_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    404: {
        "aa": "S",
        "domain": "C_terminal",
        "epitope": "PHF-1",
        "stage": "late",
        "kinases": ["GSK3", "CDK5"],
        "notes": "Major PHF-1 component",
    },
    409: {
        "aa": "S",
        "domain": "C_terminal",
        "epitope": "PG5",
        "stage": "late",
        "kinases": ["PKA", "CK1"],
        "notes": "",
    },
    412: {
        "aa": "S",
        "domain": "C_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    413: {
        "aa": "S",
        "domain": "C_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    416: {
        "aa": "S",
        "domain": "C_terminal",
        "epitope": None,
        "stage": "unknown",
        "kinases": ["CK1"],
        "notes": "",
    },
    422: {
        "aa": "S",
        "domain": "C_terminal",
        "epitope": None,
        "stage": "late",
        "kinases": ["GSK3", "CDK5"],
        "notes": "Severe AD marker",
    },
}

# Key pathological epitope combinations
TAU_EPITOPES = {
    "AT8": {
        "sites": [202, 205],
        "optional": [208],
        "description": "Early tangle marker, gold standard for AD staging",
        "stage": "early",
    },
    "AT100": {
        "sites": [212, 214],
        "optional": [199],
        "description": "Specific for AD-type phosphorylation",
        "stage": "mid",
    },
    "AT180": {
        "sites": [231],
        "optional": [235],
        "description": "Conformational epitope, early pathology",
        "stage": "early",
    },
    "AT270": {
        "sites": [181],
        "optional": [175],
        "description": "N-terminal phospho, CSF biomarker",
        "stage": "early",
    },
    "PHF-1": {
        "sites": [396, 404],
        "optional": [],
        "description": "Late-stage PHF marker",
        "stage": "late",
    },
    "12E8": {
        "sites": [262],
        "optional": [356],
        "description": "MTBR phosphorylation, MT detachment",
        "stage": "mid",
    },
    "CP13": {
        "sites": [202],
        "optional": [],
        "description": "pS202 specific",
        "stage": "early",
    },
    "TG3": {
        "sites": [231],
        "optional": [],
        "description": "Conformational epitope requiring pT231",
        "stage": "early",
    },
}

# MTBR residues that contact tubulin (from cryo-EM structures)
# PDB: 6CVN, 6CVJ (tau-microtubule complex)
TAU_TUBULIN_CONTACTS = {
    # R1 contacts
    256: {"tubulin_partner": "alpha", "interaction": "electrostatic"},
    259: {"tubulin_partner": "alpha", "interaction": "hydrophobic"},
    260: {"tubulin_partner": "alpha", "interaction": "electrostatic"},
    # R2 contacts
    287: {"tubulin_partner": "beta", "interaction": "electrostatic"},
    290: {"tubulin_partner": "beta", "interaction": "hydrophobic"},
    291: {"tubulin_partner": "beta", "interaction": "electrostatic"},
    # R3 contacts
    318: {"tubulin_partner": "alpha", "interaction": "electrostatic"},
    321: {"tubulin_partner": "alpha", "interaction": "hydrophobic"},
    322: {"tubulin_partner": "alpha", "interaction": "electrostatic"},
    # R4 contacts
    349: {"tubulin_partner": "beta", "interaction": "electrostatic"},
    352: {"tubulin_partner": "beta", "interaction": "hydrophobic"},
    353: {"tubulin_partner": "beta", "interaction": "electrostatic"},
}

# KXGS motifs - major MARK kinase targets, critical for MT binding
KXGS_MOTIFS = {
    "R1": {"K": 259, "X": 260, "G": 261, "S": 262},
    "R2": {"K": 290, "X": 291, "G": 292, "S": 293},
    "R3": {"K": 321, "X": 322, "G": 323, "S": 324},
    "R4": {"K": 353, "X": 354, "G": 355, "S": 356},
}


def get_all_phospho_positions():
    """Return list of all phosphorylatable positions."""
    return sorted(TAU_PHOSPHO_SITES.keys())


def get_sites_by_domain(domain):
    """Return phospho-sites in a specific domain."""
    return {pos: data for pos, data in TAU_PHOSPHO_SITES.items() if data["domain"] == domain}


def get_sites_by_stage(stage):
    """Return phospho-sites associated with a pathological stage."""
    return {pos: data for pos, data in TAU_PHOSPHO_SITES.items() if data["stage"] == stage}


def get_epitope_sites(epitope_name):
    """Return sites that comprise a specific epitope."""
    if epitope_name in TAU_EPITOPES:
        return TAU_EPITOPES[epitope_name]["sites"]
    return []


if __name__ == "__main__":
    print(f"Tau 2N4R sequence length: {len(TAU_2N4R_SEQUENCE)}")
    print(f"Total phospho-sites catalogued: {len(TAU_PHOSPHO_SITES)}")
    print(f"Pathological epitopes: {list(TAU_EPITOPES.keys())}")

    # Count by domain
    domains = {}
    for pos, data in TAU_PHOSPHO_SITES.items():
        d = data["domain"]
        domains[d] = domains.get(d, 0) + 1

    print("\nPhospho-sites by domain:")
    for d, count in sorted(domains.items()):
        print(f"  {d}: {count}")

    # Count by stage
    stages = {}
    for pos, data in TAU_PHOSPHO_SITES.items():
        s = data["stage"]
        stages[s] = stages.get(s, 0) + 1

    print("\nPhospho-sites by pathological stage:")
    for s, count in sorted(stages.items()):
        print(f"  {s}: {count}")
