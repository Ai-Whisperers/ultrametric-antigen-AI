"""
Los Alamos National Laboratory HIV Database client.

Provides access to:
- HIV sequence data (curated)
- Drug resistance mutations
- CTL/Antibody epitopes
- HIV immunology data
- Geographic and subtype distributions
- Transmission and treatment data

LANL HIV DB: https://www.hiv.lanl.gov/
Note: Some features require form-based queries or downloads.
"""

from typing import Optional

import pandas as pd
import requests

from ..config import settings


class LANLHIVClient:
    """Client for Los Alamos HIV Database."""

    BASE_URL = "https://www.hiv.lanl.gov"
    SEQUENCE_URL = "https://www.hiv.lanl.gov/cgi-bin/NEWALIGN/align.cgi"

    # HIV genes and their positions (HXB2 reference)
    HIV_GENES = {
        "gag": {"start": 790, "end": 2292},
        "pol": {"start": 2085, "end": 5096},
        "vif": {"start": 5041, "end": 5619},
        "vpr": {"start": 5559, "end": 5850},
        "tat": {"start": 5831, "end": 8424},  # exons combined
        "rev": {"start": 5970, "end": 8653},  # exons combined
        "vpu": {"start": 6062, "end": 6310},
        "env": {"start": 6225, "end": 8795},
        "nef": {"start": 8797, "end": 9417},
    }

    # Major HIV-1 subtypes
    SUBTYPES = ["A", "B", "C", "D", "F", "G", "H", "J", "K", "CRF01_AE", "CRF02_AG"]

    # Drug classes
    DRUG_CLASSES = {
        "NRTI": ["3TC", "ABC", "AZT", "D4T", "DDI", "FTC", "TDF"],
        "NNRTI": ["DLV", "EFV", "ETR", "NVP", "RPV"],
        "PI": ["ATV", "DRV", "FPV", "IDV", "LPV", "NFV", "RTV", "SQV", "TPV"],
        "INSTI": ["BIC", "CAB", "DTG", "EVG", "RAL"],
    }

    def __init__(self):
        """Initialize LANL HIV client."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Python LANL HIV Client",
            "Accept": "text/html,application/json",
        })

    def _get(self, endpoint: str, params: Optional[dict] = None) -> requests.Response:
        """Make GET request."""
        url = f"{self.BASE_URL}/{endpoint}" if not endpoint.startswith("http") else endpoint
        response = self.session.get(url, params=params, timeout=settings.timeout)
        response.raise_for_status()
        return response

    def _post(self, endpoint: str, data: dict) -> requests.Response:
        """Make POST request."""
        url = f"{self.BASE_URL}/{endpoint}" if not endpoint.startswith("http") else endpoint
        response = self.session.post(url, data=data, timeout=settings.timeout)
        response.raise_for_status()
        return response

    # ========== SEQUENCE DATA ==========

    def get_reference_sequence(self, gene: str = "complete") -> dict:
        """
        Get HXB2 reference sequence for a gene.

        Args:
            gene: HIV gene name or 'complete' for full genome

        Returns:
            Dictionary with sequence data
        """
        # HXB2 reference accession: K03455
        gene_info = self.HIV_GENES.get(gene.lower(), {})

        return {
            "reference": "HXB2",
            "accession": "K03455",
            "gene": gene,
            "positions": gene_info,
            "note": "Use NCBI to fetch actual sequence with accession K03455",
        }

    def search_sequences(
        self,
        subtype: Optional[str] = None,
        country: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        gene: str = "POL",
        sampling: str = "NONE",
        limit: int = 100,
    ) -> dict:
        """
        Build a sequence search query for LANL.

        Note: LANL requires form-based search. This returns query parameters.

        Args:
            subtype: HIV subtype filter
            country: Country filter
            year_from: Start year
            year_to: End year
            gene: Gene region
            sampling: Sampling strategy
            limit: Maximum sequences

        Returns:
            Dictionary with search parameters for LANL interface
        """
        params = {
            "ORGANISM": "HIV",
            "REGION": gene.upper(),
            "ALIGNMENT_TYPE": "CODON",
            "SUBORGANISM": subtype or "ALL",
            "COUNTRY": country or "ALL",
            "SAMPLING": sampling,
            "MAX_SEQ": str(limit),
        }

        if year_from:
            params["YEAR_FROM"] = str(year_from)
        if year_to:
            params["YEAR_TO"] = str(year_to)

        return {
            "url": f"{self.BASE_URL}/cgi-bin/NEWALIGN/align.cgi",
            "params": params,
            "method": "POST",
            "note": "Submit these parameters to LANL alignment interface",
        }

    # ========== DRUG RESISTANCE ==========

    def get_resistance_mutations(self, drug_class: Optional[str] = None) -> pd.DataFrame:
        """
        Get known HIV drug resistance mutations.

        Args:
            drug_class: Filter by drug class (NRTI, NNRTI, PI, INSTI)

        Returns:
            DataFrame with resistance mutation data
        """
        # Curated list of major resistance mutations
        mutations = [
            # NRTI mutations
            {"drug_class": "NRTI", "gene": "RT", "position": 41, "mutation": "M41L", "drugs_affected": "AZT, D4T"},
            {"drug_class": "NRTI", "gene": "RT", "position": 65, "mutation": "K65R", "drugs_affected": "TDF, ABC, DDI"},
            {"drug_class": "NRTI", "gene": "RT", "position": 67, "mutation": "D67N", "drugs_affected": "AZT, D4T"},
            {"drug_class": "NRTI", "gene": "RT", "position": 70, "mutation": "K70R", "drugs_affected": "AZT, TDF"},
            {"drug_class": "NRTI", "gene": "RT", "position": 74, "mutation": "L74V", "drugs_affected": "ABC, DDI"},
            {"drug_class": "NRTI", "gene": "RT", "position": 115, "mutation": "Y115F", "drugs_affected": "ABC"},
            {"drug_class": "NRTI", "gene": "RT", "position": 151, "mutation": "Q151M", "drugs_affected": "Multi-NRTI"},
            {"drug_class": "NRTI", "gene": "RT", "position": 184, "mutation": "M184V/I", "drugs_affected": "3TC, FTC"},
            {"drug_class": "NRTI", "gene": "RT", "position": 210, "mutation": "L210W", "drugs_affected": "AZT, D4T"},
            {"drug_class": "NRTI", "gene": "RT", "position": 215, "mutation": "T215Y/F", "drugs_affected": "AZT, D4T"},
            {"drug_class": "NRTI", "gene": "RT", "position": 219, "mutation": "K219Q/E", "drugs_affected": "AZT"},
            # NNRTI mutations
            {"drug_class": "NNRTI", "gene": "RT", "position": 100, "mutation": "L100I", "drugs_affected": "EFV, NVP"},
            {"drug_class": "NNRTI", "gene": "RT", "position": 103, "mutation": "K103N", "drugs_affected": "EFV, NVP"},
            {"drug_class": "NNRTI", "gene": "RT", "position": 106, "mutation": "V106A/M", "drugs_affected": "NVP, EFV"},
            {"drug_class": "NNRTI", "gene": "RT", "position": 181, "mutation": "Y181C/I", "drugs_affected": "NVP, ETR, RPV"},
            {"drug_class": "NNRTI", "gene": "RT", "position": 188, "mutation": "Y188L/C", "drugs_affected": "NVP, EFV"},
            {"drug_class": "NNRTI", "gene": "RT", "position": 190, "mutation": "G190A/S", "drugs_affected": "NVP, EFV"},
            {"drug_class": "NNRTI", "gene": "RT", "position": 230, "mutation": "M230L", "drugs_affected": "ETR, RPV"},
            # PI mutations
            {"drug_class": "PI", "gene": "PR", "position": 30, "mutation": "D30N", "drugs_affected": "NFV"},
            {"drug_class": "PI", "gene": "PR", "position": 46, "mutation": "M46I/L", "drugs_affected": "IDV, NFV"},
            {"drug_class": "PI", "gene": "PR", "position": 48, "mutation": "G48V", "drugs_affected": "SQV, ATV"},
            {"drug_class": "PI", "gene": "PR", "position": 50, "mutation": "I50V/L", "drugs_affected": "ATV, DRV"},
            {"drug_class": "PI", "gene": "PR", "position": 54, "mutation": "I54V/M/L", "drugs_affected": "Multi-PI"},
            {"drug_class": "PI", "gene": "PR", "position": 76, "mutation": "L76V", "drugs_affected": "LPV, DRV"},
            {"drug_class": "PI", "gene": "PR", "position": 82, "mutation": "V82A/F/T/S", "drugs_affected": "IDV, LPV"},
            {"drug_class": "PI", "gene": "PR", "position": 84, "mutation": "I84V", "drugs_affected": "Multi-PI"},
            {"drug_class": "PI", "gene": "PR", "position": 88, "mutation": "N88S", "drugs_affected": "NFV, ATV"},
            {"drug_class": "PI", "gene": "PR", "position": 90, "mutation": "L90M", "drugs_affected": "Multi-PI"},
            # INSTI mutations
            {"drug_class": "INSTI", "gene": "IN", "position": 66, "mutation": "T66I/A/K", "drugs_affected": "EVG"},
            {"drug_class": "INSTI", "gene": "IN", "position": 92, "mutation": "E92Q", "drugs_affected": "EVG, RAL"},
            {"drug_class": "INSTI", "gene": "IN", "position": 118, "mutation": "G118R", "drugs_affected": "DTG, BIC"},
            {"drug_class": "INSTI", "gene": "IN", "position": 140, "mutation": "G140S/A", "drugs_affected": "RAL, EVG"},
            {"drug_class": "INSTI", "gene": "IN", "position": 143, "mutation": "Y143R/C/H", "drugs_affected": "RAL"},
            {"drug_class": "INSTI", "gene": "IN", "position": 148, "mutation": "Q148H/R/K", "drugs_affected": "Multi-INSTI"},
            {"drug_class": "INSTI", "gene": "IN", "position": 155, "mutation": "N155H", "drugs_affected": "RAL, EVG"},
            {"drug_class": "INSTI", "gene": "IN", "position": 263, "mutation": "R263K", "drugs_affected": "DTG, BIC"},
        ]

        df = pd.DataFrame(mutations)

        if drug_class:
            df = df[df["drug_class"] == drug_class.upper()]

        return df

    def get_drug_info(self, drug_class: Optional[str] = None) -> pd.DataFrame:
        """
        Get information about antiretroviral drugs.

        Args:
            drug_class: Filter by drug class

        Returns:
            DataFrame with drug information
        """
        drugs = []
        for cls, drug_list in self.DRUG_CLASSES.items():
            if drug_class is None or cls == drug_class.upper():
                for drug in drug_list:
                    drugs.append({"drug_class": cls, "abbreviation": drug})

        return pd.DataFrame(drugs)

    # ========== EPITOPE DATA ==========

    def get_ctl_epitopes(
        self,
        protein: Optional[str] = None,
        hla: Optional[str] = None,
    ) -> dict:
        """
        Get CTL epitope data.

        Note: Full epitope data requires LANL interface access.

        Args:
            protein: HIV protein filter
            hla: HLA allele filter

        Returns:
            Dictionary with epitope search guidance
        """
        return {
            "database": "LANL HIV Immunology Database",
            "url": f"{self.BASE_URL}/content/immunology/ctl_search",
            "search_params": {
                "protein": protein,
                "hla": hla,
            },
            "note": "Access the LANL CTL epitope search interface for full data",
            "data_available": [
                "Optimal epitopes",
                "HLA restriction",
                "Response frequency",
                "Immunodominance",
            ],
        }

    def get_antibody_epitopes(
        self,
        protein: Optional[str] = None,
        epitope_type: Optional[str] = None,
    ) -> dict:
        """
        Get antibody epitope data.

        Args:
            protein: HIV protein (usually Env)
            epitope_type: Linear or conformational

        Returns:
            Dictionary with antibody epitope search guidance
        """
        return {
            "database": "LANL HIV Immunology Database",
            "url": f"{self.BASE_URL}/content/immunology/ab_search",
            "search_params": {
                "protein": protein or "Env",
                "epitope_type": epitope_type,
            },
            "note": "Access the LANL antibody epitope search interface for full data",
            "data_available": [
                "Broadly neutralizing antibodies",
                "Epitope mapping",
                "Neutralization data",
                "Structural information",
            ],
        }

    def get_bnab_targets(self) -> pd.DataFrame:
        """
        Get broadly neutralizing antibody (bnAb) target sites.

        Returns:
            DataFrame with bnAb target site information
        """
        targets = [
            {
                "site": "CD4 binding site",
                "protein": "gp120",
                "bnabs": "VRC01, 3BNC117, N6",
                "conservation": "High",
            },
            {
                "site": "V2 apex",
                "protein": "gp120",
                "bnabs": "PG9, PGT145, CAP256-VRC26",
                "conservation": "Moderate",
            },
            {
                "site": "V3 glycan",
                "protein": "gp120",
                "bnabs": "PGT121, PGT128, 10-1074",
                "conservation": "Moderate",
            },
            {
                "site": "gp120/gp41 interface",
                "protein": "gp120/gp41",
                "bnabs": "8ANC195, 35O22",
                "conservation": "High",
            },
            {
                "site": "MPER",
                "protein": "gp41",
                "bnabs": "10E8, 4E10, 2F5",
                "conservation": "High",
            },
            {
                "site": "Fusion peptide",
                "protein": "gp41",
                "bnabs": "VRC34.01, ACS202",
                "conservation": "High",
            },
        ]
        return pd.DataFrame(targets)

    # ========== SUBTYPE INFORMATION ==========

    def get_subtype_info(self) -> pd.DataFrame:
        """
        Get HIV-1 subtype information.

        Returns:
            DataFrame with subtype descriptions
        """
        subtypes = [
            {"subtype": "A", "prevalence": "East Africa, Russia", "features": "Common in Kenya, Uganda"},
            {"subtype": "B", "prevalence": "Americas, Europe, Australia", "features": "Most studied subtype"},
            {"subtype": "C", "prevalence": "Southern Africa, India, Ethiopia", "features": "Most prevalent globally (~50%)"},
            {"subtype": "D", "prevalence": "East/Central Africa", "features": "Associated with faster progression"},
            {"subtype": "F", "prevalence": "Central Africa, South America", "features": "Less common"},
            {"subtype": "G", "prevalence": "West/Central Africa", "features": "Often recombines"},
            {"subtype": "CRF01_AE", "prevalence": "Southeast Asia", "features": "A/E recombinant"},
            {"subtype": "CRF02_AG", "prevalence": "West Africa", "features": "A/G recombinant"},
        ]
        return pd.DataFrame(subtypes)

    def get_subtype_distribution(self, region: Optional[str] = None) -> dict:
        """
        Get HIV subtype distribution data.

        Args:
            region: Geographic region filter

        Returns:
            Dictionary with subtype distribution guidance
        """
        return {
            "database": "LANL Geography Search",
            "url": f"{self.BASE_URL}/components/sequence/HIV/search/search.html",
            "note": "Use LANL geography search for detailed subtype distributions",
            "global_estimates": {
                "C": "~48%",
                "A": "~12%",
                "B": "~11%",
                "CRF02_AG": "~8%",
                "CRF01_AE": "~5%",
                "D": "~2%",
                "G": "~5%",
                "Other": "~9%",
            },
        }

    # ========== SEQUENCE ANALYSIS TOOLS ==========

    def get_alignment_tool_params(
        self,
        sequences: Optional[list[str]] = None,
        gene: str = "POL",
    ) -> dict:
        """
        Get parameters for LANL sequence alignment tool.

        Args:
            sequences: Sequences to align (FASTA format)
            gene: Gene region for alignment

        Returns:
            Dictionary with alignment tool parameters
        """
        return {
            "tool": "LANL Gene Cutter",
            "url": f"{self.BASE_URL}/cgi-bin/GENE_CUTTER/simpleGC.cgi",
            "description": "Extracts gene regions from HIV sequences",
            "params": {
                "GENE": gene.upper(),
                "SEQUENCES": sequences,
            },
            "alternative_tools": [
                {"name": "VESPA", "purpose": "Signature pattern analysis"},
                {"name": "Highlighter", "purpose": "Mutation visualization"},
                {"name": "RIP", "purpose": "Recombinant identification"},
            ],
        }

    def get_phylogenetic_analysis_info(self) -> dict:
        """
        Get information about LANL phylogenetic analysis tools.

        Returns:
            Dictionary with tool information
        """
        return {
            "tools": [
                {
                    "name": "PhyML",
                    "url": f"{self.BASE_URL}/cgi-bin/PhyML/phyml.cgi",
                    "purpose": "Maximum likelihood phylogenetic analysis",
                },
                {
                    "name": "MEGA",
                    "url": f"{self.BASE_URL}/cgi-bin/MEGA/mega.cgi",
                    "purpose": "Evolutionary analysis",
                },
                {
                    "name": "HyPhy",
                    "url": "https://www.datamonkey.org/",
                    "purpose": "Selection pressure analysis",
                },
            ],
            "note": "These tools require sequence input through their web interfaces",
        }

    # ========== COMPENDIUM DATA ==========

    def get_compendium_info(self, year: int = 2024) -> dict:
        """
        Get information about LANL HIV sequence compendia.

        Args:
            year: Compendium year

        Returns:
            Dictionary with compendium information
        """
        return {
            "database": "LANL HIV Sequence Compendium",
            "url": f"{self.BASE_URL}/content/sequence/HIV/COMPENDIUM/compendium.html",
            "description": "Curated reference alignments for each HIV-1 subtype",
            "contents": [
                "Reference alignments per subtype",
                "Consensus sequences",
                "Ancestral sequences",
                "Complete genome references",
            ],
            "download_format": ["FASTA", "Nexus", "Phylip"],
        }

    def get_reference_strains(self) -> pd.DataFrame:
        """
        Get common HIV reference strains.

        Returns:
            DataFrame with reference strain information
        """
        strains = [
            {"name": "HXB2", "accession": "K03455", "subtype": "B", "description": "Standard reference"},
            {"name": "NL4-3", "accession": "AF324493", "subtype": "B", "description": "Infectious clone"},
            {"name": "SIVmac239", "accession": "M33262", "subtype": "SIV", "description": "Macaque SIV reference"},
            {"name": "HIV-2 ROD", "accession": "M15390", "subtype": "HIV-2", "description": "HIV-2 reference"},
            {"name": "Consensus C", "accession": "N/A", "subtype": "C", "description": "Subtype C consensus"},
            {"name": "Consensus B", "accession": "N/A", "subtype": "B", "description": "Subtype B consensus"},
        ]
        return pd.DataFrame(strains)

    # ========== TRANSMISSION DATA ==========

    def get_transmission_cluster_info(self) -> dict:
        """
        Get information about HIV transmission cluster analysis.

        Returns:
            Dictionary with transmission analysis guidance
        """
        return {
            "tools": [
                {
                    "name": "HIV-TRACE",
                    "description": "Molecular clustering for transmission networks",
                    "threshold": "1.5% genetic distance",
                },
                {
                    "name": "Cluster Picker",
                    "description": "Phylogenetic clustering",
                    "threshold": "Bootstrap support + genetic distance",
                },
            ],
            "metrics": [
                "Genetic distance (TN93)",
                "Pairwise diversity",
                "Cluster growth rate",
                "Transmission rate estimation",
            ],
        }

    # ========== HELPER METHODS ==========

    def get_codon_position(self, gene: str, aa_position: int) -> dict:
        """
        Convert amino acid position to HXB2 nucleotide position.

        Args:
            gene: Gene name
            aa_position: Amino acid position in the gene

        Returns:
            Dictionary with position information
        """
        gene_info = self.HIV_GENES.get(gene.lower())

        if not gene_info:
            return {"error": f"Unknown gene: {gene}"}

        # Calculate nucleotide position (1-based)
        nt_start = gene_info["start"] + (aa_position - 1) * 3

        return {
            "gene": gene,
            "aa_position": aa_position,
            "codon_start": nt_start,
            "codon_end": nt_start + 2,
            "reference": "HXB2",
        }

    def get_summary(self) -> dict:
        """
        Get summary of available LANL data.

        Returns:
            Dictionary with database summary
        """
        return {
            "name": "Los Alamos National Laboratory HIV Database",
            "url": self.BASE_URL,
            "data_available": {
                "sequences": "Curated HIV-1/2/SIV sequences",
                "resistance": "Drug resistance mutations and annotations",
                "immunology": "CTL and antibody epitopes",
                "phylogenetics": "Reference alignments and trees",
                "geography": "Subtype distribution by region",
                "tools": "Alignment, phylogenetics, analysis tools",
            },
            "access_note": "Full data access requires web interface; API provides guidance and curated data",
        }
