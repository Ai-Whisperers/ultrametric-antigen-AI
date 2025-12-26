"""
IEDB (Immune Epitope Database) API client for immune epitope data.

Provides access to:
- B-cell and T-cell epitopes
- MHC binding predictions
- Epitope conservation analysis
- Immunogenicity data
- Population coverage analysis

IEDB API: https://tools.iedb.org/main/tools-api/
Documentation: https://help.iedb.org/
"""

from typing import Optional

import pandas as pd
import requests

from ..config import settings


class IEDBClient:
    """Client for IEDB (Immune Epitope Database) API."""

    # Base URLs
    BASE_URL = "https://query-api.iedb.org"
    TOOLS_URL = "https://tools-cluster-interface.iedb.org/tools_api"

    # Common pathogen organism IDs
    ORGANISM_IDS = {
        "HIV-1": 11676,
        "HIV-2": 11709,
        "SARS-CoV-2": 2697049,
        "Influenza A": 11320,
        "Hepatitis B virus": 10407,
        "Hepatitis C virus": 11103,
        "Mycobacterium tuberculosis": 1773,
        "Plasmodium falciparum": 5833,
        "Treponema pallidum": 160,
        "Dengue virus": 12637,
        "Zika virus": 64320,
        "Ebola virus": 186536,
    }

    # Common HLA alleles
    COMMON_HLA = {
        "class_i": [
            "HLA-A*02:01",
            "HLA-A*01:01",
            "HLA-A*03:01",
            "HLA-A*24:02",
            "HLA-B*07:02",
            "HLA-B*08:01",
            "HLA-B*44:02",
            "HLA-C*07:01",
        ],
        "class_ii": [
            "HLA-DRB1*01:01",
            "HLA-DRB1*03:01",
            "HLA-DRB1*04:01",
            "HLA-DRB1*07:01",
            "HLA-DRB1*15:01",
            "HLA-DQB1*02:01",
            "HLA-DQB1*06:02",
        ],
    }

    def __init__(self):
        """Initialize IEDB client."""
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """Make GET request."""
        response = self.session.get(url, params=params, timeout=settings.timeout)
        response.raise_for_status()
        return response.json()

    def _post(self, url: str, data: dict) -> dict:
        """Make POST request."""
        response = self.session.post(url, data=data, timeout=settings.timeout)
        response.raise_for_status()

        # Handle different response formats
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type:
            return response.json()
        else:
            return {"text": response.text}

    # ========== EPITOPE SEARCH ==========

    def search_epitopes(
        self,
        organism: Optional[str] = None,
        epitope_sequence: Optional[str] = None,
        epitope_type: Optional[str] = None,
        mhc_restriction: Optional[str] = None,
        host: str = "human",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Search for epitopes in IEDB.

        Args:
            organism: Source organism name
            epitope_sequence: Epitope sequence pattern
            epitope_type: Type (linear, discontinuous, non-peptidic)
            mhc_restriction: MHC allele restriction
            host: Host organism
            limit: Maximum results

        Returns:
            DataFrame with epitope data
        """
        params = {"limit": limit}

        if organism:
            organism_id = self.ORGANISM_IDS.get(organism)
            if organism_id:
                params["source_organism_iri"] = f"http://purl.obolibrary.org/obo/NCBITaxon_{organism_id}"
            else:
                params["source_organism_label"] = organism

        if epitope_sequence:
            params["linear_peptide_seq"] = epitope_sequence

        if epitope_type:
            params["structure_type"] = epitope_type

        if mhc_restriction:
            params["mhc_allele_name"] = mhc_restriction

        try:
            data = self._get(f"{self.BASE_URL}/epitope_search", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_epitope(self, epitope_id: int) -> dict:
        """
        Get details for a specific epitope.

        Args:
            epitope_id: IEDB epitope ID

        Returns:
            Dictionary with epitope details
        """
        try:
            return self._get(f"{self.BASE_URL}/epitope/{epitope_id}")
        except Exception:
            return {}

    # ========== T-CELL EPITOPES ==========

    def search_tcell_epitopes(
        self,
        organism: Optional[str] = None,
        mhc_class: Optional[str] = None,
        assay_type: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Search for T-cell epitopes.

        Args:
            organism: Source organism
            mhc_class: MHC class (I or II)
            assay_type: Assay type (ELISPOT, tetramer, etc.)
            limit: Maximum results

        Returns:
            DataFrame with T-cell epitope data
        """
        params = {"limit": limit, "assay_category": "T cell"}

        if organism:
            organism_id = self.ORGANISM_IDS.get(organism)
            if organism_id:
                params["source_organism_iri"] = f"http://purl.obolibrary.org/obo/NCBITaxon_{organism_id}"

        if mhc_class:
            params["mhc_class"] = mhc_class

        if assay_type:
            params["assay_type"] = assay_type

        try:
            data = self._get(f"{self.BASE_URL}/tcell_search", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_hiv_ctl_epitopes(self, gene: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        """
        Get HIV CTL (Cytotoxic T Lymphocyte) epitopes.

        Args:
            gene: HIV gene (gag, pol, env, nef, etc.)
            limit: Maximum results

        Returns:
            DataFrame with HIV CTL epitopes
        """
        params = {
            "limit": limit,
            "source_organism_iri": f"http://purl.obolibrary.org/obo/NCBITaxon_{self.ORGANISM_IDS['HIV-1']}",
            "mhc_class": "I",
        }

        if gene:
            params["source_antigen_label"] = gene

        try:
            data = self._get(f"{self.BASE_URL}/tcell_search", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ========== B-CELL EPITOPES ==========

    def search_bcell_epitopes(
        self,
        organism: Optional[str] = None,
        assay_type: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Search for B-cell epitopes.

        Args:
            organism: Source organism
            assay_type: Assay type
            limit: Maximum results

        Returns:
            DataFrame with B-cell epitope data
        """
        params = {"limit": limit, "assay_category": "B cell"}

        if organism:
            organism_id = self.ORGANISM_IDS.get(organism)
            if organism_id:
                params["source_organism_iri"] = f"http://purl.obolibrary.org/obo/NCBITaxon_{organism_id}"

        if assay_type:
            params["assay_type"] = assay_type

        try:
            data = self._get(f"{self.BASE_URL}/bcell_search", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_neutralizing_epitopes(
        self,
        organism: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Get neutralizing antibody epitopes.

        Args:
            organism: Target organism
            limit: Maximum results

        Returns:
            DataFrame with neutralizing epitope data
        """
        params = {
            "limit": limit,
            "assay_category": "B cell",
            "qualitative_measure": "Positive",
        }

        organism_id = self.ORGANISM_IDS.get(organism)
        if organism_id:
            params["source_organism_iri"] = f"http://purl.obolibrary.org/obo/NCBITaxon_{organism_id}"

        try:
            data = self._get(f"{self.BASE_URL}/bcell_search", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ========== MHC BINDING PREDICTIONS ==========

    def predict_mhc_binding(
        self,
        sequence: str,
        alleles: Optional[list[str]] = None,
        method: str = "recommended",
        length: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Predict MHC class I binding for a sequence.

        Args:
            sequence: Amino acid sequence
            alleles: List of HLA alleles (defaults to common alleles)
            method: Prediction method (recommended, netmhcpan, ann, etc.)
            length: Peptide length for predictions (8-14)

        Returns:
            DataFrame with binding predictions
        """
        if alleles is None:
            alleles = self.COMMON_HLA["class_i"][:4]

        data = {
            "method": method,
            "sequence_text": sequence,
            "allele": ",".join(alleles),
        }

        if length:
            data["length"] = str(length)

        try:
            response = self.session.post(
                f"{self.TOOLS_URL}/mhci/",
                data=data,
                timeout=120,  # Predictions can take time
            )
            response.raise_for_status()

            # Parse tab-separated results
            lines = response.text.strip().split("\n")
            if len(lines) > 1:
                headers = lines[0].split("\t")
                rows = [line.split("\t") for line in lines[1:]]
                return pd.DataFrame(rows, columns=headers)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def predict_mhc_class_ii_binding(
        self,
        sequence: str,
        alleles: Optional[list[str]] = None,
        method: str = "recommended",
    ) -> pd.DataFrame:
        """
        Predict MHC class II binding.

        Args:
            sequence: Amino acid sequence
            alleles: List of HLA-DR alleles
            method: Prediction method

        Returns:
            DataFrame with binding predictions
        """
        if alleles is None:
            alleles = self.COMMON_HLA["class_ii"][:4]

        data = {
            "method": method,
            "sequence_text": sequence,
            "allele": ",".join(alleles),
        }

        try:
            response = self.session.post(
                f"{self.TOOLS_URL}/mhcii/",
                data=data,
                timeout=120,
            )
            response.raise_for_status()

            lines = response.text.strip().split("\n")
            if len(lines) > 1:
                headers = lines[0].split("\t")
                rows = [line.split("\t") for line in lines[1:]]
                return pd.DataFrame(rows, columns=headers)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ========== IMMUNOGENICITY PREDICTIONS ==========

    def predict_immunogenicity(
        self,
        peptides: list[str],
        mhc_allele: str = "HLA-A*02:01",
    ) -> pd.DataFrame:
        """
        Predict T-cell immunogenicity for peptides.

        Args:
            peptides: List of peptide sequences
            mhc_allele: MHC allele for prediction

        Returns:
            DataFrame with immunogenicity scores
        """
        data = {
            "method": "immunogenicity",
            "sequence_text": "\n".join(peptides),
            "mhc": mhc_allele,
        }

        try:
            response = self.session.post(
                f"{self.TOOLS_URL}/immunogenicity/",
                data=data,
                timeout=60,
            )
            response.raise_for_status()

            lines = response.text.strip().split("\n")
            if len(lines) > 1:
                headers = lines[0].split("\t")
                rows = [line.split("\t") for line in lines[1:]]
                return pd.DataFrame(rows, columns=headers)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ========== EPITOPE CONSERVATION ==========

    def analyze_epitope_conservation(
        self,
        epitope_sequence: str,
        sequences: list[str],
    ) -> dict:
        """
        Analyze conservation of an epitope across sequences.

        Args:
            epitope_sequence: Query epitope sequence
            sequences: List of protein sequences to check

        Returns:
            Dictionary with conservation analysis
        """
        data = {
            "epitope_sequence": epitope_sequence,
            "sequence_text": "\n".join(sequences),
            "method": "identity",
        }

        try:
            response = self.session.post(
                f"{self.TOOLS_URL}/conservancy/",
                data=data,
                timeout=60,
            )
            response.raise_for_status()
            return {"result": response.text}
        except Exception:
            return {}

    # ========== POPULATION COVERAGE ==========

    def calculate_population_coverage(
        self,
        epitopes: list[str],
        alleles: list[str],
        population: str = "World",
        mhc_class: str = "I",
    ) -> dict:
        """
        Calculate population coverage for epitope set.

        Args:
            epitopes: List of epitope sequences
            alleles: List of restricting HLA alleles (one per epitope)
            population: Population name
            mhc_class: MHC class (I, II, or combined)

        Returns:
            Dictionary with population coverage data
        """
        # Format epitope-allele pairs
        epitope_data = "\n".join(f"{e}\t{a}" for e, a in zip(epitopes, alleles))

        data = {
            "epitopes": epitope_data,
            "population": population,
            "mhc_class": mhc_class,
        }

        try:
            response = self.session.post(
                f"{self.TOOLS_URL}/population/",
                data=data,
                timeout=60,
            )
            response.raise_for_status()
            return {"result": response.text}
        except Exception:
            return {}

    # ========== PATHOGEN-SPECIFIC EPITOPES ==========

    def get_hiv_epitopes(
        self,
        epitope_type: Optional[str] = None,
        protein: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get HIV-1 epitopes.

        Args:
            epitope_type: T-cell or B-cell
            protein: HIV protein (Gag, Pol, Env, Nef, etc.)
            limit: Maximum results

        Returns:
            DataFrame with HIV epitopes
        """
        params = {
            "limit": limit,
            "source_organism_iri": f"http://purl.obolibrary.org/obo/NCBITaxon_{self.ORGANISM_IDS['HIV-1']}",
        }

        if protein:
            params["source_antigen_label"] = protein

        try:
            endpoint = "epitope_search"
            if epitope_type == "T-cell":
                endpoint = "tcell_search"
            elif epitope_type == "B-cell":
                endpoint = "bcell_search"

            data = self._get(f"{self.BASE_URL}/{endpoint}", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_covid_epitopes(
        self,
        protein: Optional[str] = None,
        epitope_type: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get SARS-CoV-2 epitopes.

        Args:
            protein: Spike, Nucleocapsid, Membrane, etc.
            epitope_type: T-cell or B-cell
            limit: Maximum results

        Returns:
            DataFrame with COVID-19 epitopes
        """
        params = {
            "limit": limit,
            "source_organism_iri": f"http://purl.obolibrary.org/obo/NCBITaxon_{self.ORGANISM_IDS['SARS-CoV-2']}",
        }

        if protein:
            params["source_antigen_label"] = protein

        try:
            endpoint = "epitope_search"
            if epitope_type == "T-cell":
                endpoint = "tcell_search"
            elif epitope_type == "B-cell":
                endpoint = "bcell_search"

            data = self._get(f"{self.BASE_URL}/{endpoint}", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_tb_epitopes(self, limit: int = 500) -> pd.DataFrame:
        """
        Get Mycobacterium tuberculosis epitopes.

        Returns:
            DataFrame with TB epitopes
        """
        params = {
            "limit": limit,
            "source_organism_iri": f"http://purl.obolibrary.org/obo/NCBITaxon_{self.ORGANISM_IDS['Mycobacterium tuberculosis']}",
        }

        try:
            data = self._get(f"{self.BASE_URL}/epitope_search", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_malaria_epitopes(self, limit: int = 500) -> pd.DataFrame:
        """
        Get Plasmodium falciparum epitopes.

        Returns:
            DataFrame with malaria epitopes
        """
        params = {
            "limit": limit,
            "source_organism_iri": f"http://purl.obolibrary.org/obo/NCBITaxon_{self.ORGANISM_IDS['Plasmodium falciparum']}",
        }

        try:
            data = self._get(f"{self.BASE_URL}/epitope_search", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ========== REFERENCE DATA ==========

    def get_reference_antigens(
        self,
        organism: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get reference antigens for an organism.

        Args:
            organism: Organism name

        Returns:
            DataFrame with reference antigens
        """
        params = {}

        if organism:
            organism_id = self.ORGANISM_IDS.get(organism)
            if organism_id:
                params["source_organism_iri"] = f"http://purl.obolibrary.org/obo/NCBITaxon_{organism_id}"

        try:
            data = self._get(f"{self.BASE_URL}/antigen", params)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_hla_allele_frequencies(
        self,
        population: str = "World",
    ) -> pd.DataFrame:
        """
        Get HLA allele frequencies for a population.

        Args:
            population: Population name

        Returns:
            DataFrame with allele frequencies
        """
        # This typically requires the population coverage tool
        try:
            response = self.session.get(
                f"{self.TOOLS_URL}/population/allele_frequencies",
                params={"population": population},
                timeout=30,
            )
            response.raise_for_status()

            lines = response.text.strip().split("\n")
            if len(lines) > 1:
                headers = lines[0].split("\t")
                rows = [line.split("\t") for line in lines[1:]]
                return pd.DataFrame(rows, columns=headers)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ========== EPITOPE ANALYSIS ==========

    def get_epitope_summary(
        self,
        organism: str,
    ) -> dict:
        """
        Get summary statistics for epitopes from an organism.

        Args:
            organism: Organism name

        Returns:
            Dictionary with epitope summary
        """
        tcell = self.search_tcell_epitopes(organism=organism, limit=1000)
        bcell = self.search_bcell_epitopes(organism=organism, limit=1000)

        return {
            "organism": organism,
            "total_tcell_epitopes": len(tcell),
            "total_bcell_epitopes": len(bcell),
            "total_epitopes": len(tcell) + len(bcell),
        }
