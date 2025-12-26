"""
CARD (Comprehensive Antibiotic Resistance Database) API client.

Provides access to:
- Antibiotic resistance ontology (ARO)
- Resistance genes and mechanisms
- ESKAPE pathogen data
- Resistance detection models

CARD API: https://card.mcmaster.ca/api
"""

from typing import Optional

import pandas as pd
import requests

from ..config import settings


class CARDClient:
    """Client for CARD (Comprehensive Antibiotic Resistance Database) API."""

    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize CARD client.

        Args:
            api_url: API base URL (defaults to public CARD)
        """
        self.base_url = (api_url or settings.card.api_url).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

        # Cache for ontology data
        self._aro_cache = None
        self._drug_classes_cache = None

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict | list:
        """Make GET request to API."""
        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            timeout=settings.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_aro_terms(self, refresh: bool = False) -> pd.DataFrame:
        """
        Get Antibiotic Resistance Ontology terms.

        Args:
            refresh: Force refresh of cached data

        Returns:
            DataFrame with ARO terms and descriptions
        """
        if self._aro_cache is None or refresh:
            try:
                data = self._get("ontology")
                self._aro_cache = pd.DataFrame(data)
            except requests.exceptions.RequestException:
                # Fallback to basic structure if API unavailable
                return pd.DataFrame(columns=["aro_id", "name", "description", "category"])

        return self._aro_cache

    def search_aro(self, query: str) -> pd.DataFrame:
        """
        Search ARO for resistance terms.

        Args:
            query: Search query

        Returns:
            DataFrame with matching ARO terms
        """
        try:
            data = self._get("ontology/search", {"q": query})
            return pd.DataFrame(data)
        except requests.exceptions.RequestException:
            # Try local search in cache
            aro = self.get_aro_terms()
            if aro.empty:
                return pd.DataFrame()

            mask = aro.apply(
                lambda row: any(
                    query.lower() in str(v).lower() for v in row.values if isinstance(v, str)
                ),
                axis=1,
            )
            return aro[mask]

    def get_resistance_genes(self, pathogen: Optional[str] = None) -> pd.DataFrame:
        """
        Get antibiotic resistance genes.

        Args:
            pathogen: Filter by pathogen name (e.g., "Escherichia coli")

        Returns:
            DataFrame with resistance genes
        """
        try:
            params = {}
            if pathogen:
                params["pathogen"] = pathogen

            data = self._get("resistance-genes", params)
            return pd.DataFrame(data)
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def get_drug_classes(self, refresh: bool = False) -> pd.DataFrame:
        """
        Get antibiotic drug classes.

        Args:
            refresh: Force refresh of cached data

        Returns:
            DataFrame with drug classes
        """
        if self._drug_classes_cache is None or refresh:
            try:
                data = self._get("drug-classes")
                self._drug_classes_cache = pd.DataFrame(data)
            except requests.exceptions.RequestException:
                return pd.DataFrame(
                    {
                        "drug_class": [
                            "aminoglycoside",
                            "beta-lactam",
                            "fluoroquinolone",
                            "glycopeptide",
                            "macrolide",
                            "tetracycline",
                            "sulfonamide",
                            "carbapenem",
                            "cephalosporin",
                        ]
                    }
                )

        return self._drug_classes_cache

    def get_resistance_mechanisms(self) -> pd.DataFrame:
        """
        Get resistance mechanisms categories.

        Returns:
            DataFrame with resistance mechanisms
        """
        try:
            data = self._get("mechanisms")
            return pd.DataFrame(data)
        except requests.exceptions.RequestException:
            # Return common mechanisms if API unavailable
            return pd.DataFrame(
                {
                    "mechanism": [
                        "antibiotic efflux",
                        "antibiotic inactivation",
                        "antibiotic target alteration",
                        "antibiotic target protection",
                        "antibiotic target replacement",
                        "reduced permeability to antibiotic",
                    ]
                }
            )

    def get_eskape_pathogens(self) -> pd.DataFrame:
        """
        Get ESKAPE pathogen information.

        ESKAPE: Enterococcus faecium, Staphylococcus aureus,
        Klebsiella pneumoniae, Acinetobacter baumannii,
        Pseudomonas aeruginosa, Enterobacter spp.

        Returns:
            DataFrame with ESKAPE pathogen data
        """
        eskape = [
            {
                "code": "E",
                "pathogen": "Enterococcus faecium",
                "type": "Gram-positive",
                "key_resistance": "Vancomycin",
            },
            {
                "code": "S",
                "pathogen": "Staphylococcus aureus",
                "type": "Gram-positive",
                "key_resistance": "Methicillin (MRSA)",
            },
            {
                "code": "K",
                "pathogen": "Klebsiella pneumoniae",
                "type": "Gram-negative",
                "key_resistance": "Carbapenems (KPC)",
            },
            {
                "code": "A",
                "pathogen": "Acinetobacter baumannii",
                "type": "Gram-negative",
                "key_resistance": "Multi-drug",
            },
            {
                "code": "P",
                "pathogen": "Pseudomonas aeruginosa",
                "type": "Gram-negative",
                "key_resistance": "Multi-drug",
            },
            {
                "code": "E",
                "pathogen": "Enterobacter species",
                "type": "Gram-negative",
                "key_resistance": "Beta-lactams",
            },
        ]

        df = pd.DataFrame(eskape)

        # Try to enrich with API data
        for i, row in df.iterrows():
            genes = self.get_resistance_genes(row["pathogen"])
            if not genes.empty:
                df.at[i, "resistance_gene_count"] = len(genes)

        return df

    def get_gene_details(self, gene_name: str) -> dict:
        """
        Get detailed information about a resistance gene.

        Args:
            gene_name: Gene name or ARO accession

        Returns:
            Dictionary with gene details
        """
        try:
            data = self._get(f"gene/{gene_name}")
            return data
        except requests.exceptions.RequestException:
            return {}

    def get_pathogens_for_drug_class(self, drug_class: str) -> pd.DataFrame:
        """
        Get pathogens with resistance to a specific drug class.

        Args:
            drug_class: Drug class name (e.g., "carbapenem", "fluoroquinolone")

        Returns:
            DataFrame with pathogen-resistance associations
        """
        try:
            data = self._get("resistance/by-drug-class", {"class": drug_class})
            return pd.DataFrame(data)
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def get_prevalence_data(self, gene: Optional[str] = None, pathogen: Optional[str] = None) -> pd.DataFrame:
        """
        Get prevalence data for resistance genes.

        Args:
            gene: Filter by gene name
            pathogen: Filter by pathogen

        Returns:
            DataFrame with prevalence statistics
        """
        try:
            params = {}
            if gene:
                params["gene"] = gene
            if pathogen:
                params["pathogen"] = pathogen

            data = self._get("prevalence", params)
            return pd.DataFrame(data)
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def get_snp_data(self, gene: str) -> pd.DataFrame:
        """
        Get known SNPs associated with resistance in a gene.

        Args:
            gene: Gene name

        Returns:
            DataFrame with SNP positions and effects
        """
        try:
            data = self._get(f"snps/{gene}")
            return pd.DataFrame(data)
        except requests.exceptions.RequestException:
            return pd.DataFrame()

    def analyze_sequence(self, sequence: str, sequence_type: str = "nucleotide") -> dict:
        """
        Analyze a sequence for resistance genes using RGI (Resistance Gene Identifier).

        Note: Full RGI analysis may require local installation.
        This method provides a simplified API-based check.

        Args:
            sequence: DNA or protein sequence
            sequence_type: "nucleotide" or "protein"

        Returns:
            Dictionary with analysis results
        """
        try:
            response = self.session.post(
                f"{self.base_url}/analyze",
                json={"sequence": sequence, "type": sequence_type},
                timeout=120,  # Analysis may take time
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "note": "For full RGI analysis, install CARD locally: "
                "https://github.com/arpcard/rgi",
            }

    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of CARD detection models.

        Returns:
            DataFrame with model information
        """
        try:
            data = self._get("models")
            return pd.DataFrame(data)
        except requests.exceptions.RequestException:
            return pd.DataFrame(
                {
                    "model_type": [
                        "protein homolog",
                        "protein variant",
                        "protein overexpression",
                        "rRNA gene variant",
                    ],
                    "description": [
                        "Genes detected by similarity to known resistance genes",
                        "Genes requiring specific mutations for resistance",
                        "Genes causing resistance when overexpressed",
                        "rRNA mutations conferring resistance",
                    ],
                }
            )
