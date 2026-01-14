"""
UniProt REST API client for protein sequence and function data.

Provides access to:
- Protein sequences and annotations
- Protein function and structure information
- Cross-references to other databases
- Taxonomy and organism data
- Disease associations
- Post-translational modifications

UniProt API: https://rest.uniprot.org/
Documentation: https://www.uniprot.org/help/api
"""

from typing import Optional

import pandas as pd
import requests

from ..config import settings


class UniProtClient:
    """Client for UniProt REST API."""

    # Base URL for UniProt REST API
    BASE_URL = "https://rest.uniprot.org"

    # Common organism IDs
    ORGANISM_IDS = {
        "human": 9606,
        "Homo sapiens": 9606,
        "mouse": 10090,
        "Mus musculus": 10090,
        "rat": 10116,
        "Rattus norvegicus": 10116,
        "HIV-1": 11676,
        "HIV-2": 11709,
        "HBV": 10407,
        "HCV": 11103,
        "SARS-CoV-2": 2697049,
        "Mycobacterium tuberculosis": 83332,
        "Escherichia coli": 83333,
        "Plasmodium falciparum": 36329,
    }

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize UniProt client.

        Args:
            base_url: API base URL (defaults to public UniProt)
        """
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request to API."""
        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            timeout=settings.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _search(
        self,
        query: str,
        database: str = "uniprotkb",
        fields: Optional[list[str]] = None,
        size: int = 100,
        format: str = "json",
    ) -> pd.DataFrame:
        """
        Search UniProt databases.

        Args:
            query: Search query in UniProt query language
            database: Database to search (uniprotkb, uniref, uniparc)
            fields: Fields to return
            size: Maximum results
            format: Response format

        Returns:
            DataFrame with search results
        """
        params = {
            "query": query,
            "size": size,
            "format": format,
        }

        if fields:
            params["fields"] = ",".join(fields)

        response = self.session.get(
            f"{self.base_url}/{database}/search",
            params=params,
            timeout=settings.timeout,
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            return pd.DataFrame()

        # Flatten nested structures for DataFrame
        rows = []
        for entry in results:
            row = self._flatten_entry(entry)
            rows.append(row)

        return pd.DataFrame(rows)

    def _flatten_entry(self, entry: dict) -> dict:
        """Flatten a UniProt entry for DataFrame conversion."""
        flat = {
            "accession": entry.get("primaryAccession"),
            "entry_type": entry.get("entryType"),
            "uniprotkb_id": entry.get("uniProtkbId"),
        }

        # Organism
        if "organism" in entry:
            org = entry["organism"]
            flat["organism"] = org.get("scientificName")
            flat["taxon_id"] = org.get("taxonId")

        # Protein names
        if "proteinDescription" in entry:
            desc = entry["proteinDescription"]
            if "recommendedName" in desc:
                flat["protein_name"] = desc["recommendedName"].get("fullName", {}).get("value")

        # Gene names
        if "genes" in entry and entry["genes"]:
            gene = entry["genes"][0]
            if "geneName" in gene:
                flat["gene_name"] = gene["geneName"].get("value")

        # Sequence
        if "sequence" in entry:
            seq = entry["sequence"]
            flat["sequence_length"] = seq.get("length")
            flat["sequence_mass"] = seq.get("molWeight")
            flat["sequence"] = seq.get("value")

        # Keywords
        if "keywords" in entry:
            flat["keywords"] = ", ".join(kw.get("name", "") for kw in entry["keywords"])

        return flat

    # ========== PROTEIN SEARCH ==========

    def search_proteins(
        self,
        query: str,
        organism: Optional[str] = None,
        reviewed: Optional[bool] = True,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Search for proteins in UniProtKB.

        Args:
            query: Search query (gene name, protein name, keyword)
            organism: Organism name or taxon ID
            reviewed: If True, only Swiss-Prot (reviewed) entries
            limit: Maximum results

        Returns:
            DataFrame with protein information
        """
        query_parts = [query]

        if organism:
            taxon_id = self.ORGANISM_IDS.get(organism, organism)
            query_parts.append(f"organism_id:{taxon_id}")

        if reviewed:
            query_parts.append("reviewed:true")

        full_query = " AND ".join(f"({p})" for p in query_parts)

        fields = [
            "accession",
            "id",
            "protein_name",
            "gene_names",
            "organism_name",
            "length",
            "reviewed",
        ]

        return self._search(full_query, fields=fields, size=limit)

    def get_protein(self, accession: str) -> dict:
        """
        Get detailed information for a protein.

        Args:
            accession: UniProt accession (e.g., P53_HUMAN, P04637)

        Returns:
            Dictionary with protein details
        """
        try:
            return self._get(f"uniprotkb/{accession}")
        except requests.exceptions.HTTPError:
            return {}

    def get_protein_sequence(self, accession: str) -> str:
        """
        Get protein sequence in FASTA format.

        Args:
            accession: UniProt accession

        Returns:
            FASTA sequence
        """
        response = self.session.get(
            f"{self.base_url}/uniprotkb/{accession}.fasta",
            timeout=settings.timeout,
        )
        response.raise_for_status()
        return response.text

    # ========== ORGANISM-SPECIFIC SEARCHES ==========

    def get_human_proteins(
        self,
        keyword: Optional[str] = None,
        gene: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get human proteins.

        Args:
            keyword: Optional keyword filter
            gene: Optional gene name filter
            limit: Maximum results

        Returns:
            DataFrame with human proteins
        """
        query_parts = ["organism_id:9606"]

        if keyword:
            query_parts.append(f"keyword:{keyword}")
        if gene:
            query_parts.append(f"gene:{gene}")

        query = " AND ".join(query_parts)
        return self._search(query, size=limit)

    def get_hiv_proteins(
        self,
        gene: Optional[str] = None,
        strain: str = "HIV-1",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get HIV proteins.

        Args:
            gene: Gene name filter (gag, pol, env, etc.)
            strain: HIV-1 or HIV-2
            limit: Maximum results

        Returns:
            DataFrame with HIV proteins
        """
        taxon_id = self.ORGANISM_IDS.get(strain, 11676)
        query_parts = [f"organism_id:{taxon_id}"]

        if gene:
            query_parts.append(f"gene:{gene}")

        query = " AND ".join(query_parts)
        return self._search(query, size=limit)

    def get_viral_proteins(
        self,
        virus: str,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get proteins for a virus.

        Args:
            virus: Virus name (HIV-1, HBV, HCV, SARS-CoV-2)
            limit: Maximum results

        Returns:
            DataFrame with viral proteins
        """
        taxon_id = self.ORGANISM_IDS.get(virus)
        if not taxon_id:
            # Search by name
            return self._search(f"organism_name:{virus}", size=limit)

        return self._search(f"organism_id:{taxon_id}", size=limit)

    def get_pathogen_proteins(
        self,
        pathogen: str,
        virulence_only: bool = False,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get proteins for a pathogen.

        Args:
            pathogen: Pathogen name
            virulence_only: Only return virulence factors
            limit: Maximum results

        Returns:
            DataFrame with pathogen proteins
        """
        query_parts = []

        taxon_id = self.ORGANISM_IDS.get(pathogen)
        if taxon_id:
            query_parts.append(f"organism_id:{taxon_id}")
        else:
            query_parts.append(f"organism_name:{pathogen}")

        if virulence_only:
            query_parts.append("keyword:Virulence")

        query = " AND ".join(query_parts)
        return self._search(query, size=limit)

    # ========== FUNCTION SEARCHES ==========

    def search_by_function(
        self,
        function_term: str,
        organism: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Search proteins by function.

        Args:
            function_term: Function description term
            organism: Optional organism filter
            limit: Maximum results

        Returns:
            DataFrame with proteins matching function
        """
        query_parts = [f'cc_function:"{function_term}"']

        if organism:
            taxon_id = self.ORGANISM_IDS.get(organism, organism)
            query_parts.append(f"organism_id:{taxon_id}")

        query = " AND ".join(query_parts)
        return self._search(query, size=limit)

    def search_by_keyword(
        self,
        keyword: str,
        organism: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Search proteins by UniProt keyword.

        Args:
            keyword: UniProt keyword (e.g., "Kinase", "Receptor", "Membrane")
            organism: Optional organism filter
            limit: Maximum results

        Returns:
            DataFrame with proteins matching keyword
        """
        query_parts = [f"keyword:{keyword}"]

        if organism:
            taxon_id = self.ORGANISM_IDS.get(organism, organism)
            query_parts.append(f"organism_id:{taxon_id}")

        query = " AND ".join(query_parts)
        return self._search(query, size=limit)

    def get_enzymes(
        self,
        ec_number: Optional[str] = None,
        organism: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get enzyme proteins.

        Args:
            ec_number: EC classification number (e.g., "3.4.23" for aspartic proteases)
            organism: Optional organism filter
            limit: Maximum results

        Returns:
            DataFrame with enzyme proteins
        """
        query_parts = []

        if ec_number:
            query_parts.append(f"ec:{ec_number}")
        else:
            query_parts.append("ec:*")

        if organism:
            taxon_id = self.ORGANISM_IDS.get(organism, organism)
            query_parts.append(f"organism_id:{taxon_id}")

        query = " AND ".join(query_parts)
        return self._search(query, size=limit)

    # ========== DISEASE ASSOCIATIONS ==========

    def get_disease_proteins(
        self,
        disease: str,
        organism: str = "human",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get proteins associated with a disease.

        Args:
            disease: Disease name or OMIM ID
            organism: Organism name
            limit: Maximum results

        Returns:
            DataFrame with disease-associated proteins
        """
        taxon_id = self.ORGANISM_IDS.get(organism, 9606)
        query = f'cc_disease:"{disease}" AND organism_id:{taxon_id}'
        return self._search(query, size=limit)

    def get_cancer_proteins(
        self,
        cancer_type: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get proteins associated with cancer.

        Args:
            cancer_type: Specific cancer type
            limit: Maximum results

        Returns:
            DataFrame with cancer-associated proteins
        """
        if cancer_type:
            query = f'cc_disease:"{cancer_type}" AND organism_id:9606'
        else:
            query = 'keyword:"Proto-oncogene" OR keyword:"Tumor suppressor" AND organism_id:9606'

        return self._search(query, size=limit)

    # ========== DRUG TARGETS ==========

    def get_drug_targets(
        self,
        drug_name: Optional[str] = None,
        organism: str = "human",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get proteins that are drug targets.

        Args:
            drug_name: Optional drug name filter
            organism: Organism name
            limit: Maximum results

        Returns:
            DataFrame with drug target proteins
        """
        taxon_id = self.ORGANISM_IDS.get(organism, 9606)

        if drug_name:
            query = f'cc_pharmaceutical:"{drug_name}" AND organism_id:{taxon_id}'
        else:
            query = f"cc_pharmaceutical:* AND organism_id:{taxon_id}"

        return self._search(query, size=limit)

    # ========== STRUCTURAL INFORMATION ==========

    def get_proteins_with_structure(
        self,
        organism: Optional[str] = None,
        pdb_id: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get proteins with 3D structure.

        Args:
            organism: Optional organism filter
            pdb_id: Optional PDB ID filter
            limit: Maximum results

        Returns:
            DataFrame with proteins having structures
        """
        query_parts = ["database:pdb"]

        if organism:
            taxon_id = self.ORGANISM_IDS.get(organism, organism)
            query_parts.append(f"organism_id:{taxon_id}")

        if pdb_id:
            query_parts.append(f"xref:pdb-{pdb_id}")

        query = " AND ".join(query_parts)
        return self._search(query, size=limit)

    # ========== CROSS-REFERENCES ==========

    def get_protein_xrefs(self, accession: str) -> pd.DataFrame:
        """
        Get cross-references for a protein.

        Args:
            accession: UniProt accession

        Returns:
            DataFrame with database cross-references
        """
        protein = self.get_protein(accession)

        if not protein or "uniProtKBCrossReferences" not in protein:
            return pd.DataFrame()

        xrefs = protein["uniProtKBCrossReferences"]
        rows = []

        for xref in xrefs:
            row = {
                "database": xref.get("database"),
                "id": xref.get("id"),
            }
            # Add properties
            for prop in xref.get("properties", []):
                row[prop.get("key")] = prop.get("value")
            rows.append(row)

        return pd.DataFrame(rows)

    # ========== POST-TRANSLATIONAL MODIFICATIONS ==========

    def get_ptm_proteins(
        self,
        ptm_type: str,
        organism: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get proteins with specific post-translational modifications.

        Args:
            ptm_type: PTM type (Phosphorylation, Glycosylation, Acetylation, etc.)
            organism: Optional organism filter
            limit: Maximum results

        Returns:
            DataFrame with PTM proteins
        """
        query_parts = [f'ft_mod_res:"{ptm_type}"']

        if organism:
            taxon_id = self.ORGANISM_IDS.get(organism, organism)
            query_parts.append(f"organism_id:{taxon_id}")

        query = " AND ".join(query_parts)
        return self._search(query, size=limit)

    # ========== SEQUENCE FEATURES ==========

    def get_protein_features(self, accession: str) -> pd.DataFrame:
        """
        Get sequence features for a protein.

        Args:
            accession: UniProt accession

        Returns:
            DataFrame with sequence features (domains, sites, etc.)
        """
        protein = self.get_protein(accession)

        if not protein or "features" not in protein:
            return pd.DataFrame()

        features = protein["features"]
        rows = []

        for feat in features:
            row = {
                "type": feat.get("type"),
                "description": feat.get("description"),
                "start": feat.get("location", {}).get("start", {}).get("value"),
                "end": feat.get("location", {}).get("end", {}).get("value"),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    # ========== BATCH OPERATIONS ==========

    def batch_get_proteins(self, accessions: list[str]) -> pd.DataFrame:
        """
        Get multiple proteins by accession.

        Args:
            accessions: List of UniProt accessions

        Returns:
            DataFrame with protein information
        """
        if not accessions:
            return pd.DataFrame()

        query = " OR ".join(f"accession:{acc}" for acc in accessions[:100])  # Limit batch size
        return self._search(query, size=len(accessions))

    def get_proteome(
        self,
        organism: str,
        reference_only: bool = True,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get proteins from an organism's proteome.

        Args:
            organism: Organism name
            reference_only: Only reference proteome
            limit: Maximum results

        Returns:
            DataFrame with proteome proteins
        """
        taxon_id = self.ORGANISM_IDS.get(organism)
        if not taxon_id:
            return pd.DataFrame()

        query_parts = [f"organism_id:{taxon_id}"]

        if reference_only:
            query_parts.append("proteome:*")

        query = " AND ".join(query_parts)
        return self._search(query, size=limit)
