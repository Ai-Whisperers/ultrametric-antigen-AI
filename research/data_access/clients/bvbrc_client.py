"""
BV-BRC (Bacterial and Viral Bioinformatics Resource Center) API client.

Provides access to:
- Bacterial genomes and annotations
- Viral sequences
- AMR (Antimicrobial Resistance) data
- Protein families
- Specialty genes

BV-BRC API: https://www.bv-brc.org/api
Documentation: https://www.bv-brc.org/docs/api/
"""

from typing import Optional

import pandas as pd
import requests

from ..config import settings


class BVBRCClient:
    """Client for BV-BRC (Bacterial and Viral Bioinformatics Resource Center) API."""

    # Data types available in BV-BRC
    DATA_TYPES = [
        "genome",
        "genome_feature",
        "genome_sequence",
        "pathway",
        "sp_gene",  # Specialty genes
        "genome_amr",  # AMR phenotypes
        "taxonomy",
        "protein_family_ref",
        "subsystem",
        "protein_structure",
    ]

    # Common organism taxon IDs
    TAXON_IDS = {
        "Mycobacterium tuberculosis": 1773,
        "Treponema pallidum": 160,
        "Staphylococcus aureus": 1280,
        "Escherichia coli": 562,
        "Klebsiella pneumoniae": 573,
        "Pseudomonas aeruginosa": 287,
        "Acinetobacter baumannii": 470,
        "Enterococcus faecium": 1352,
        "Enterobacter cloacae": 550,
        "Salmonella enterica": 28901,
        "Neisseria gonorrhoeae": 485,
    }

    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize BV-BRC client.

        Args:
            api_url: API base URL (defaults to public BV-BRC)
        """
        self.base_url = (api_url or settings.bvbrc.api_url).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/rqlquery+x-www-form-urlencoded",
        })

    def _get(self, endpoint: str, params: Optional[dict] = None) -> list | dict:
        """Make GET request to API."""
        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            timeout=settings.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _query(
        self,
        data_type: str,
        query: str,
        select: Optional[list[str]] = None,
        limit: int = 25,
        offset: int = 0,
    ) -> pd.DataFrame:
        """
        Execute a query against BV-BRC data.

        Args:
            data_type: Type of data to query (genome, genome_feature, etc.)
            query: RQL query string
            select: Fields to return
            limit: Maximum results
            offset: Starting offset for pagination

        Returns:
            DataFrame with query results
        """
        url = f"{self.base_url}/{data_type}"

        # Build RQL query
        rql_parts = []
        if query:
            rql_parts.append(query)
        if select:
            rql_parts.append(f"select({','.join(select)})")
        rql_parts.append(f"limit({limit},{offset})")

        rql = "&".join(rql_parts)

        # BV-BRC API requires POST with RQL in body
        response = self.session.post(
            url,
            data=rql,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/rqlquery+x-www-form-urlencoded",
            },
            timeout=settings.timeout,
        )
        response.raise_for_status()
        data = response.json()

        return pd.DataFrame(data)

    def search_genomes(
        self,
        organism: Optional[str] = None,
        genome_name: Optional[str] = None,
        genome_status: Optional[str] = "Complete",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Search for bacterial/viral genomes.

        Args:
            organism: Organism name to search
            genome_name: Genome name pattern
            genome_status: Genome status (Complete, WGS, etc.) or None for any
            limit: Maximum results

        Returns:
            DataFrame with genome information
        """
        query_parts = []

        if organism:
            # Use taxon ID for reliable search
            taxon_id = self.TAXON_IDS.get(organism)
            if taxon_id:
                query_parts.append(f"eq(taxon_id,{taxon_id})")
            else:
                # Fallback to genome_name contains for unknown organisms
                query_parts.append(f"contains(genome_name,{organism.split()[0]})")
        if genome_name:
            query_parts.append(f"contains(genome_name,{genome_name})")
        if genome_status:
            query_parts.append(f"eq(genome_status,{genome_status})")

        query = "&".join(query_parts) if query_parts else ""

        select = [
            "genome_id",
            "genome_name",
            "taxon_id",
            "genome_status",
            "contigs",
            "sequences",
            "genome_length",
            "gc_content",
            "completion_date",
        ]

        try:
            return self._query("genome", query, select=select, limit=limit)
        except requests.exceptions.HTTPError:
            # Try without status filter if it fails
            if genome_status:
                query_parts = [p for p in query_parts if "genome_status" not in p]
                query = "&".join(query_parts) if query_parts else ""
                return self._query("genome", query, select=select, limit=limit)
            return pd.DataFrame()

    def get_genome(self, genome_id: str) -> dict:
        """
        Get details of a specific genome.

        Args:
            genome_id: BV-BRC genome ID

        Returns:
            Dictionary with genome details
        """
        try:
            data = self._get(f"genome/{genome_id}")
            return data[0] if isinstance(data, list) and data else data
        except (requests.exceptions.RequestException, IndexError):
            return {}

    def get_genome_features(
        self,
        genome_id: str,
        feature_type: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get features (genes, CDS, etc.) for a genome.

        Args:
            genome_id: BV-BRC genome ID
            feature_type: Filter by feature type (CDS, gene, tRNA, etc.)
            limit: Maximum results

        Returns:
            DataFrame with feature information
        """
        query_parts = [f"eq(genome_id,{genome_id})"]

        if feature_type:
            query_parts.append(f"eq(feature_type,{feature_type})")

        query = "&".join(query_parts)

        select = [
            "feature_id",
            "genome_id",
            "feature_type",
            "patric_id",
            "product",
            "gene",
            "start",
            "end",
            "strand",
            "na_length",
            "aa_length",
        ]

        return self._query("genome_feature", query, select=select, limit=limit)

    def get_amr_phenotypes(
        self,
        genome_id: Optional[str] = None,
        antibiotic: Optional[str] = None,
        phenotype: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get AMR (Antimicrobial Resistance) phenotype data.

        Args:
            genome_id: Filter by genome ID
            antibiotic: Filter by antibiotic name
            phenotype: Filter by phenotype (Resistant, Susceptible, Intermediate)
            limit: Maximum results

        Returns:
            DataFrame with AMR phenotype data
        """
        query_parts = []

        if genome_id:
            query_parts.append(f"eq(genome_id,{genome_id})")
        if antibiotic:
            query_parts.append(f"contains(antibiotic,{antibiotic})")
        if phenotype:
            query_parts.append(f"eq(resistant_phenotype,{phenotype})")

        query = "&".join(query_parts) if query_parts else "ne(genome_id,0)"

        select = [
            "genome_id",
            "genome_name",
            "antibiotic",
            "resistant_phenotype",
            "measurement",
            "measurement_unit",
            "laboratory_typing_method",
            "source",
        ]

        return self._query("genome_amr", query, select=select, limit=limit)

    def get_specialty_genes(
        self,
        genome_id: Optional[str] = None,
        property_type: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get specialty genes (virulence factors, AMR genes, etc.).

        Args:
            genome_id: Filter by genome ID
            property_type: Filter by property type (Virulence Factor, Antibiotic Resistance, etc.)
            limit: Maximum results

        Returns:
            DataFrame with specialty gene information
        """
        query_parts = []

        if genome_id:
            query_parts.append(f"eq(genome_id,{genome_id})")
        if property_type:
            query_parts.append(f"eq(property,{property_type})")

        query = "&".join(query_parts) if query_parts else "ne(genome_id,0)"

        select = [
            "genome_id",
            "genome_name",
            "feature_id",
            "gene",
            "product",
            "property",
            "source",
            "source_id",
            "organism",
        ]

        return self._query("sp_gene", query, select=select, limit=limit)

    def get_virulence_factors(self, genome_id: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        """
        Get virulence factor genes.

        Args:
            genome_id: Filter by genome ID
            limit: Maximum results

        Returns:
            DataFrame with virulence factor information
        """
        return self.get_specialty_genes(
            genome_id=genome_id,
            property_type="Virulence Factor",
            limit=limit,
        )

    def get_resistance_genes(self, genome_id: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        """
        Get antibiotic resistance genes.

        Args:
            genome_id: Filter by genome ID
            limit: Maximum results

        Returns:
            DataFrame with resistance gene information
        """
        return self.get_specialty_genes(
            genome_id=genome_id,
            property_type="Antibiotic Resistance",
            limit=limit,
        )

    def get_pathways(self, genome_id: str, limit: int = 200) -> pd.DataFrame:
        """
        Get metabolic pathways for a genome.

        Args:
            genome_id: BV-BRC genome ID
            limit: Maximum results

        Returns:
            DataFrame with pathway information
        """
        query = f"eq(genome_id,{genome_id})"

        select = [
            "genome_id",
            "genome_name",
            "pathway_id",
            "pathway_name",
            "pathway_class",
            "ec_number",
            "ec_description",
        ]

        return self._query("pathway", query, select=select, limit=limit)

    def get_protein_families(
        self,
        genome_id: Optional[str] = None,
        family_type: str = "global",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get protein family information.

        Args:
            genome_id: Filter by genome ID
            family_type: Family type (global, local)
            limit: Maximum results

        Returns:
            DataFrame with protein family data
        """
        query_parts = []

        if genome_id:
            query_parts.append(f"eq(genome_id,{genome_id})")
        query_parts.append(f"eq(family_type,{family_type})")

        query = "&".join(query_parts)

        return self._query("genome_feature", query, limit=limit)

    def get_subsystems(self, genome_id: str, limit: int = 200) -> pd.DataFrame:
        """
        Get subsystem information for a genome.

        Args:
            genome_id: BV-BRC genome ID
            limit: Maximum results

        Returns:
            DataFrame with subsystem information
        """
        query = f"eq(genome_id,{genome_id})"

        select = [
            "genome_id",
            "genome_name",
            "subsystem_id",
            "subsystem_name",
            "superclass",
            "class",
            "subclass",
            "role_id",
            "role_name",
        ]

        return self._query("subsystem", query, select=select, limit=limit)

    def search_by_taxonomy(self, taxon_id: int, limit: int = 100) -> pd.DataFrame:
        """
        Search genomes by taxonomy ID.

        Args:
            taxon_id: NCBI taxonomy ID
            limit: Maximum results

        Returns:
            DataFrame with matching genomes
        """
        query = f"eq(taxon_id,{taxon_id})"
        return self._query("genome", query, limit=limit)

    def get_tb_genomes(self, limit: int = 100) -> pd.DataFrame:
        """
        Get Mycobacterium tuberculosis genomes.

        Returns:
            DataFrame with TB genome information
        """
        return self.search_genomes(organism="Mycobacterium tuberculosis", limit=limit)

    def get_syphilis_genomes(self, limit: int = 100) -> pd.DataFrame:
        """
        Get Treponema pallidum (syphilis) genomes.

        Returns:
            DataFrame with syphilis pathogen genomes
        """
        return self.search_genomes(organism="Treponema pallidum", limit=limit)

    def get_eskape_genomes(self, pathogen: str, limit: int = 50) -> pd.DataFrame:
        """
        Get genomes for ESKAPE pathogens.

        Args:
            pathogen: ESKAPE pathogen name
                - "Enterococcus faecium"
                - "Staphylococcus aureus"
                - "Klebsiella pneumoniae"
                - "Acinetobacter baumannii"
                - "Pseudomonas aeruginosa"
                - "Enterobacter cloacae"
            limit: Maximum results

        Returns:
            DataFrame with pathogen genomes
        """
        return self.search_genomes(organism=pathogen, limit=limit)

    def get_sequence(self, feature_id: str) -> dict:
        """
        Get nucleotide and amino acid sequence for a feature.

        Args:
            feature_id: BV-BRC feature ID

        Returns:
            Dictionary with sequences
        """
        try:
            data = self._get(f"genome_feature/{feature_id}")
            if isinstance(data, list) and data:
                feature = data[0]
                return {
                    "feature_id": feature.get("feature_id"),
                    "gene": feature.get("gene"),
                    "product": feature.get("product"),
                    "na_sequence": feature.get("na_sequence"),
                    "aa_sequence": feature.get("aa_sequence"),
                }
            return {}
        except requests.exceptions.RequestException:
            return {}

    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary of available data in BV-BRC.

        Returns:
            DataFrame with data type counts
        """
        summaries = []

        for data_type in ["genome", "genome_amr", "sp_gene"]:
            try:
                # Get count by limiting to 1 and checking total
                result = self._query(data_type, "ne(genome_id,0)", limit=1)
                summaries.append({"data_type": data_type, "available": True})
            except Exception:
                summaries.append({"data_type": data_type, "available": False})

        return pd.DataFrame(summaries)

    # ========== EPITOPE DATA ==========

    def get_epitopes(
        self,
        organism: Optional[str] = None,
        epitope_type: Optional[str] = None,
        protein: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get epitope data (B-cell and T-cell epitopes).

        Args:
            organism: Filter by organism name
            epitope_type: Filter by type (bcell, tcell)
            protein: Filter by protein name
            limit: Maximum results

        Returns:
            DataFrame with epitope information
        """
        query_parts = []

        if organism:
            taxon_id = self.TAXON_IDS.get(organism)
            if taxon_id:
                query_parts.append(f"eq(taxon_id,{taxon_id})")
            else:
                query_parts.append(f"contains(organism,{organism.split()[0]})")

        if epitope_type:
            query_parts.append(f"eq(epitope_type,{epitope_type})")

        if protein:
            query_parts.append(f"contains(protein,{protein})")

        query = "&".join(query_parts) if query_parts else "ne(epitope_id,0)"

        select = [
            "epitope_id",
            "epitope_type",
            "epitope_sequence",
            "organism",
            "protein",
            "start",
            "end",
            "bcell_assay",
            "mhc_allele",
        ]

        try:
            return self._query("epitope", query, select=select, limit=limit)
        except Exception:
            return pd.DataFrame()

    def get_epitope_summary(
        self,
        organism: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Get epitope summary by organism.

        Args:
            organism: Filter by organism
            limit: Maximum results

        Returns:
            DataFrame with epitope summary
        """
        epitopes = self.get_epitopes(organism=organism, limit=limit)

        if epitopes.empty:
            return pd.DataFrame()

        summary = epitopes.groupby(["organism", "epitope_type", "protein"]).size()
        return summary.reset_index(name="count")

    # ========== PROTEIN STRUCTURES ==========

    def search_protein_structures(
        self,
        organism: Optional[str] = None,
        gene: Optional[str] = None,
        pdb_id: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Search for protein structures.

        Args:
            organism: Filter by organism
            gene: Filter by gene name
            pdb_id: Filter by PDB ID
            limit: Maximum results

        Returns:
            DataFrame with structure information
        """
        query_parts = []

        if organism:
            taxon_id = self.TAXON_IDS.get(organism)
            if taxon_id:
                query_parts.append(f"eq(taxon_id,{taxon_id})")

        if gene:
            query_parts.append(f"contains(gene,{gene})")

        if pdb_id:
            query_parts.append(f"eq(pdb_id,{pdb_id})")

        query = "&".join(query_parts) if query_parts else "ne(pdb_id,null)"

        select = [
            "pdb_id",
            "title",
            "organism_name",
            "gene",
            "product",
            "sequence_md5",
            "method",
            "resolution",
            "release_date",
        ]

        try:
            return self._query("protein_structure", query, select=select, limit=limit)
        except Exception:
            return pd.DataFrame()

    def get_structure_details(self, pdb_id: str) -> dict:
        """
        Get details of a specific protein structure.

        Args:
            pdb_id: PDB structure ID

        Returns:
            Dictionary with structure details
        """
        try:
            data = self._get(f"protein_structure/{pdb_id}")
            return data[0] if isinstance(data, list) and data else data
        except Exception:
            return {}

    # ========== GENE ONTOLOGY ==========

    def get_gene_ontology(
        self,
        genome_id: Optional[str] = None,
        go_term: Optional[str] = None,
        ontology: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get Gene Ontology annotations.

        Args:
            genome_id: Filter by genome ID
            go_term: Filter by GO term ID (e.g., GO:0006915)
            ontology: Filter by ontology type (biological_process, molecular_function, cellular_component)
            limit: Maximum results

        Returns:
            DataFrame with GO annotations
        """
        query_parts = []

        if genome_id:
            query_parts.append(f"eq(genome_id,{genome_id})")

        if go_term:
            query_parts.append(f"eq(go_id,{go_term})")

        if ontology:
            query_parts.append(f"eq(ontology,{ontology})")

        query = "&".join(query_parts) if query_parts else "ne(feature_id,0)"

        select = [
            "feature_id",
            "genome_id",
            "gene",
            "product",
            "go_id",
            "go_term",
            "ontology",
            "evidence_code",
        ]

        try:
            # GO annotations are often in genome_feature with GO fields
            return self._query("genome_feature", query + "&ne(go_id,null)", select=select, limit=limit)
        except Exception:
            return pd.DataFrame()

    def get_go_enrichment(
        self,
        genome_id: str,
        ontology: str = "biological_process",
    ) -> pd.DataFrame:
        """
        Get GO term enrichment for a genome.

        Args:
            genome_id: BV-BRC genome ID
            ontology: Ontology type

        Returns:
            DataFrame with GO term frequencies
        """
        go_data = self.get_gene_ontology(genome_id=genome_id, ontology=ontology, limit=2000)

        if go_data.empty:
            return pd.DataFrame()

        # Count GO terms
        if "go_term" in go_data.columns:
            counts = go_data.groupby(["go_id", "go_term"]).size()
            return counts.reset_index(name="gene_count").sort_values("gene_count", ascending=False)

        return pd.DataFrame()

    # ========== SPECIALTY GENES (Extended) ==========

    def get_drug_targets(
        self,
        genome_id: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get drug target genes.

        Args:
            genome_id: Filter by genome ID
            limit: Maximum results

        Returns:
            DataFrame with drug target information
        """
        return self.get_specialty_genes(
            genome_id=genome_id,
            property_type="Drug Target",
            limit=limit,
        )

    def get_transporters(
        self,
        genome_id: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get transporter genes.

        Args:
            genome_id: Filter by genome ID
            limit: Maximum results

        Returns:
            DataFrame with transporter information
        """
        return self.get_specialty_genes(
            genome_id=genome_id,
            property_type="Transporter",
            limit=limit,
        )

    def get_essential_genes(
        self,
        genome_id: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get essential genes.

        Args:
            genome_id: Filter by genome ID
            limit: Maximum results

        Returns:
            DataFrame with essential gene information
        """
        return self.get_specialty_genes(
            genome_id=genome_id,
            property_type="Essential Gene",
            limit=limit,
        )

    # ========== PROTEIN FAMILIES (Extended) ==========

    def get_figfam_assignments(
        self,
        genome_id: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get FIGfam protein family assignments for a genome.

        Args:
            genome_id: BV-BRC genome ID
            limit: Maximum results

        Returns:
            DataFrame with FIGfam assignments
        """
        query = f"eq(genome_id,{genome_id})&ne(figfam_id,null)"

        select = [
            "feature_id",
            "gene",
            "product",
            "figfam_id",
            "plfam_id",
            "pgfam_id",
        ]

        try:
            return self._query("genome_feature", query, select=select, limit=limit)
        except Exception:
            return pd.DataFrame()

    # ========== COMPARATIVE GENOMICS ==========

    def compare_genomes(
        self,
        genome_ids: list[str],
        feature_type: str = "CDS",
    ) -> pd.DataFrame:
        """
        Get features across multiple genomes for comparison.

        Args:
            genome_ids: List of genome IDs to compare
            feature_type: Type of features to compare

        Returns:
            DataFrame with features from all genomes
        """
        all_features = []

        for genome_id in genome_ids:
            features = self.get_genome_features(genome_id, feature_type=feature_type)
            if not features.empty:
                all_features.append(features)

        if all_features:
            return pd.concat(all_features, ignore_index=True)
        return pd.DataFrame()

    def get_pan_genome(
        self,
        organism: str,
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Get pan-genome analysis data (shared and unique genes across genomes).

        Args:
            organism: Organism name
            limit: Maximum genomes to include

        Returns:
            DataFrame with pan-genome data
        """
        # Get genomes for organism
        genomes = self.search_genomes(organism=organism, limit=limit)

        if genomes.empty:
            return pd.DataFrame()

        # Get protein families across genomes
        all_families = []
        for genome_id in genomes["genome_id"].tolist()[:10]:  # Limit for performance
            try:
                families = self.get_figfam_assignments(genome_id, limit=1000)
                if not families.empty:
                    families["genome_id"] = genome_id
                    all_families.append(families)
            except Exception:
                continue

        if all_families:
            combined = pd.concat(all_families, ignore_index=True)
            # Count occurrences of each family
            if "figfam_id" in combined.columns:
                counts = combined.groupby("figfam_id").agg(
                    genome_count=("genome_id", "nunique"),
                    total_genes=("feature_id", "count"),
                ).reset_index()
                return counts.sort_values("genome_count", ascending=False)

        return pd.DataFrame()

    # ========== SURVEILLANCE DATA ==========

    def get_surveillance_data(
        self,
        organism: Optional[str] = None,
        collection_year: Optional[int] = None,
        country: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get epidemiological surveillance data.

        Args:
            organism: Filter by organism
            collection_year: Filter by collection year
            country: Filter by country
            limit: Maximum results

        Returns:
            DataFrame with surveillance data
        """
        query_parts = []

        if organism:
            taxon_id = self.TAXON_IDS.get(organism)
            if taxon_id:
                query_parts.append(f"eq(taxon_id,{taxon_id})")

        if collection_year:
            query_parts.append(f"eq(collection_year,{collection_year})")

        if country:
            query_parts.append(f"eq(isolation_country,{country})")

        query = "&".join(query_parts) if query_parts else "ne(genome_id,0)"

        select = [
            "genome_id",
            "genome_name",
            "collection_year",
            "collection_date",
            "isolation_country",
            "geographic_location",
            "host_name",
            "host_health",
            "isolation_source",
        ]

        try:
            return self._query("genome", query, select=select, limit=limit)
        except Exception:
            return pd.DataFrame()

    # ========== ANTIBIOTIC RESISTANCE (Extended) ==========

    def get_amr_mechanisms(
        self,
        genome_id: Optional[str] = None,
        mechanism: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get AMR mechanism annotations.

        Args:
            genome_id: Filter by genome ID
            mechanism: Filter by mechanism type
            limit: Maximum results

        Returns:
            DataFrame with AMR mechanism data
        """
        query_parts = []

        if genome_id:
            query_parts.append(f"eq(genome_id,{genome_id})")

        if mechanism:
            query_parts.append(f"contains(mechanism,{mechanism})")

        query = "&".join(query_parts) if query_parts else "ne(genome_id,0)"

        select = [
            "genome_id",
            "genome_name",
            "gene",
            "product",
            "property",
            "source",
            "mechanism",
            "classification",
        ]

        try:
            return self._query("sp_gene", query + "&eq(property,Antibiotic Resistance)", select=select, limit=limit)
        except Exception:
            return pd.DataFrame()

    def get_amr_by_antibiotic_class(
        self,
        antibiotic_class: str,
        organism: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Get AMR data filtered by antibiotic class.

        Args:
            antibiotic_class: Antibiotic class (e.g., "Fluoroquinolone", "Beta-lactam")
            organism: Optional organism filter
            limit: Maximum results

        Returns:
            DataFrame with AMR phenotype data
        """
        query_parts = [f"contains(antibiotic,{antibiotic_class})"]

        if organism:
            taxon_id = self.TAXON_IDS.get(organism)
            if taxon_id:
                query_parts.append(f"eq(taxon_id,{taxon_id})")

        query = "&".join(query_parts)

        return self._query("genome_amr", query, limit=limit)
