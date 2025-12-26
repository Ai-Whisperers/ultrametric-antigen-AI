"""
NCBI/Entrez API client for sequence data retrieval.

Provides access to:
- GenBank sequences (HIV, SARS-CoV-2, Influenza, etc.)
- PubMed literature
- Taxonomy information
- Protein sequences

Uses Biopython's Entrez module with proper rate limiting.
"""

import time
from typing import Iterator, Optional

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

from ..config import settings


class NCBIClient:
    """Client for NCBI/Entrez API access."""

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize NCBI client.

        Args:
            email: Email for NCBI identification (required by NCBI)
            api_key: Optional API key for higher rate limits
        """
        self.email = email or settings.ncbi.email
        self.api_key = api_key or settings.ncbi.api_key

        if not self.email:
            raise ValueError(
                "NCBI requires an email address. Set NCBI_EMAIL in .env or pass email parameter."
            )

        # Configure Entrez
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key

        # Rate limiting
        self._last_request_time = 0
        self._min_interval = 1.0 / (10 if self.api_key else 3)

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def search(
        self,
        database: str,
        query: str,
        max_results: int = 100,
        retstart: int = 0,
    ) -> dict:
        """
        Search NCBI database.

        Args:
            database: NCBI database name (nucleotide, protein, pubmed, taxonomy)
            query: Search query string
            max_results: Maximum number of results to return
            retstart: Starting index for pagination

        Returns:
            Dictionary with search results including IDs and count
        """
        self._rate_limit()

        with Entrez.esearch(
            db=database,
            term=query,
            retmax=max_results,
            retstart=retstart,
            usehistory="y",
        ) as handle:
            results = Entrez.read(handle)

        return {
            "ids": results["IdList"],
            "count": int(results["Count"]),
            "query_key": results.get("QueryKey"),
            "webenv": results.get("WebEnv"),
        }

    def fetch_sequences(
        self,
        ids: list[str],
        database: str = "nucleotide",
        format: str = "fasta",
    ) -> Iterator[SeqRecord]:
        """
        Fetch sequences by ID.

        Args:
            ids: List of sequence IDs (accession numbers or GIs)
            database: NCBI database (nucleotide, protein)
            format: Output format (fasta, genbank, gb)

        Yields:
            BioPython SeqRecord objects
        """
        # Fetch in batches to avoid timeouts
        batch_size = 100

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            self._rate_limit()

            with Entrez.efetch(
                db=database,
                id=",".join(batch_ids),
                rettype=format,
                retmode="text",
            ) as handle:
                for record in SeqIO.parse(handle, format):
                    yield record

    def fetch_genbank(self, accession: str) -> SeqRecord:
        """
        Fetch a single GenBank record.

        Args:
            accession: GenBank accession number

        Returns:
            BioPython SeqRecord with full GenBank annotations
        """
        self._rate_limit()

        with Entrez.efetch(
            db="nucleotide",
            id=accession,
            rettype="gb",
            retmode="text",
        ) as handle:
            record = SeqIO.read(handle, "genbank")

        return record

    def search_hiv_sequences(
        self,
        subtype: Optional[str] = None,
        gene: Optional[str] = None,
        country: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search for HIV sequences with filters.

        Args:
            subtype: HIV subtype (A, B, C, etc.)
            gene: Gene name (pol, env, gag, etc.)
            country: Country of origin
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = ["HIV-1[Organism]"]

        if subtype:
            query_parts.append(f"subtype {subtype}[Title]")
        if gene:
            query_parts.append(f"{gene}[Gene Name]")
        if country:
            query_parts.append(f"{country}[Country]")

        query = " AND ".join(query_parts)
        return self.search("nucleotide", query, max_results)

    def search_sars_cov2_sequences(
        self,
        lineage: Optional[str] = None,
        gene: Optional[str] = None,
        country: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search for SARS-CoV-2 sequences.

        Args:
            lineage: Variant lineage (e.g., BA.1, XBB.1.5)
            gene: Gene name (S, N, ORF1ab, etc.)
            country: Country of origin
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = ["SARS-CoV-2[Organism]"]

        if lineage:
            query_parts.append(f"{lineage}[Title]")
        if gene:
            query_parts.append(f"{gene}[Gene Name]")
        if country:
            query_parts.append(f"{country}[Country]")

        query = " AND ".join(query_parts)
        return self.search("nucleotide", query, max_results)

    def search_influenza_sequences(
        self,
        type_: str = "A",
        subtype: Optional[str] = None,
        segment: Optional[int] = None,
        host: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search for Influenza sequences.

        Args:
            type_: Influenza type (A, B, C)
            subtype: HA/NA subtype (e.g., H1N1, H3N2)
            segment: Genome segment number (1-8)
            host: Host organism
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = [f"Influenza {type_} virus[Organism]"]

        if subtype:
            query_parts.append(f"{subtype}[Title]")
        if segment:
            segment_names = {
                1: "PB2",
                2: "PB1",
                3: "PA",
                4: "HA",
                5: "NP",
                6: "NA",
                7: "M",
                8: "NS",
            }
            if segment in segment_names:
                query_parts.append(f"{segment_names[segment]}[Gene Name]")
        if host:
            query_parts.append(f"{host}[Host]")

        query = " AND ".join(query_parts)
        return self.search("nucleotide", query, max_results)

    def search_tuberculosis_sequences(
        self,
        gene: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search for Mycobacterium tuberculosis sequences.

        Args:
            gene: Gene name (rpoB, katG, inhA, etc.)
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = ["Mycobacterium tuberculosis[Organism]"]

        if gene:
            query_parts.append(f"{gene}[Gene Name]")

        query = " AND ".join(query_parts)
        return self.search("nucleotide", query, max_results)

    def get_sequence_summary(self, ids: list[str], database: str = "nucleotide") -> pd.DataFrame:
        """
        Get summary information for sequences.

        Args:
            ids: List of sequence IDs
            database: NCBI database

        Returns:
            DataFrame with sequence metadata
        """
        self._rate_limit()

        with Entrez.esummary(db=database, id=",".join(ids)) as handle:
            records = Entrez.read(handle)

        data = []
        for record in records:
            data.append(
                {
                    "id": record.get("Id"),
                    "accession": record.get("AccessionVersion"),
                    "title": record.get("Title"),
                    "length": record.get("Length"),
                    "create_date": record.get("CreateDate"),
                    "update_date": record.get("UpdateDate"),
                    "organism": record.get("Organism"),
                }
            )

        return pd.DataFrame(data)

    def fetch_pubmed_abstracts(self, query: str, max_results: int = 100) -> pd.DataFrame:
        """
        Search PubMed and fetch abstracts.

        Args:
            query: PubMed search query
            max_results: Maximum results

        Returns:
            DataFrame with article information
        """
        # Search PubMed
        search_results = self.search("pubmed", query, max_results)

        if not search_results["ids"]:
            return pd.DataFrame()

        # Fetch details
        self._rate_limit()
        with Entrez.efetch(
            db="pubmed",
            id=",".join(search_results["ids"]),
            rettype="abstract",
            retmode="xml",
        ) as handle:
            records = Entrez.read(handle)

        data = []
        for article in records.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            article_data = medline.get("Article", {})

            # Extract abstract
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts)

            data.append(
                {
                    "pmid": str(medline.get("PMID", "")),
                    "title": str(article_data.get("ArticleTitle", "")),
                    "abstract": abstract,
                    "journal": article_data.get("Journal", {}).get("Title", ""),
                    "year": article_data.get("Journal", {})
                    .get("JournalIssue", {})
                    .get("PubDate", {})
                    .get("Year", ""),
                }
            )

        return pd.DataFrame(data)

    # =========================================================================
    # Additional Virus Searches
    # =========================================================================

    def search_hbv_sequences(
        self,
        genotype: Optional[str] = None,
        gene: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search for Hepatitis B virus sequences.

        Args:
            genotype: HBV genotype (A, B, C, D, E, F, G, H)
            gene: Gene name (S, C, P, X - surface, core, polymerase, X)
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = ["Hepatitis B virus[Organism]"]

        if genotype:
            query_parts.append(f"genotype {genotype}[Title]")
        if gene:
            gene_map = {"S": "surface", "C": "core", "P": "polymerase", "X": "X protein"}
            gene_name = gene_map.get(gene.upper(), gene)
            query_parts.append(f"{gene_name}[Title]")

        query = " AND ".join(query_parts)
        return self.search("nucleotide", query, max_results)

    def search_hcv_sequences(
        self,
        genotype: Optional[str] = None,
        region: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search for Hepatitis C virus sequences.

        Args:
            genotype: HCV genotype (1a, 1b, 2, 3, 4, 5, 6)
            region: Genomic region (NS3, NS5A, NS5B, E1, E2, core)
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = ["Hepatitis C virus[Organism]"]

        if genotype:
            query_parts.append(f"genotype {genotype}[Title]")
        if region:
            query_parts.append(f"{region}[Title]")

        query = " AND ".join(query_parts)
        return self.search("nucleotide", query, max_results)

    def search_fiv_sequences(
        self,
        subtype: Optional[str] = None,
        gene: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search for Feline Immunodeficiency Virus (FIV) sequences.

        Args:
            subtype: FIV subtype (A, B, C, D, E)
            gene: Gene name (gag, pol, env)
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = ["Feline immunodeficiency virus[Organism]"]

        if subtype:
            query_parts.append(f"subtype {subtype}[Title]")
        if gene:
            query_parts.append(f"{gene}[Gene Name]")

        query = " AND ".join(query_parts)
        return self.search("nucleotide", query, max_results)

    def search_treponema_sequences(
        self,
        subspecies: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search for Treponema pallidum (syphilis) sequences.

        Args:
            subspecies: Subspecies (pallidum, pertenue, endemicum)
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        if subspecies:
            query = f"Treponema pallidum subsp. {subspecies}[Organism]"
        else:
            query = "Treponema pallidum[Organism]"

        return self.search("nucleotide", query, max_results)

    # =========================================================================
    # ClinVar - Clinical Variants
    # =========================================================================

    def search_clinvar(
        self,
        gene: Optional[str] = None,
        condition: Optional[str] = None,
        significance: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search ClinVar for clinical variants.

        Args:
            gene: Gene symbol (e.g., BRCA1, TP53)
            condition: Disease/condition name
            significance: Clinical significance (pathogenic, benign, uncertain)
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = []

        if gene:
            query_parts.append(f"{gene}[Gene Name]")
        if condition:
            query_parts.append(f"{condition}[Disease/Phenotype]")
        if significance:
            query_parts.append(f"{significance}[Clinical Significance]")

        query = " AND ".join(query_parts) if query_parts else "human[Organism]"
        return self.search("clinvar", query, max_results)

    def get_clinvar_details(self, variant_ids: list[str]) -> pd.DataFrame:
        """
        Get detailed ClinVar variant information.

        Args:
            variant_ids: List of ClinVar variant IDs

        Returns:
            DataFrame with variant details
        """
        self._rate_limit()

        with Entrez.esummary(db="clinvar", id=",".join(variant_ids)) as handle:
            result = Entrez.read(handle)

        data = []
        doc_set = result.get("DocumentSummarySet", {})
        for item in doc_set.get("DocumentSummary", []):
            var_set = item.get("variation_set", [{}])
            data.append({
                "uid": item.get("uid"),
                "title": item.get("title"),
                "gene": item.get("gene_sort"),
                "clinical_significance": item.get("clinical_significance", {}).get("description") if isinstance(item.get("clinical_significance"), dict) else None,
                "variation_type": var_set[0].get("variation_type") if var_set else None,
                "conditions": "; ".join(
                    str(t.get("trait_name", "")) for t in item.get("trait_set", [])
                ),
            })

        return pd.DataFrame(data)

    # =========================================================================
    # dbSNP - SNP Database
    # =========================================================================

    def search_snp(
        self,
        gene: Optional[str] = None,
        chromosome: Optional[str] = None,
        clinical: bool = False,
        max_results: int = 500,
    ) -> dict:
        """
        Search dbSNP for SNPs.

        Args:
            gene: Gene symbol
            chromosome: Chromosome number
            clinical: Only return clinically significant SNPs
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = ["human[Organism]"]

        if gene:
            query_parts.append(f"{gene}[Gene Name]")
        if chromosome:
            query_parts.append(f"{chromosome}[Chromosome]")
        if clinical:
            query_parts.append("clinical[Filter]")

        query = " AND ".join(query_parts)
        return self.search("snp", query, max_results)

    # =========================================================================
    # Protein Database
    # =========================================================================

    def search_proteins(
        self,
        organism: Optional[str] = None,
        gene: Optional[str] = None,
        keyword: Optional[str] = None,
        max_results: int = 500,
    ) -> dict:
        """
        Search NCBI Protein database.

        Args:
            organism: Organism name
            gene: Gene name
            keyword: Keyword search
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = []

        if organism:
            query_parts.append(f"{organism}[Organism]")
        if gene:
            query_parts.append(f"{gene}[Gene Name]")
        if keyword:
            query_parts.append(keyword)

        query = " AND ".join(query_parts) if query_parts else "all[Filter]"
        return self.search("protein", query, max_results)

    def fetch_protein_sequences(self, ids: list[str]) -> Iterator[SeqRecord]:
        """
        Fetch protein sequences by ID.

        Args:
            ids: List of protein accession numbers or GIs

        Yields:
            BioPython SeqRecord objects
        """
        return self.fetch_sequences(ids, database="protein", format="fasta")

    def get_protein_summary(self, ids: list[str]) -> pd.DataFrame:
        """
        Get summary information for proteins.

        Args:
            ids: List of protein IDs

        Returns:
            DataFrame with protein metadata
        """
        return self.get_sequence_summary(ids, database="protein")

    # =========================================================================
    # Gene Database
    # =========================================================================

    def search_genes(
        self,
        symbol: Optional[str] = None,
        organism: str = "human",
        keyword: Optional[str] = None,
        max_results: int = 100,
    ) -> dict:
        """
        Search NCBI Gene database.

        Args:
            symbol: Gene symbol
            organism: Organism name
            keyword: Keyword search
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = [f"{organism}[Organism]"]

        if symbol:
            query_parts.append(f"{symbol}[Gene Name]")
        if keyword:
            query_parts.append(keyword)

        query = " AND ".join(query_parts)
        return self.search("gene", query, max_results)

    def get_gene_details(self, gene_ids: list[str]) -> pd.DataFrame:
        """
        Get detailed gene information.

        Args:
            gene_ids: List of NCBI Gene IDs

        Returns:
            DataFrame with gene details
        """
        self._rate_limit()

        with Entrez.esummary(db="gene", id=",".join(gene_ids)) as handle:
            result = Entrez.read(handle)

        data = []
        doc_set = result.get("DocumentSummarySet", {})
        for item in doc_set.get("DocumentSummary", []):
            org = item.get("Organism", {})
            data.append({
                "gene_id": item.get("uid"),
                "symbol": item.get("Name"),
                "description": item.get("Description"),
                "chromosome": item.get("Chromosome"),
                "map_location": item.get("MapLocation"),
                "organism": org.get("ScientificName") if isinstance(org, dict) else None,
                "aliases": item.get("OtherAliases"),
            })

        return pd.DataFrame(data)

    # =========================================================================
    # Structure Database (PDB via NCBI)
    # =========================================================================

    def search_structures(
        self,
        protein: Optional[str] = None,
        organism: Optional[str] = None,
        method: Optional[str] = None,
        max_results: int = 100,
    ) -> dict:
        """
        Search NCBI Structure database (MMDB/PDB).

        Args:
            protein: Protein name or keyword
            organism: Organism name
            method: Experimental method (X-ray, NMR, Cryo-EM)
            max_results: Maximum results

        Returns:
            Search results dictionary
        """
        query_parts = []

        if protein:
            query_parts.append(protein)
        if organism:
            query_parts.append(f"{organism}[Organism]")
        if method:
            query_parts.append(f"{method}[Method]")

        query = " AND ".join(query_parts) if query_parts else "all[Filter]"
        return self.search("structure", query, max_results)

    # =========================================================================
    # Taxonomy Database
    # =========================================================================

    def get_taxonomy(self, organism: str) -> dict:
        """
        Get taxonomy information for an organism.

        Args:
            organism: Organism name

        Returns:
            Dictionary with taxonomy information
        """
        # Search for organism
        results = self.search("taxonomy", f"{organism}[Scientific Name]", max_results=1)

        if not results["ids"]:
            return {}

        # Fetch taxonomy details
        self._rate_limit()
        with Entrez.efetch(db="taxonomy", id=results["ids"][0], retmode="xml") as handle:
            records = Entrez.read(handle)

        if not records:
            return {}

        taxon = records[0]
        other_names = taxon.get("OtherNames", {})
        common_names = other_names.get("CommonName", []) if isinstance(other_names, dict) else []
        genetic_code = taxon.get("GeneticCode", {})

        return {
            "taxon_id": taxon.get("TaxId"),
            "scientific_name": taxon.get("ScientificName"),
            "common_name": common_names[0] if common_names else None,
            "lineage": taxon.get("Lineage"),
            "division": taxon.get("Division"),
            "genetic_code": genetic_code.get("GCName") if isinstance(genetic_code, dict) else None,
        }
