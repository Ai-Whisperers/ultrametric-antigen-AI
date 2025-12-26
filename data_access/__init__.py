"""
Unified Data Access Module for Bioinformatics Research.

This module provides API-based access to multiple biological databases
without requiring bulk data downloads.

Supported Databases:
    - NCBI/Entrez: GenBank, PubMed, ClinVar, dbSNP, proteins, genes, structures
    - Stanford HIVDB: HIV drug resistance analysis (Sierra GraphQL)
    - cBioPortal: Cancer genomics, mutations, fusions, methylation, protein levels
    - MalariaGEN: Malaria genomics (cloud-based, zero download)
    - CARD: Antibiotic resistance ontology and genes
    - BV-BRC: Bacterial/viral genomes, AMR, epitopes, protein structures, GO
    - UniProt: Protein sequences, functions, disease associations
    - IEDB: Immune epitopes, MHC binding predictions
    - LANL HIV: Curated HIV sequences, resistance mutations, immunology

Quick Start:
    ```python
    from data_access import DataHub

    # Initialize the data hub
    hub = DataHub()

    # Check configuration
    warnings = hub.validate()
    for w in warnings:
        print(f"Warning: {w}")

    # Access individual clients
    hiv_data = hub.hivdb.get_drug_classes()
    protein_data = hub.uniprot.search_proteins("kinase", organism="human")
    epitopes = hub.iedb.get_hiv_epitopes()
    ```

Configuration:
    Copy config/.env.template to config/.env and fill in your credentials.
    Most APIs are public and don't require authentication.
    NCBI requires an email address; an API key is optional but recommended.
"""

from typing import Optional

import pandas as pd

from .config import settings, Settings
from .clients import (
    NCBIClient,
    HIVDBClient,
    CBioPortalClient,
    MalariaGENClient,
    CARDClient,
    BVBRCClient,
    UniProtClient,
    IEDBClient,
    LANLHIVClient,
)


class DataHub:
    """
    Unified interface to all biological database APIs.

    Provides lazy-loaded access to individual database clients
    and cross-database query capabilities.

    Clients:
        - ncbi: NCBI/Entrez (GenBank, PubMed, ClinVar, dbSNP, proteins, genes)
        - hivdb: Stanford HIVDB (drug resistance analysis)
        - cbioportal: cBioPortal (cancer genomics, mutations, fusions)
        - malariagen: MalariaGEN (malaria genomics)
        - card: CARD (antibiotic resistance)
        - bvbrc: BV-BRC (bacterial/viral genomes, AMR, epitopes)
        - uniprot: UniProt (protein sequences, functions)
        - iedb: IEDB (immune epitopes, MHC binding)
        - lanl: LANL HIV Database (curated HIV data)
    """

    def __init__(self):
        """Initialize the DataHub with lazy-loaded clients."""
        self._ncbi: Optional[NCBIClient] = None
        self._hivdb: Optional[HIVDBClient] = None
        self._cbioportal: Optional[CBioPortalClient] = None
        self._malariagen: Optional[MalariaGENClient] = None
        self._card: Optional[CARDClient] = None
        self._bvbrc: Optional[BVBRCClient] = None
        self._uniprot: Optional[UniProtClient] = None
        self._iedb: Optional[IEDBClient] = None
        self._lanl: Optional[LANLHIVClient] = None

    @property
    def ncbi(self) -> NCBIClient:
        """Get NCBI/Entrez client (lazy-loaded)."""
        if self._ncbi is None:
            self._ncbi = NCBIClient()
        return self._ncbi

    @property
    def hivdb(self) -> HIVDBClient:
        """Get Stanford HIVDB client (lazy-loaded)."""
        if self._hivdb is None:
            self._hivdb = HIVDBClient()
        return self._hivdb

    @property
    def cbioportal(self) -> CBioPortalClient:
        """Get cBioPortal client (lazy-loaded)."""
        if self._cbioportal is None:
            self._cbioportal = CBioPortalClient()
        return self._cbioportal

    @property
    def malariagen(self) -> MalariaGENClient:
        """Get MalariaGEN client (lazy-loaded)."""
        if self._malariagen is None:
            self._malariagen = MalariaGENClient()
        return self._malariagen

    @property
    def card(self) -> CARDClient:
        """Get CARD client (lazy-loaded)."""
        if self._card is None:
            self._card = CARDClient()
        return self._card

    @property
    def bvbrc(self) -> BVBRCClient:
        """Get BV-BRC client (lazy-loaded)."""
        if self._bvbrc is None:
            self._bvbrc = BVBRCClient()
        return self._bvbrc

    @property
    def uniprot(self) -> UniProtClient:
        """Get UniProt client (lazy-loaded)."""
        if self._uniprot is None:
            self._uniprot = UniProtClient()
        return self._uniprot

    @property
    def iedb(self) -> IEDBClient:
        """Get IEDB client (lazy-loaded)."""
        if self._iedb is None:
            self._iedb = IEDBClient()
        return self._iedb

    @property
    def lanl(self) -> LANLHIVClient:
        """Get LANL HIV Database client (lazy-loaded)."""
        if self._lanl is None:
            self._lanl = LANLHIVClient()
        return self._lanl

    def validate(self) -> list[str]:
        """
        Validate configuration and return any warnings.

        Returns:
            List of configuration warnings
        """
        return settings.validate()

    def test_connections(self) -> pd.DataFrame:
        """
        Test connections to all APIs.

        Returns:
            DataFrame with connection status for each API
        """
        results = []

        # Test HIVDB (public, no auth)
        try:
            self.hivdb.get_algorithms()
            results.append({"api": "HIVDB", "status": "OK", "message": "Connected"})
        except Exception as e:
            results.append({"api": "HIVDB", "status": "ERROR", "message": str(e)})

        # Test cBioPortal (public)
        try:
            self.cbioportal.get_cancer_types()
            results.append({"api": "cBioPortal", "status": "OK", "message": "Connected"})
        except Exception as e:
            results.append({"api": "cBioPortal", "status": "ERROR", "message": str(e)})

        # Test CARD (public)
        try:
            self.card.get_drug_classes()
            results.append({"api": "CARD", "status": "OK", "message": "Connected"})
        except Exception as e:
            results.append({"api": "CARD", "status": "ERROR", "message": str(e)})

        # Test BV-BRC (public)
        try:
            self.bvbrc.get_data_summary()
            results.append({"api": "BV-BRC", "status": "OK", "message": "Connected"})
        except Exception as e:
            results.append({"api": "BV-BRC", "status": "ERROR", "message": str(e)})

        # Test NCBI (requires email)
        try:
            if settings.ncbi.email:
                self.ncbi.search("nucleotide", "HIV-1[Organism]", max_results=1)
                results.append({"api": "NCBI", "status": "OK", "message": "Connected"})
            else:
                results.append({"api": "NCBI", "status": "SKIP", "message": "Email not configured"})
        except Exception as e:
            results.append({"api": "NCBI", "status": "ERROR", "message": str(e)})

        # Test MalariaGEN (requires package)
        try:
            self.malariagen.get_dataset_summary()
            results.append({"api": "MalariaGEN", "status": "OK", "message": "Connected"})
        except ImportError:
            results.append({"api": "MalariaGEN", "status": "SKIP", "message": "Package not installed"})
        except Exception as e:
            results.append({"api": "MalariaGEN", "status": "ERROR", "message": str(e)})

        # Test UniProt (public)
        try:
            self.uniprot.search_proteins("test", limit=1)
            results.append({"api": "UniProt", "status": "OK", "message": "Connected"})
        except Exception as e:
            results.append({"api": "UniProt", "status": "ERROR", "message": str(e)})

        # Test IEDB (public)
        try:
            self.iedb.get_epitope_summary("HIV-1")
            results.append({"api": "IEDB", "status": "OK", "message": "Connected"})
        except Exception as e:
            results.append({"api": "IEDB", "status": "ERROR", "message": str(e)})

        # Test LANL HIV (curated data, no API call needed)
        try:
            self.lanl.get_resistance_mutations()
            results.append({"api": "LANL HIV", "status": "OK", "message": "Curated data available"})
        except Exception as e:
            results.append({"api": "LANL HIV", "status": "ERROR", "message": str(e)})

        return pd.DataFrame(results)

    # =========================================================================
    # Cross-database convenience methods
    # =========================================================================

    def search_hiv_resistance(self, sequence: str) -> dict:
        """
        Analyze HIV sequence for drug resistance.

        Args:
            sequence: HIV nucleotide or amino acid sequence

        Returns:
            Dictionary with resistance analysis results
        """
        return self.hivdb.analyze_sequence(sequence)

    def get_hiv_sequences(self, subtype: Optional[str] = None, max_results: int = 100) -> pd.DataFrame:
        """
        Get HIV sequences from NCBI.

        Args:
            subtype: HIV subtype filter
            max_results: Maximum sequences to return

        Returns:
            DataFrame with sequence metadata
        """
        results = self.ncbi.search_hiv_sequences(subtype=subtype, max_results=max_results)
        if results["ids"]:
            return self.ncbi.get_sequence_summary(results["ids"])
        return pd.DataFrame()

    def get_malaria_samples(self, country: Optional[str] = None) -> pd.DataFrame:
        """
        Get malaria sample metadata.

        Args:
            country: Filter by country

        Returns:
            DataFrame with sample metadata
        """
        if country:
            return self.malariagen.get_pf_samples_by_country(country)
        return self.malariagen.get_pf_sample_metadata()

    def get_amr_data(self, pathogen: Optional[str] = None) -> pd.DataFrame:
        """
        Get antimicrobial resistance data.

        Args:
            pathogen: Filter by pathogen name

        Returns:
            DataFrame with AMR phenotypes
        """
        return self.bvbrc.get_amr_phenotypes(limit=500)

    def get_cancer_mutations(self, gene: str, study_limit: int = 10) -> pd.DataFrame:
        """
        Get cancer mutations for a gene across studies.

        Args:
            gene: Gene symbol (e.g., TP53, BRCA1)
            study_limit: Maximum studies to query

        Returns:
            DataFrame with mutations
        """
        studies = self.cbioportal.get_studies()
        study_ids = studies["studyId"].tolist()[:study_limit]
        return self.cbioportal.get_mutations_by_gene(gene, study_ids)

    def get_tb_genomes(self, limit: int = 50) -> pd.DataFrame:
        """
        Get Mycobacterium tuberculosis genomes.

        Args:
            limit: Maximum genomes

        Returns:
            DataFrame with TB genome information
        """
        return self.bvbrc.get_tb_genomes(limit=limit)

    def get_syphilis_genomes(self, limit: int = 50) -> pd.DataFrame:
        """
        Get Treponema pallidum (syphilis) genomes.

        Args:
            limit: Maximum genomes

        Returns:
            DataFrame with syphilis pathogen information
        """
        return self.bvbrc.get_syphilis_genomes(limit=limit)

    def get_eskape_summary(self) -> pd.DataFrame:
        """
        Get summary of ESKAPE pathogens.

        Returns:
            DataFrame with ESKAPE pathogen information
        """
        return self.card.get_eskape_pathogens()

    # =========================================================================
    # New cross-database convenience methods
    # =========================================================================

    def get_hiv_epitopes(self, epitope_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get HIV epitopes from IEDB.

        Args:
            epitope_type: T-cell or B-cell

        Returns:
            DataFrame with HIV epitopes
        """
        return self.iedb.get_hiv_epitopes(epitope_type=epitope_type)

    def get_hiv_resistance_mutations(self, drug_class: Optional[str] = None) -> pd.DataFrame:
        """
        Get curated HIV drug resistance mutations from LANL.

        Args:
            drug_class: Filter by drug class (NRTI, NNRTI, PI, INSTI)

        Returns:
            DataFrame with resistance mutations
        """
        return self.lanl.get_resistance_mutations(drug_class=drug_class)

    def get_viral_proteins(self, virus: str, limit: int = 100) -> pd.DataFrame:
        """
        Get proteins for a virus from UniProt.

        Args:
            virus: Virus name (HIV-1, HBV, HCV, SARS-CoV-2)
            limit: Maximum results

        Returns:
            DataFrame with viral proteins
        """
        return self.uniprot.get_viral_proteins(virus, limit=limit)

    def get_disease_proteins(self, disease: str, limit: int = 100) -> pd.DataFrame:
        """
        Get proteins associated with a disease from UniProt.

        Args:
            disease: Disease name
            limit: Maximum results

        Returns:
            DataFrame with disease-associated proteins
        """
        return self.uniprot.get_disease_proteins(disease, limit=limit)

    def get_gene_fusions(self, study_id: str) -> pd.DataFrame:
        """
        Get gene fusions from a cBioPortal study.

        Args:
            study_id: cBioPortal study ID

        Returns:
            DataFrame with fusion data
        """
        return self.cbioportal.get_gene_fusions(study_id)

    def get_clinvar_variants(self, gene: str, limit: int = 100) -> pd.DataFrame:
        """
        Get clinical variants for a gene from ClinVar.

        Args:
            gene: Gene symbol
            limit: Maximum results

        Returns:
            DataFrame with clinical variants
        """
        return self.ncbi.search_clinvar(gene=gene, max_results=limit)

    def predict_mhc_binding(
        self,
        sequence: str,
        alleles: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Predict MHC binding for a sequence using IEDB.

        Args:
            sequence: Amino acid sequence
            alleles: HLA alleles (defaults to common alleles)

        Returns:
            DataFrame with binding predictions
        """
        return self.iedb.predict_mhc_binding(sequence, alleles=alleles)

    def get_pathogen_proteins(
        self,
        pathogen: str,
        virulence_only: bool = False,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get proteins for a pathogen from UniProt.

        Args:
            pathogen: Pathogen name (e.g., Mycobacterium tuberculosis)
            virulence_only: Only return virulence factors
            limit: Maximum results

        Returns:
            DataFrame with pathogen proteins
        """
        return self.uniprot.get_pathogen_proteins(
            pathogen, virulence_only=virulence_only, limit=limit
        )

    def get_bnab_targets(self) -> pd.DataFrame:
        """
        Get broadly neutralizing antibody target sites from LANL.

        Returns:
            DataFrame with bnAb target information
        """
        return self.lanl.get_bnab_targets()

    def get_survival_data(self, study_id: str) -> pd.DataFrame:
        """
        Get patient survival data from cBioPortal.

        Args:
            study_id: cBioPortal study ID

        Returns:
            DataFrame with survival data
        """
        return self.cbioportal.get_survival_data(study_id)

    def get_available_data_summary(self) -> dict:
        """
        Get summary of all available data sources.

        Returns:
            Dictionary with data source summaries
        """
        return {
            "ncbi": {
                "description": "NCBI/Entrez databases",
                "data_types": [
                    "nucleotide", "protein", "gene", "structure",
                    "clinvar", "snp", "pubmed", "taxonomy"
                ],
                "viruses": ["HIV", "HBV", "HCV", "FIV", "SARS-CoV-2"],
            },
            "hivdb": {
                "description": "Stanford HIVDB",
                "data_types": ["drug_resistance", "mutations", "subtypes"],
                "genes": ["PR", "RT", "IN"],
            },
            "cbioportal": {
                "description": "cBioPortal cancer genomics",
                "data_types": [
                    "mutations", "copy_number", "expression",
                    "structural_variants", "methylation", "protein_levels"
                ],
                "studies": "300+ cancer studies",
            },
            "malariagen": {
                "description": "MalariaGEN cloud data",
                "data_types": ["genotypes", "sample_metadata", "population_genetics"],
                "species": ["P. falciparum", "P. vivax", "Anopheles"],
            },
            "card": {
                "description": "CARD AMR database",
                "data_types": ["resistance_genes", "drug_classes", "mechanisms"],
            },
            "bvbrc": {
                "description": "BV-BRC bacterial/viral database",
                "data_types": [
                    "genomes", "amr_phenotypes", "specialty_genes",
                    "epitopes", "protein_structures", "gene_ontology"
                ],
                "pathogens": ["TB", "Syphilis", "ESKAPE"],
            },
            "uniprot": {
                "description": "UniProt protein database",
                "data_types": [
                    "sequences", "functions", "diseases",
                    "drug_targets", "ptm", "structures"
                ],
            },
            "iedb": {
                "description": "IEDB immune epitopes",
                "data_types": [
                    "t_cell_epitopes", "b_cell_epitopes",
                    "mhc_binding", "immunogenicity"
                ],
                "tools": ["MHC-I prediction", "MHC-II prediction", "population coverage"],
            },
            "lanl": {
                "description": "LANL HIV Database",
                "data_types": [
                    "curated_sequences", "resistance_mutations",
                    "ctl_epitopes", "bnab_targets", "subtypes"
                ],
            },
        }


# Convenience exports
__all__ = [
    "DataHub",
    "settings",
    "Settings",
    "NCBIClient",
    "HIVDBClient",
    "CBioPortalClient",
    "MalariaGENClient",
    "CARDClient",
    "BVBRCClient",
    "UniProtClient",
    "IEDBClient",
    "LANLHIVClient",
]
