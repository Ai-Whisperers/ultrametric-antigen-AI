"""
cBioPortal API client for cancer genomics data.

Provides access to:
- Cancer studies and clinical data
- Mutation profiles
- Copy number alterations
- Gene expression data

Uses the cBioPortal REST API: https://www.cbioportal.org/api
"""

from typing import Optional

import pandas as pd
import requests

from ..config import settings


class CBioPortalClient:
    """Client for cBioPortal REST API."""

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize cBioPortal client.

        Args:
            url: API base URL (defaults to public cBioPortal)
            token: Optional authentication token for institutional instances
        """
        self.base_url = (url or settings.cbioportal.url).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        if token or settings.cbioportal.token:
            self.session.headers.update(
                {"Authorization": f"Bearer {token or settings.cbioportal.token}"}
            )

    def _get(self, endpoint: str, params: Optional[dict] = None) -> list | dict:
        """Make GET request to API."""
        response = self.session.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            timeout=settings.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: dict) -> list | dict:
        """Make POST request to API."""
        response = self.session.post(
            f"{self.base_url}/{endpoint}",
            json=data,
            timeout=settings.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_cancer_types(self) -> pd.DataFrame:
        """
        Get all cancer types.

        Returns:
            DataFrame with cancer type information
        """
        data = self._get("cancer-types")
        return pd.DataFrame(data)

    def get_studies(self, keyword: Optional[str] = None) -> pd.DataFrame:
        """
        Get available cancer studies.

        Args:
            keyword: Optional keyword to filter studies

        Returns:
            DataFrame with study information
        """
        data = self._get("studies")
        df = pd.DataFrame(data)

        if keyword and not df.empty:
            mask = (
                df["name"].str.contains(keyword, case=False, na=False)
                | df["description"].str.contains(keyword, case=False, na=False)
            )
            df = df[mask]

        return df

    def get_study(self, study_id: str) -> dict:
        """
        Get details of a specific study.

        Args:
            study_id: Study identifier

        Returns:
            Study details
        """
        return self._get(f"studies/{study_id}")

    def get_patients(self, study_id: str) -> pd.DataFrame:
        """
        Get patients in a study.

        Args:
            study_id: Study identifier

        Returns:
            DataFrame with patient information
        """
        data = self._get(f"studies/{study_id}/patients")
        return pd.DataFrame(data)

    def get_samples(self, study_id: str) -> pd.DataFrame:
        """
        Get samples in a study.

        Args:
            study_id: Study identifier

        Returns:
            DataFrame with sample information
        """
        data = self._get(f"studies/{study_id}/samples")
        return pd.DataFrame(data)

    def get_clinical_data(self, study_id: str, attribute_ids: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Get clinical data for a study.

        Args:
            study_id: Study identifier
            attribute_ids: Specific clinical attributes to fetch

        Returns:
            DataFrame with clinical data
        """
        data = self._get(f"studies/{study_id}/clinical-data", {"clinicalDataType": "PATIENT"})
        df = pd.DataFrame(data)

        if attribute_ids and not df.empty:
            df = df[df["clinicalAttributeId"].isin(attribute_ids)]

        return df

    def get_molecular_profiles(self, study_id: str) -> pd.DataFrame:
        """
        Get molecular profiles available for a study.

        Args:
            study_id: Study identifier

        Returns:
            DataFrame with molecular profile information
        """
        data = self._get(f"studies/{study_id}/molecular-profiles")
        return pd.DataFrame(data)

    def get_mutations(
        self,
        study_id: str,
        gene_ids: Optional[list[int]] = None,
        sample_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get mutations for a study.

        Args:
            study_id: Study identifier
            gene_ids: Entrez gene IDs to filter
            sample_ids: Sample IDs to filter

        Returns:
            DataFrame with mutation data
        """
        # Get mutation profile
        profiles = self.get_molecular_profiles(study_id)
        mutation_profiles = profiles[profiles["molecularAlterationType"] == "MUTATION_EXTENDED"]

        if mutation_profiles.empty:
            return pd.DataFrame()

        profile_id = mutation_profiles.iloc[0]["molecularProfileId"]

        # Build query
        query = {"sampleListId": f"{study_id}_all"}

        if gene_ids:
            query["entrezGeneIds"] = gene_ids

        data = self._post(f"molecular-profiles/{profile_id}/mutations/fetch", query)
        return pd.DataFrame(data)

    def get_mutations_by_gene(self, gene_symbol: str, study_ids: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Get all mutations for a specific gene across studies.

        Args:
            gene_symbol: Hugo gene symbol (e.g., TP53, BRCA1)
            study_ids: List of study IDs to query (queries all if None)

        Returns:
            DataFrame with mutations
        """
        # Get gene info
        genes = self._post("genes/fetch", {"geneIds": [gene_symbol]})
        if not genes:
            return pd.DataFrame()

        gene_id = genes[0]["entrezGeneId"]

        # Get mutations across studies
        if study_ids is None:
            studies = self.get_studies()
            study_ids = studies["studyId"].tolist()[:20]  # Limit to avoid timeout

        all_mutations = []
        for study_id in study_ids:
            try:
                mutations = self.get_mutations(study_id, gene_ids=[gene_id])
                if not mutations.empty:
                    mutations["studyId"] = study_id
                    all_mutations.append(mutations)
            except requests.exceptions.HTTPError:
                continue  # Some studies may not have mutation data

        if all_mutations:
            return pd.concat(all_mutations, ignore_index=True)
        return pd.DataFrame()

    def get_copy_number_alterations(self, study_id: str, gene_ids: Optional[list[int]] = None) -> pd.DataFrame:
        """
        Get copy number alterations for a study.

        Args:
            study_id: Study identifier
            gene_ids: Entrez gene IDs to filter

        Returns:
            DataFrame with CNA data
        """
        profiles = self.get_molecular_profiles(study_id)
        cna_profiles = profiles[profiles["molecularAlterationType"] == "COPY_NUMBER_ALTERATION"]

        if cna_profiles.empty:
            return pd.DataFrame()

        profile_id = cna_profiles.iloc[0]["molecularProfileId"]

        query = {"sampleListId": f"{study_id}_all"}
        if gene_ids:
            query["entrezGeneIds"] = gene_ids

        data = self._post(f"molecular-profiles/{profile_id}/discrete-copy-number/fetch", query)
        return pd.DataFrame(data)

    def get_gene_expression(
        self,
        study_id: str,
        gene_ids: list[int],
        sample_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get gene expression data for a study.

        Args:
            study_id: Study identifier
            gene_ids: Entrez gene IDs
            sample_ids: Sample IDs to filter

        Returns:
            DataFrame with expression data
        """
        profiles = self.get_molecular_profiles(study_id)
        expr_profiles = profiles[profiles["molecularAlterationType"] == "MRNA_EXPRESSION"]

        if expr_profiles.empty:
            return pd.DataFrame()

        profile_id = expr_profiles.iloc[0]["molecularProfileId"]

        query = {
            "entrezGeneIds": gene_ids,
            "sampleListId": f"{study_id}_all" if not sample_ids else None,
        }
        if sample_ids:
            query["sampleIds"] = sample_ids

        data = self._post(f"molecular-profiles/{profile_id}/molecular-data/fetch", query)
        return pd.DataFrame(data)

    def search_genes(self, keyword: str) -> pd.DataFrame:
        """
        Search for genes by keyword.

        Args:
            keyword: Gene name or keyword

        Returns:
            DataFrame with gene information
        """
        data = self._get("genes", {"keyword": keyword})
        return pd.DataFrame(data)

    def get_mutation_counts(self, study_id: str) -> pd.DataFrame:
        """
        Get mutation counts per sample for a study.

        Args:
            study_id: Study identifier

        Returns:
            DataFrame with mutation counts
        """
        mutations = self.get_mutations(study_id)
        if mutations.empty:
            return pd.DataFrame()

        counts = mutations.groupby("sampleId").size().reset_index(name="mutation_count")
        return counts

    def get_top_mutated_genes(self, study_id: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get most frequently mutated genes in a study.

        Args:
            study_id: Study identifier
            top_n: Number of top genes to return

        Returns:
            DataFrame with gene mutation frequencies
        """
        mutations = self.get_mutations(study_id)
        if mutations.empty:
            return pd.DataFrame()

        gene_counts = (
            mutations.groupby(["entrezGeneId", "gene"])
            .agg(
                mutation_count=("sampleId", "count"),
                unique_samples=("sampleId", "nunique"),
            )
            .reset_index()
        )

        total_samples = mutations["sampleId"].nunique()
        gene_counts["frequency"] = gene_counts["unique_samples"] / total_samples * 100

        return gene_counts.nlargest(top_n, "frequency")

    # ========== STRUCTURAL VARIANTS (Gene Fusions) ==========

    def get_structural_variants(
        self,
        study_id: str,
        gene_ids: Optional[list[int]] = None,
        sample_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get structural variants (gene fusions) for a study.

        Args:
            study_id: Study identifier
            gene_ids: Entrez gene IDs to filter
            sample_ids: Sample IDs to filter

        Returns:
            DataFrame with structural variant data
        """
        profiles = self.get_molecular_profiles(study_id)
        sv_profiles = profiles[profiles["molecularAlterationType"] == "STRUCTURAL_VARIANT"]

        if sv_profiles.empty:
            return pd.DataFrame()

        profile_id = sv_profiles.iloc[0]["molecularProfileId"]

        query = {}
        if sample_ids:
            query["sampleIds"] = [{"sampleId": s, "studyId": study_id} for s in sample_ids]
        else:
            query["sampleListId"] = f"{study_id}_all"

        if gene_ids:
            query["entrezGeneIds"] = gene_ids

        try:
            data = self._post(f"molecular-profiles/{profile_id}/structural-variant/fetch", query)
            return pd.DataFrame(data)
        except requests.exceptions.HTTPError:
            return pd.DataFrame()

    def get_gene_fusions(
        self,
        study_id: str,
        gene_symbols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get gene fusions for a study.

        Args:
            study_id: Study identifier
            gene_symbols: Gene symbols to filter (e.g., ["ALK", "ROS1"])

        Returns:
            DataFrame with fusion data including partner genes
        """
        gene_ids = None
        if gene_symbols:
            genes = self._post("genes/fetch", {"geneIds": gene_symbols})
            gene_ids = [g["entrezGeneId"] for g in genes] if genes else None

        sv_data = self.get_structural_variants(study_id, gene_ids=gene_ids)

        if sv_data.empty:
            return pd.DataFrame()

        # Filter for fusions specifically
        if "eventType" in sv_data.columns:
            sv_data = sv_data[sv_data["eventType"].str.contains("FUSION", case=False, na=False)]

        return sv_data

    def get_fusion_genes_summary(self, study_id: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get summary of most common fusion genes in a study.

        Args:
            study_id: Study identifier
            top_n: Number of top fusion genes to return

        Returns:
            DataFrame with fusion gene frequencies
        """
        fusions = self.get_structural_variants(study_id)

        if fusions.empty:
            return pd.DataFrame()

        # Count gene involvement in fusions
        gene_counts = []
        for col in ["site1HugoSymbol", "site2HugoSymbol"]:
            if col in fusions.columns:
                counts = fusions[col].value_counts()
                gene_counts.append(counts)

        if not gene_counts:
            return pd.DataFrame()

        combined = pd.concat(gene_counts).groupby(level=0).sum()
        result = combined.reset_index()
        result.columns = ["gene", "fusion_count"]

        return result.nlargest(top_n, "fusion_count")

    # ========== METHYLATION DATA ==========

    def get_methylation_data(
        self,
        study_id: str,
        gene_ids: list[int],
        sample_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get DNA methylation data for a study.

        Args:
            study_id: Study identifier
            gene_ids: Entrez gene IDs
            sample_ids: Sample IDs to filter

        Returns:
            DataFrame with methylation beta values
        """
        profiles = self.get_molecular_profiles(study_id)
        meth_profiles = profiles[profiles["molecularAlterationType"] == "METHYLATION"]

        if meth_profiles.empty:
            # Try alternate naming
            meth_profiles = profiles[profiles["molecularAlterationType"].str.contains("METHYLATION", case=False, na=False)]

        if meth_profiles.empty:
            return pd.DataFrame()

        profile_id = meth_profiles.iloc[0]["molecularProfileId"]

        query = {"entrezGeneIds": gene_ids}
        if sample_ids:
            query["sampleIds"] = [{"sampleId": s, "studyId": study_id} for s in sample_ids]
        else:
            query["sampleListId"] = f"{study_id}_all"

        try:
            data = self._post(f"molecular-profiles/{profile_id}/molecular-data/fetch", query)
            return pd.DataFrame(data)
        except requests.exceptions.HTTPError:
            return pd.DataFrame()

    def get_methylation_by_gene(
        self,
        gene_symbol: str,
        study_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get methylation data for a specific gene across studies.

        Args:
            gene_symbol: Hugo gene symbol (e.g., MGMT, MLH1)
            study_ids: List of study IDs (queries studies with methylation if None)

        Returns:
            DataFrame with methylation data across studies
        """
        genes = self._post("genes/fetch", {"geneIds": [gene_symbol]})
        if not genes:
            return pd.DataFrame()

        gene_id = genes[0]["entrezGeneId"]

        if study_ids is None:
            # Find studies with methylation data
            all_studies = self.get_studies()
            study_ids = []
            for sid in all_studies["studyId"].tolist()[:30]:
                try:
                    profiles = self.get_molecular_profiles(sid)
                    if any(profiles["molecularAlterationType"].str.contains("METHYLATION", case=False, na=False)):
                        study_ids.append(sid)
                    if len(study_ids) >= 10:
                        break
                except Exception:
                    continue

        all_data = []
        for study_id in study_ids:
            try:
                meth = self.get_methylation_data(study_id, gene_ids=[gene_id])
                if not meth.empty:
                    meth["studyId"] = study_id
                    all_data.append(meth)
            except Exception:
                continue

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    # ========== PROTEIN LEVELS (RPPA) ==========

    def get_protein_data(
        self,
        study_id: str,
        gene_ids: Optional[list[int]] = None,
        sample_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get protein expression data (RPPA) for a study.

        Args:
            study_id: Study identifier
            gene_ids: Entrez gene IDs to filter
            sample_ids: Sample IDs to filter

        Returns:
            DataFrame with protein expression levels
        """
        profiles = self.get_molecular_profiles(study_id)
        protein_profiles = profiles[profiles["molecularAlterationType"] == "PROTEIN_LEVEL"]

        if protein_profiles.empty:
            return pd.DataFrame()

        profile_id = protein_profiles.iloc[0]["molecularProfileId"]

        query = {}
        if gene_ids:
            query["entrezGeneIds"] = gene_ids
        if sample_ids:
            query["sampleIds"] = [{"sampleId": s, "studyId": study_id} for s in sample_ids]
        else:
            query["sampleListId"] = f"{study_id}_all"

        try:
            data = self._post(f"molecular-profiles/{profile_id}/molecular-data/fetch", query)
            return pd.DataFrame(data)
        except requests.exceptions.HTTPError:
            return pd.DataFrame()

    def get_protein_by_gene(
        self,
        gene_symbol: str,
        study_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get protein expression data for a gene across studies.

        Args:
            gene_symbol: Hugo gene symbol
            study_ids: Study IDs to query (finds RPPA studies if None)

        Returns:
            DataFrame with protein expression across studies
        """
        genes = self._post("genes/fetch", {"geneIds": [gene_symbol]})
        if not genes:
            return pd.DataFrame()

        gene_id = genes[0]["entrezGeneId"]

        if study_ids is None:
            # Find studies with protein data
            all_studies = self.get_studies()
            study_ids = []
            for sid in all_studies["studyId"].tolist()[:30]:
                try:
                    profiles = self.get_molecular_profiles(sid)
                    if any(profiles["molecularAlterationType"] == "PROTEIN_LEVEL"):
                        study_ids.append(sid)
                    if len(study_ids) >= 10:
                        break
                except Exception:
                    continue

        all_data = []
        for study_id in study_ids:
            try:
                protein = self.get_protein_data(study_id, gene_ids=[gene_id])
                if not protein.empty:
                    protein["studyId"] = study_id
                    all_data.append(protein)
            except Exception:
                continue

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    # ========== CLINICAL ATTRIBUTES ==========

    def get_clinical_attributes(self, study_id: str) -> pd.DataFrame:
        """
        Get all clinical attributes available for a study.

        Args:
            study_id: Study identifier

        Returns:
            DataFrame with clinical attribute definitions
        """
        data = self._get(f"studies/{study_id}/clinical-attributes")
        return pd.DataFrame(data)

    def get_sample_clinical_data(
        self,
        study_id: str,
        attribute_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Get sample-level clinical data (tumor characteristics, etc).

        Args:
            study_id: Study identifier
            attribute_ids: Specific clinical attributes to fetch

        Returns:
            DataFrame with sample clinical data
        """
        data = self._get(f"studies/{study_id}/clinical-data", {"clinicalDataType": "SAMPLE"})
        df = pd.DataFrame(data)

        if attribute_ids and not df.empty:
            df = df[df["clinicalAttributeId"].isin(attribute_ids)]

        return df

    # ========== TREATMENT DATA ==========

    def get_treatments(self, study_id: str) -> pd.DataFrame:
        """
        Get treatment data for a study.

        Args:
            study_id: Study identifier

        Returns:
            DataFrame with treatment information
        """
        try:
            data = self._get(f"studies/{study_id}/treatments")
            return pd.DataFrame(data)
        except requests.exceptions.HTTPError:
            return pd.DataFrame()

    # ========== SURVIVAL DATA ==========

    def get_survival_data(self, study_id: str) -> pd.DataFrame:
        """
        Get survival data for patients in a study.

        Args:
            study_id: Study identifier

        Returns:
            DataFrame with survival times and status
        """
        clinical = self.get_clinical_data(study_id)

        if clinical.empty:
            return pd.DataFrame()

        # Common survival attribute IDs
        survival_attrs = [
            "OS_STATUS", "OS_MONTHS",  # Overall survival
            "DFS_STATUS", "DFS_MONTHS",  # Disease-free survival
            "PFS_STATUS", "PFS_MONTHS",  # Progression-free survival
            "DSS_STATUS", "DSS_MONTHS",  # Disease-specific survival
        ]

        survival_data = clinical[clinical["clinicalAttributeId"].isin(survival_attrs)]

        if survival_data.empty:
            return pd.DataFrame()

        # Pivot to wide format
        pivot = survival_data.pivot_table(
            index="patientId",
            columns="clinicalAttributeId",
            values="value",
            aggfunc="first",
        ).reset_index()

        return pivot

    # ========== GENOMIC SIGNATURE SCORES ==========

    def get_generic_assay_data(
        self,
        study_id: str,
        profile_type: str = "GENERIC_ASSAY",
    ) -> pd.DataFrame:
        """
        Get generic assay data (signatures, scores, etc).

        Args:
            study_id: Study identifier
            profile_type: Type of generic assay profile

        Returns:
            DataFrame with assay data
        """
        profiles = self.get_molecular_profiles(study_id)
        assay_profiles = profiles[profiles["molecularAlterationType"] == profile_type]

        if assay_profiles.empty:
            return pd.DataFrame()

        profile_id = assay_profiles.iloc[0]["molecularProfileId"]

        try:
            data = self._post(
                f"molecular-profiles/{profile_id}/molecular-data/fetch",
                {"sampleListId": f"{study_id}_all"},
            )
            return pd.DataFrame(data)
        except requests.exceptions.HTTPError:
            return pd.DataFrame()

    # ========== AVAILABLE PROFILE TYPES ==========

    def get_available_data_types(self, study_id: str) -> dict:
        """
        Get summary of available data types for a study.

        Args:
            study_id: Study identifier

        Returns:
            Dictionary with available data types and their counts
        """
        profiles = self.get_molecular_profiles(study_id)

        if profiles.empty:
            return {}

        summary = profiles.groupby("molecularAlterationType").size().to_dict()

        return {
            "mutation": summary.get("MUTATION_EXTENDED", 0) > 0,
            "copy_number": summary.get("COPY_NUMBER_ALTERATION", 0) > 0,
            "expression": summary.get("MRNA_EXPRESSION", 0) > 0,
            "structural_variant": summary.get("STRUCTURAL_VARIANT", 0) > 0,
            "methylation": any("METHYLATION" in k for k in summary.keys()),
            "protein": summary.get("PROTEIN_LEVEL", 0) > 0,
            "generic_assay": summary.get("GENERIC_ASSAY", 0) > 0,
            "profile_counts": summary,
        }

    def find_studies_with_data_type(
        self,
        data_type: str,
        keyword: Optional[str] = None,
        max_studies: int = 50,
    ) -> pd.DataFrame:
        """
        Find studies that have a specific data type.

        Args:
            data_type: One of: mutation, copy_number, expression, structural_variant,
                       methylation, protein, generic_assay
            keyword: Optional keyword to filter study names
            max_studies: Maximum number of studies to check

        Returns:
            DataFrame with matching studies
        """
        all_studies = self.get_studies(keyword=keyword)

        if all_studies.empty:
            return pd.DataFrame()

        matching = []
        for _, study in all_studies.head(max_studies).iterrows():
            study_id = study["studyId"]
            try:
                available = self.get_available_data_types(study_id)
                if available.get(data_type, False):
                    matching.append(study.to_dict())
            except Exception:
                continue

        return pd.DataFrame(matching)
