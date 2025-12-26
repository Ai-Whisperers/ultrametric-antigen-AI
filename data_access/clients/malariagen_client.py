"""
MalariaGEN API client for malaria genomics data.

Provides cloud-based access (no download required) to:
- Plasmodium falciparum (Pf7) - 20,864 samples
- Plasmodium vivax (Pv4)
- Anopheles mosquito data (Ag3, Af1)

Uses the malariagen_data package which streams data from Google Cloud Storage.
"""

from typing import Optional

import pandas as pd

# Lazy import to avoid dependency errors if not installed
malariagen_data = None


def _ensure_malariagen():
    """Ensure malariagen_data is imported."""
    global malariagen_data
    if malariagen_data is None:
        try:
            import malariagen_data as md

            malariagen_data = md
        except ImportError:
            raise ImportError(
                "malariagen_data package not installed. "
                "Install with: pip install malariagen_data"
            )


class MalariaGENClient:
    """Client for MalariaGEN cloud-based data access."""

    def __init__(self):
        """Initialize MalariaGEN client."""
        _ensure_malariagen()
        self._pf7 = None
        self._pv4 = None
        self._ag3 = None

    @property
    def pf7(self):
        """Lazy load Pf7 (P. falciparum) data resource."""
        if self._pf7 is None:
            self._pf7 = malariagen_data.Pf7()
        return self._pf7

    @property
    def pv4(self):
        """Lazy load Pv4 (P. vivax) data resource."""
        if self._pv4 is None:
            self._pv4 = malariagen_data.Pv4()
        return self._pv4

    @property
    def ag3(self):
        """Lazy load Ag3 (Anopheles gambiae) data resource."""
        if self._ag3 is None:
            self._ag3 = malariagen_data.Ag3()
        return self._ag3

    # =========================================================================
    # Plasmodium falciparum (Pf7) Methods
    # =========================================================================

    def get_pf_sample_metadata(self) -> pd.DataFrame:
        """
        Get P. falciparum sample metadata.

        Returns:
            DataFrame with sample information (20,864 samples)
        """
        return self.pf7.sample_metadata()

    def get_pf_samples_by_country(self, country: str) -> pd.DataFrame:
        """
        Get P. falciparum samples from a specific country.

        Args:
            country: Country name

        Returns:
            Filtered DataFrame
        """
        df = self.pf7.sample_metadata()
        return df[df["country"] == country]

    def get_pf_samples_by_region(self, region: str) -> pd.DataFrame:
        """
        Get P. falciparum samples from a specific region.

        Args:
            region: Region name (e.g., "Africa", "Southeast Asia")

        Returns:
            Filtered DataFrame
        """
        df = self.pf7.sample_metadata()
        # Handle different column names
        if "admin1_name" in df.columns:
            return df[df["admin1_name"].str.contains(region, case=False, na=False)]
        return df

    def get_pf_drug_resistance_variants(
        self, gene: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get known drug resistance variants in P. falciparum.

        Args:
            gene: Filter by gene name (e.g., "kelch13", "pfcrt", "pfmdr1")

        Returns:
            DataFrame with resistance variants
        """
        # Pf7 provides drug resistance markers
        try:
            variants = self.pf7.drug_resistant_variants()
            if gene:
                variants = variants[variants["gene"].str.contains(gene, case=False, na=False)]
            return variants
        except AttributeError:
            # Fallback if method not available
            return pd.DataFrame()

    def get_pf_snp_calls(
        self,
        region: str,
        sample_sets: Optional[list[str]] = None,
    ):
        """
        Get SNP calls for a genomic region.

        Args:
            region: Genomic region (e.g., "Pf3D7_13_v3:1-100000")
            sample_sets: Sample set IDs to include

        Returns:
            SNP data (xarray Dataset or similar)
        """
        return self.pf7.snp_calls(region=region, sample_sets=sample_sets)

    def get_pf_sample_sets(self) -> pd.DataFrame:
        """
        Get available sample sets in Pf7.

        Returns:
            DataFrame with sample set information
        """
        return self.pf7.sample_sets()

    def get_pf_country_summary(self) -> pd.DataFrame:
        """
        Get summary of samples per country.

        Returns:
            DataFrame with country-level counts
        """
        df = self.pf7.sample_metadata()
        summary = df.groupby("country").size().reset_index(name="sample_count")
        return summary.sort_values("sample_count", ascending=False)

    # =========================================================================
    # Plasmodium vivax (Pv4) Methods
    # =========================================================================

    def get_pv_sample_metadata(self) -> pd.DataFrame:
        """
        Get P. vivax sample metadata.

        Returns:
            DataFrame with sample information
        """
        return self.pv4.sample_metadata()

    def get_pv_samples_by_country(self, country: str) -> pd.DataFrame:
        """
        Get P. vivax samples from a specific country.

        Args:
            country: Country name

        Returns:
            Filtered DataFrame
        """
        df = self.pv4.sample_metadata()
        return df[df["country"] == country]

    # =========================================================================
    # Anopheles (Ag3) Methods
    # =========================================================================

    def get_ag_sample_metadata(self) -> pd.DataFrame:
        """
        Get Anopheles gambiae sample metadata.

        Returns:
            DataFrame with sample information
        """
        return self.ag3.sample_metadata()

    def get_ag_samples_by_country(self, country: str) -> pd.DataFrame:
        """
        Get Anopheles samples from a specific country.

        Args:
            country: Country name

        Returns:
            Filtered DataFrame
        """
        df = self.ag3.sample_metadata()
        return df[df["country"] == country]

    def get_ag_sample_sets(self) -> pd.DataFrame:
        """
        Get available Anopheles sample sets.

        Returns:
            DataFrame with sample set information
        """
        return self.ag3.sample_sets()

    # =========================================================================
    # Cross-dataset Methods
    # =========================================================================

    def get_dataset_summary(self) -> pd.DataFrame:
        """
        Get summary of all available datasets.

        Returns:
            DataFrame with dataset summaries
        """
        summaries = []

        try:
            pf_meta = self.pf7.sample_metadata()
            summaries.append(
                {
                    "dataset": "Pf7",
                    "organism": "Plasmodium falciparum",
                    "sample_count": len(pf_meta),
                    "countries": pf_meta["country"].nunique() if "country" in pf_meta.columns else 0,
                }
            )
        except Exception:
            pass

        try:
            pv_meta = self.pv4.sample_metadata()
            summaries.append(
                {
                    "dataset": "Pv4",
                    "organism": "Plasmodium vivax",
                    "sample_count": len(pv_meta),
                    "countries": pv_meta["country"].nunique() if "country" in pv_meta.columns else 0,
                }
            )
        except Exception:
            pass

        try:
            ag_meta = self.ag3.sample_metadata()
            summaries.append(
                {
                    "dataset": "Ag3",
                    "organism": "Anopheles gambiae",
                    "sample_count": len(ag_meta),
                    "countries": ag_meta["country"].nunique() if "country" in ag_meta.columns else 0,
                }
            )
        except Exception:
            pass

        return pd.DataFrame(summaries)

    def search_samples(self, query: str) -> pd.DataFrame:
        """
        Search across all datasets for samples matching a query.

        Args:
            query: Search string (matches country, region, etc.)

        Returns:
            Combined DataFrame with matching samples
        """
        results = []

        for dataset_name, get_metadata in [
            ("Pf7", lambda: self.pf7.sample_metadata()),
            ("Pv4", lambda: self.pv4.sample_metadata()),
            ("Ag3", lambda: self.ag3.sample_metadata()),
        ]:
            try:
                df = get_metadata()
                # Search across string columns
                mask = df.apply(
                    lambda row: any(
                        query.lower() in str(v).lower()
                        for v in row.values
                        if isinstance(v, str)
                    ),
                    axis=1,
                )
                matches = df[mask].copy()
                if not matches.empty:
                    matches["_dataset"] = dataset_name
                    results.append(matches)
            except Exception:
                continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
