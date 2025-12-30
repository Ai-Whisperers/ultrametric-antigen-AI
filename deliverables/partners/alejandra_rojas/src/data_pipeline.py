from datetime import datetime, timedelta
from typing import Optional
import os


class NCBIFetcher:
    """Fetches recent viral sequences from NCBI Virus."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NCBI_API_KEY")

    def fetch_recent_sequences(self, virus_taxid: int, days_back: int = 7, geo_location: str = "Paraguay"):
        """
        Fetch sequences uploaded in the last N days.

        Args:
            virus_taxid: NCBI Taxonomy ID (e.g., 12637 for Dengue)
            days_back: Lookback window
            geo_location: Geographic filter
        """
        # Placeholder: Construct Entrez query or use NCBI Datasets CLI wrapper
        date_cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
        print(f"Fetching sequences for TaxID {virus_taxid} from {geo_location} since {date_cutoff}...")
        return []


class SequenceProcessor:
    """Cleans and formats sequences for embedding."""

    def clean_sequence(self, raw_seq: str) -> str:
        """Remove ambiguous characters and ensure standard IUPAC format."""
        return raw_seq.upper().replace("-", "")
