"""
AlphaFold Protein Structure Database API Client.

Rate-limited client for programmatic access to AlphaFold predictions.

Usage:
    from research.alphafold3.utils.alphafold_client import AlphaFoldClient

    client = AlphaFoldClient(delay=0.2)  # 5 req/s
    predictions = client.get_prediction("P00533")

Documentation: research/alphafold3/docs/ALPHAFOLD_API_REFERENCE.md
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)


class AlphaFoldAPIError(Exception):
    """Exception for AlphaFold API errors."""

    pass


class AlphaFoldClient:
    """
    Rate-limited client for AlphaFold API.

    Attributes:
        BASE_URL: AlphaFold API base URL
        FTP_BASE: FTP server for bulk downloads
    """

    BASE_URL = "https://alphafold.ebi.ac.uk/api"
    FTP_BASE = "ftp://ftp.ebi.ac.uk/pub/databases/alphafold/"

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        delay: float = 0.1,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize AlphaFold API client.

        Args:
            cache_dir: Directory for caching responses (None = no caching)
            delay: Seconds between requests (default 0.1 = 10 req/s max)
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_time = 0.0

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "TernaryVAE-Research/1.0 (https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics)",
                "Accept": "application/json",
            }
        )

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching enabled: {self.cache_dir}")

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _cache_key(self, url: str) -> str:
        """Generate cache filename from URL."""
        return hashlib.md5(url.encode()).hexdigest() + ".json"

    def _get_cached(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / self._cache_key(url)
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                cache_file.unlink()  # Remove corrupted cache
        return None

    def _save_cache(self, url: str, data: Any) -> None:
        """Save response to cache."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / self._cache_key(url)
        cache_file.write_text(json.dumps(data, indent=2))

    def _request(
        self, url: str, use_cache: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Make rate-limited request with caching and retries.

        Args:
            url: Full URL to request
            use_cache: Whether to use cache

        Returns:
            JSON response data

        Raises:
            AlphaFoldAPIError: On request failure
        """
        # Check cache first
        if use_cache:
            cached = self._get_cached(url)
            if cached is not None:
                logger.debug(f"Cache hit: {url}")
                return cached

        # Rate limit
        self._rate_limit()

        # Make request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()

                # Cache successful response
                if use_cache:
                    self._save_cache(url, data)

                return data

            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    raise AlphaFoldAPIError(f"Not found: {url}") from e
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    last_error = e
                else:
                    raise AlphaFoldAPIError(
                        f"HTTP {response.status_code}: {response.text}"
                    ) from e

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)

        raise AlphaFoldAPIError(f"Request failed after {self.max_retries} attempts") from last_error

    def get_prediction(self, uniprot_id: str) -> List[Dict[str, Any]]:
        """
        Get AlphaFold predictions for a UniProt accession.

        Args:
            uniprot_id: UniProt accession (e.g., 'P00533' for EGFR)

        Returns:
            List of model entries containing:
                - modelEntityId: Unique model ID
                - sequenceStart/End: Covered region
                - sequence: Amino acid sequence
                - cifUrl, pdbUrl, bcifUrl: Structure file URLs
                - paeDocUrl: PAE JSON URL
                - globalMetricValue: Mean pLDDT score
                - isUniProtReviewed: SwissProt (true) or TrEMBL
                - latestVersion: Model version

        Example:
            >>> client = AlphaFoldClient()
            >>> predictions = client.get_prediction("P00533")
            >>> print(predictions[0]['globalMetricValue'])  # pLDDT
            85.2
        """
        url = f"{self.BASE_URL}/prediction/{uniprot_id}"
        return self._request(url)

    def get_uniprot_summary(self, uniprot_id: str) -> Dict[str, Any]:
        """
        Get UniProt summary with AlphaFold model info.

        Args:
            uniprot_id: UniProt accession or entry name

        Returns:
            Summary object with UniProt details and structures
        """
        url = f"{self.BASE_URL}/uniprot/summary/{uniprot_id}.json"
        return self._request(url)

    def get_sequence_summary(
        self, identifier: str, id_type: str = "md5"
    ) -> Dict[str, Any]:
        """
        Get models by sequence or sequence MD5.

        Args:
            identifier: Sequence string or MD5 hash
            id_type: 'sequence' or 'md5'

        Returns:
            Sequence summary with available models
        """
        url = f"{self.BASE_URL}/sequence/summary?id={identifier}&type={id_type}"
        return self._request(url)

    def get_annotations(
        self, uniprot_id: str, annotation_type: str = "MUTAGEN"
    ) -> Dict[str, Any]:
        """
        Get variant annotations (e.g., AlphaMissense pathogenicity).

        Args:
            uniprot_id: UniProt accession
            annotation_type: Annotation type (default: 'MUTAGEN' for AlphaMissense)

        Returns:
            Annotation data with affected residues
        """
        url = f"{self.BASE_URL}/annotations/{uniprot_id}.json?type={annotation_type}"
        return self._request(url)

    def download_structure(
        self,
        uniprot_id: str,
        output_dir: Union[str, Path],
        format: str = "cif",
        overwrite: bool = False,
    ) -> Path:
        """
        Download structure file for a UniProt accession.

        Args:
            uniprot_id: UniProt accession
            output_dir: Directory to save file
            format: 'cif' (mmCIF), 'pdb', or 'bcif' (binary CIF)
            overwrite: Overwrite existing file

        Returns:
            Path to downloaded file

        Raises:
            AlphaFoldAPIError: If no prediction found or format unavailable
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = self.get_prediction(uniprot_id)
        if not predictions:
            raise AlphaFoldAPIError(f"No predictions found for {uniprot_id}")

        entry = predictions[0]
        url_key = f"{format}Url"
        if url_key not in entry:
            raise AlphaFoldAPIError(f"Format '{format}' not available for {uniprot_id}")

        url = entry[url_key]
        filename = url.split("/")[-1]
        output_path = output_dir / filename

        if output_path.exists() and not overwrite:
            logger.info(f"File exists, skipping: {output_path}")
            return output_path

        self._rate_limit()
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        output_path.write_bytes(response.content)
        logger.info(f"Downloaded: {output_path}")
        return output_path

    def download_pae(
        self,
        uniprot_id: str,
        output_dir: Union[str, Path],
        overwrite: bool = False,
    ) -> Path:
        """
        Download PAE (Predicted Aligned Error) JSON for a protein.

        Args:
            uniprot_id: UniProt accession
            output_dir: Directory to save file
            overwrite: Overwrite existing file

        Returns:
            Path to PAE JSON file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = self.get_prediction(uniprot_id)
        if not predictions:
            raise AlphaFoldAPIError(f"No predictions found for {uniprot_id}")

        entry = predictions[0]
        if "paeDocUrl" not in entry:
            raise AlphaFoldAPIError(f"PAE not available for {uniprot_id}")

        url = entry["paeDocUrl"]
        filename = url.split("/")[-1]
        output_path = output_dir / filename

        if output_path.exists() and not overwrite:
            logger.info(f"File exists, skipping: {output_path}")
            return output_path

        self._rate_limit()
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        output_path.write_bytes(response.content)
        logger.info(f"Downloaded PAE: {output_path}")
        return output_path

    def batch_get_predictions(
        self, uniprot_ids: List[str], progress: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get predictions for multiple UniProt IDs with progress tracking.

        Args:
            uniprot_ids: List of UniProt accessions
            progress: Show progress bar (requires tqdm)

        Returns:
            Dict mapping UniProt ID to predictions (empty list if not found)
        """
        results = {}

        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(uniprot_ids, desc="Fetching predictions")
            except ImportError:
                iterator = uniprot_ids
                logger.warning("tqdm not installed, progress bar disabled")
        else:
            iterator = uniprot_ids

        for uniprot_id in iterator:
            try:
                results[uniprot_id] = self.get_prediction(uniprot_id)
            except AlphaFoldAPIError as e:
                logger.warning(f"Failed to get {uniprot_id}: {e}")
                results[uniprot_id] = []

        return results


# Convenience functions
def get_structure_url(uniprot_id: str, format: str = "cif") -> str:
    """
    Get direct URL to structure file (no API call).

    Args:
        uniprot_id: UniProt accession
        format: 'cif', 'pdb', or 'bcif'

    Returns:
        URL string
    """
    format_map = {
        "cif": "model_v4.cif",
        "pdb": "model_v4.pdb",
        "bcif": "model_v4.bcif",
    }
    suffix = format_map.get(format, "model_v4.cif")
    return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-{suffix}"


def get_pae_url(uniprot_id: str) -> str:
    """
    Get direct URL to PAE JSON (no API call).

    Args:
        uniprot_id: UniProt accession

    Returns:
        URL string
    """
    return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v4.json"


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    client = AlphaFoldClient(
        cache_dir=Path("cache/alphafold"),
        delay=0.2,  # Conservative: 5 req/s
    )

    # Get EGFR structure info
    print("Fetching EGFR (P00533)...")
    predictions = client.get_prediction("P00533")

    if predictions:
        entry = predictions[0]
        print(f"Model ID: {entry['modelEntityId']}")
        print(f"pLDDT: {entry['globalMetricValue']}")
        print(f"Sequence length: {entry['sequenceEnd'] - entry['sequenceStart'] + 1}")
        print(f"PDB URL: {entry['pdbUrl']}")
    else:
        print("No predictions found")
