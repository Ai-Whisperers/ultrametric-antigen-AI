# AlphaFold Protein Structure Database - API Reference

**Doc-Type:** API Reference · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Overview

The AlphaFold API provides programmatic access to 241+ million predicted protein structures. All endpoints are keyed on UniProt accessions.

**Base URL:** `https://alphafold.ebi.ac.uk/api`

**Documentation:** https://alphafold.ebi.ac.uk/api-docs

**OpenAPI Spec:** https://alphafold.ebi.ac.uk/api/openapi.json

---

## Rate Limiting

> **IMPORTANT:** Be respectful of EMBL-EBI resources.

| Guideline | Recommendation |
|-----------|----------------|
| **Requests per second** | Max 10 req/s for batch operations |
| **Batch downloads** | Use FTP for bulk data (not API) |
| **Caching** | Cache responses locally (structures don't change) |
| **User-Agent** | Include project identifier in headers |
| **Contact** | alphafolddb@ebi.ac.uk for high-volume access |

**Best Practices:**
```python
import time

def rate_limited_request(url, delay=0.1):
    """Make request with rate limiting."""
    time.sleep(delay)  # 100ms between requests = 10 req/s max
    response = requests.get(url, headers={
        'User-Agent': 'TernaryVAE-Research/1.0 (contact@example.com)'
    })
    return response
```

---

## Endpoints

### 1. Get All Models for UniProt Accession

**Endpoint:** `GET /prediction/{qualifier}`

**Description:** Retrieve all AlphaFold models for a UniProt accession.

**Parameters:**

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `qualifier` | path | Yes | UniProt accession (e.g., `Q5VSL9`, `P00533`) |
| `sequence_checksum` | query | No | MD5 checksum of UniProt sequence |

**Example Request:**
```bash
curl "https://alphafold.ebi.ac.uk/api/prediction/P00533"
```

**Example Response:**
```json
[
  {
    "modelEntityId": "AF-P00533-F1",
    "sequenceStart": 1,
    "sequenceEnd": 1210,
    "sequence": "MRPSGTAGAALL...",
    "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v4.bcif",
    "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v4.cif",
    "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v4.pdb",
    "paeDocUrl": "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-predicted_aligned_error_v4.json",
    "isUniProtReviewed": true,
    "isUniProtReferenceProteome": true,
    "latestVersion": 4,
    "globalMetricValue": 85.2,
    "globalMetricType": "pLDDT"
  }
]
```

---

### 2. Get UniProt Summary

**Endpoint:** `GET /uniprot/summary/{qualifier}.json`

**Description:** Retrieve AlphaFold models for a specified UniProt entry.

**Parameters:**

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `qualifier` | path | Yes | UniProtKB accession, entry name, or MD5 checksum |

**Example Request:**
```bash
curl "https://alphafold.ebi.ac.uk/api/uniprot/summary/P00533.json"
```

---

### 3. Get Sequence Summary

**Endpoint:** `GET /sequence/summary`

**Description:** Retrieve models for a sequence (useful when UniProt ID unknown).

**Parameters:**

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `id` | query | Yes | Sequence or MD5 checksum |
| `type` | query | No | `"sequence"` (default) or `"md5"` |

**Example Request:**
```bash
# By sequence MD5
curl "https://alphafold.ebi.ac.uk/api/sequence/summary?id=abc123def456&type=md5"
```

---

### 4. Get Annotations (AlphaMissense)

**Endpoint:** `GET /annotations/{qualifier}.json`

**Description:** Retrieve variant annotations (e.g., AlphaMissense pathogenicity predictions).

**Parameters:**

| Parameter | Location | Required | Description |
|-----------|----------|----------|-------------|
| `qualifier` | path | Yes | UniProt accession |
| `type` | query | Yes | Annotation type (e.g., `MUTAGEN` for AlphaMissense) |

**Example Request:**
```bash
curl "https://alphafold.ebi.ac.uk/api/annotations/P00533.json?type=MUTAGEN"
```

---

## Response Schema

### NewEntrySummary Object

| Field | Type | Description |
|-------|------|-------------|
| `modelEntityId` | string | Unique model identifier (e.g., `AF-P00533-F1`) |
| `sequenceStart` | int | Start position in UniProt sequence |
| `sequenceEnd` | int | End position in UniProt sequence |
| `sequence` | string | Amino acid sequence |
| `bcifUrl` | string | URL to binary CIF file |
| `cifUrl` | string | URL to mmCIF file |
| `pdbUrl` | string | URL to PDB file |
| `paeDocUrl` | string | URL to PAE JSON (predicted aligned error) |
| `isUniProtReviewed` | bool | SwissProt (true) or TrEMBL (false) |
| `isUniProtReferenceProteome` | bool | Part of reference proteome |
| `latestVersion` | int | Model version number |
| `globalMetricValue` | float | Mean pLDDT score (0-100) |
| `globalMetricType` | string | `"pLDDT"` |

---

## API Deprecation Timeline

**Important:** Field names are changing. Dual support until 2026-06-25.

| Deprecated Field | New Field | Sunset Date |
|------------------|-----------|-------------|
| `entryId` | `modelEntityId` | 2026-06-25 |
| `uniprotStart` | `sequenceStart` | 2026-06-25 |
| `uniprotEnd` | `sequenceEnd` | 2026-06-25 |
| `uniprotSequence` | `sequence` | 2026-06-25 |
| `isReviewed` | `isUniProtReviewed` | 2026-06-25 |
| `isReferenceProteome` | `isUniProtReferenceProteome` | 2026-06-25 |
| `paeImageUrl` | **REMOVED** | 2026-06-25 |

---

## Bulk Downloads (Recommended for Large Datasets)

For downloading many structures, use FTP instead of API:

**FTP Base:** `ftp://ftp.ebi.ac.uk/pub/databases/alphafold/`

| Dataset | Path | Size |
|---------|------|------|
| Human proteome | `/latest/UP000005640_9606_HUMAN_v4.tar` | ~15 GB |
| Model organisms | `/latest/` | Various |
| Full database | `/latest/` | ~23 TB |

**Download Script:**
```bash
# Download human proteome
wget -r -np -nH --cut-dirs=3 \
    ftp://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar
```

---

## Python Client Example

```python
"""
AlphaFold API client with rate limiting.
Location: research/alphafold3/utils/alphafold_client.py
"""
import requests
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

class AlphaFoldClient:
    """Rate-limited client for AlphaFold API."""

    BASE_URL = "https://alphafold.ebi.ac.uk/api"

    def __init__(self, cache_dir: Optional[Path] = None, delay: float = 0.1):
        """
        Args:
            cache_dir: Directory for caching responses
            delay: Seconds between requests (default 0.1 = 10 req/s)
        """
        self.cache_dir = cache_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TernaryVAE-Research/1.0'
        })

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """Apply rate limiting."""
        time.sleep(self.delay)

    def _cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def get_prediction(self, uniprot_id: str) -> List[Dict[str, Any]]:
        """
        Get AlphaFold predictions for a UniProt accession.

        Args:
            uniprot_id: UniProt accession (e.g., 'P00533')

        Returns:
            List of model entries with URLs and metadata
        """
        self._rate_limit()
        url = f"{self.BASE_URL}/prediction/{uniprot_id}"

        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def download_structure(
        self,
        uniprot_id: str,
        output_dir: Path,
        format: str = "cif"
    ) -> Path:
        """
        Download structure file for a UniProt accession.

        Args:
            uniprot_id: UniProt accession
            output_dir: Directory to save file
            format: 'cif', 'pdb', or 'bcif'

        Returns:
            Path to downloaded file
        """
        predictions = self.get_prediction(uniprot_id)
        if not predictions:
            raise ValueError(f"No predictions found for {uniprot_id}")

        entry = predictions[0]
        url_key = f"{format}Url"
        if url_key not in entry:
            raise ValueError(f"Format {format} not available")

        url = entry[url_key]
        filename = url.split('/')[-1]
        output_path = output_dir / filename

        self._rate_limit()
        response = self.session.get(url)
        response.raise_for_status()

        output_path.write_bytes(response.content)
        return output_path

    def get_annotations(
        self,
        uniprot_id: str,
        annotation_type: str = "MUTAGEN"
    ) -> Dict[str, Any]:
        """
        Get annotations (e.g., AlphaMissense) for a protein.

        Args:
            uniprot_id: UniProt accession
            annotation_type: Type of annotation (default: MUTAGEN)

        Returns:
            Annotation data
        """
        self._rate_limit()
        url = f"{self.BASE_URL}/annotations/{uniprot_id}.json?type={annotation_type}"

        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


# Usage example
if __name__ == "__main__":
    client = AlphaFoldClient(
        cache_dir=Path("cache/alphafold"),
        delay=0.2  # 5 req/s for safety
    )

    # Get EGFR structure info
    predictions = client.get_prediction("P00533")
    print(f"Found {len(predictions)} models")
    print(f"pLDDT: {predictions[0]['globalMetricValue']}")
    print(f"PDB URL: {predictions[0]['pdbUrl']}")
```

---

## File Formats

### mmCIF (.cif)
- Standard macromolecular format
- Contains full atomic coordinates
- Recommended for most analyses

### Binary CIF (.bcif)
- Compressed mmCIF
- Faster to download/parse
- Use for large-scale processing

### PDB (.pdb)
- Legacy format
- Limited to 99,999 atoms
- Use for compatibility with older tools

### PAE JSON (.json)
- Predicted Aligned Error matrix
- Per-residue confidence in relative positions
- Essential for domain boundary detection

---

## Quality Metrics

### pLDDT (per-residue confidence)

| Score | Interpretation |
|-------|----------------|
| > 90 | Very high confidence |
| 70-90 | High confidence |
| 50-70 | Low confidence |
| < 50 | Very low confidence (likely disordered) |

### PAE (Predicted Aligned Error)

- Matrix of expected position errors between residue pairs
- Low PAE (<5Å) indicates confident relative positioning
- High PAE suggests flexible/uncertain regions

---

## Related Documentation

- [DATASETS_INDEX.md](../../../docs/DATASETS_INDEX.md) - Centralized dataset registry
- [AlphaFold Download Page](https://alphafold.ebi.ac.uk/download) - Bulk downloads
- [EBI Proteins API](https://www.ebi.ac.uk/proteins/api/doc/) - Related UniProt API

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-03 | 1.0 | Initial API reference with 4 endpoints |

---

## Sources

- [AlphaFold API Documentation](https://alphafold.ebi.ac.uk/api-docs)
- [AlphaFold Database Release Notes](https://www.ebi.ac.uk/pdbe/news/alphafold-database-release-notes)
- [API Deprecation Timeline](https://www.ebi.ac.uk/pdbe/news/breaking-changes-afdb-predictions-api)
- [AlphaFold Database 2024 Paper](https://academic.oup.com/nar/article/52/D1/D368/7337620)
