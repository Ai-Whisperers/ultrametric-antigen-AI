"""
Configuration settings for the data access module.

Loads configuration from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file from config directory or project root
config_dir = Path(__file__).parent
project_root = config_dir.parent.parent

# Try loading from multiple locations
for env_path in [config_dir / ".env", project_root / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break


class NCBIConfig:
    """NCBI/Entrez API configuration."""

    email: str = os.getenv("NCBI_EMAIL", "")
    api_key: Optional[str] = os.getenv("NCBI_API_KEY") or None

    # Rate limits: 3/sec without key, 10/sec with key
    @property
    def rate_limit(self) -> int:
        return 10 if self.api_key else 3


class HIVDBConfig:
    """Stanford HIVDB Sierra API configuration."""

    endpoint: str = os.getenv("HIVDB_ENDPOINT", "https://hivdb.stanford.edu/graphql")


class CBioPortalConfig:
    """cBioPortal API configuration."""

    url: str = os.getenv("CBIOPORTAL_URL", "https://www.cbioportal.org/api")
    token: Optional[str] = os.getenv("CBIOPORTAL_TOKEN") or None


class CARDConfig:
    """CARD (Comprehensive Antibiotic Resistance Database) configuration."""

    api_url: str = os.getenv("CARD_API_URL", "https://card.mcmaster.ca/api")


class BVBRCConfig:
    """BV-BRC (Bacterial and Viral Bioinformatics Resource Center) configuration."""

    api_url: str = os.getenv("BVBRC_API_URL", "https://www.bv-brc.org/api")


class Settings:
    """Main settings container."""

    # API configurations
    ncbi = NCBIConfig()
    hivdb = HIVDBConfig()
    cbioportal = CBioPortalConfig()
    card = CARDConfig()
    bvbrc = BVBRCConfig()

    # General settings
    cache_dir: Path = Path(os.getenv("DATA_CACHE_DIR", "./data_cache"))
    timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if not cls.ncbi.email:
            warnings.append("NCBI_EMAIL not set - required for NCBI/Entrez API access")

        if not cls.ncbi.api_key:
            warnings.append(
                "NCBI_API_KEY not set - limited to 3 requests/sec (get key at "
                "https://www.ncbi.nlm.nih.gov/account/settings/)"
            )

        return warnings

    @classmethod
    def ensure_cache_dir(cls) -> Path:
        """Ensure cache directory exists and return path."""
        cls.cache_dir.mkdir(parents=True, exist_ok=True)
        return cls.cache_dir


# Global settings instance
settings = Settings()
