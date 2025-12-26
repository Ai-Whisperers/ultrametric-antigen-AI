"""
Application settings - centralized configuration.

Settings are loaded from environment variables with sensible defaults.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass(frozen=True)
class EncoderSettings:
    """Encoder configuration."""

    prime: int = 3
    embedding_dim: int = 3
    curvature: float = -1.0
    padic_precision: int = 10
    use_amino_acid_features: bool = True
    hyperbolic_scale: float = 0.5


@dataclass(frozen=True)
class DataSettings:
    """Data access configuration."""

    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds

    # Dataset paths
    stanford_dir: str = "research/datasets"
    lanl_dir: str = "research/datasets"
    catnap_dir: str = "research/datasets"


@dataclass(frozen=True)
class MLSettings:
    """Machine learning configuration."""

    device: str = "cuda"
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 10
    model_dir: Path = Path("models")


@dataclass(frozen=True)
class APISettings:
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    cors_origins: tuple[str, ...] = ("*",)


@dataclass(frozen=True)
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[Path] = None


@dataclass
class Settings:
    """
    Main settings container.

    Aggregates all configuration categories.
    """

    encoder: EncoderSettings = field(default_factory=EncoderSettings)
    data: DataSettings = field(default_factory=DataSettings)
    ml: MLSettings = field(default_factory=MLSettings)
    api: APISettings = field(default_factory=APISettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Load settings from environment variables.

        Environment variables follow pattern: SECTION_KEY
        e.g., ENCODER_PRIME, DATA_CACHE_ENABLED
        """
        return cls(
            encoder=EncoderSettings(
                prime=int(os.getenv("ENCODER_PRIME", "3")),
                embedding_dim=int(os.getenv("ENCODER_DIM", "3")),
                curvature=float(os.getenv("ENCODER_CURVATURE", "-1.0")),
                padic_precision=int(os.getenv("ENCODER_PRECISION", "10")),
            ),
            data=DataSettings(
                data_dir=Path(os.getenv("DATA_DIR", "data")),
                cache_dir=Path(os.getenv("CACHE_DIR", ".cache")),
                cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            ),
            ml=MLSettings(
                device=os.getenv("ML_DEVICE", "cuda"),
                batch_size=int(os.getenv("ML_BATCH_SIZE", "32")),
                learning_rate=float(os.getenv("ML_LR", "1e-3")),
            ),
            api=APISettings(
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8000")),
                debug=os.getenv("API_DEBUG", "false").lower() == "true",
            ),
            logging=LoggingSettings(
                level=os.getenv("LOG_LEVEL", "INFO"),
            ),
        )


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """Reset settings (for testing)."""
    global _settings
    _settings = None
