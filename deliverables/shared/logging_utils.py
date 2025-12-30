# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Centralized logging utilities for bioinformatics tools.

Provides consistent logging configuration across all deliverables
with support for file and console output, log levels, and context.

Usage:
    from shared.logging_utils import get_logger, setup_logging

    # Setup logging once at application start
    setup_logging(level="INFO", log_file="analysis.log")

    # Get logger for specific module
    logger = get_logger("peptide_analysis")
    logger.info("Processing peptide: %s", sequence)
    logger.warning("Low confidence prediction: %.2f", confidence)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Custom log levels
ANALYSIS = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(ANALYSIS, "ANALYSIS")


class BioinformaticsFormatter(logging.Formatter):
    """Custom formatter for bioinformatics logging.

    Includes timestamp, level, module, and message with color support
    for console output.
    """

    # ANSI color codes for console
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "ANALYSIS": "\033[35m",  # Magenta
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m",      # Reset
    }

    def __init__(self, use_colors: bool = True, include_context: bool = True):
        """Initialize formatter.

        Args:
            use_colors: Enable ANSI colors for console
            include_context: Include module/function context
        """
        self.use_colors = use_colors
        self.include_context = include_context

        if include_context:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        else:
            fmt = "%(asctime)s | %(levelname)-8s | %(message)s"

        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        # Add color if enabled and outputting to console
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            reset = self.COLORS["RESET"]
            record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


class AnalysisLogger(logging.Logger):
    """Extended logger with bioinformatics-specific methods."""

    def analysis(self, msg: str, *args, **kwargs):
        """Log analysis result (custom level between INFO and WARNING).

        Args:
            msg: Message format string
            *args: Message format arguments
            **kwargs: Additional logging kwargs
        """
        if self.isEnabledFor(ANALYSIS):
            self._log(ANALYSIS, msg, args, **kwargs)

    def prediction(
        self,
        target: str,
        value: float,
        confidence: Optional[float] = None,
        **kwargs,
    ):
        """Log a prediction result.

        Args:
            target: What was predicted (e.g., "MIC", "HC50", "DDG")
            value: Predicted value
            confidence: Optional confidence score (0-1)
            **kwargs: Additional context (passed to log message)
        """
        msg = f"Prediction: {target}={value:.4g}"
        if confidence is not None:
            msg += f" (confidence={confidence:.2%})"
        if kwargs:
            context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            msg += f" [{context}]"
        self.info(msg)

    def model_metrics(self, name: str, metrics: dict):
        """Log model training/evaluation metrics.

        Args:
            name: Model name
            metrics: Dictionary of metric names and values
        """
        metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in metrics.items() if isinstance(v, (int, float)))
        self.analysis(f"Model '{name}': {metrics_str}")

    def peptide_analysis(self, sequence: str, properties: dict):
        """Log peptide analysis results.

        Args:
            sequence: Peptide sequence
            properties: Computed properties
        """
        props_str = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in properties.items())
        self.debug(f"Peptide {sequence[:15]}...: {props_str}")


# Set custom logger class
logging.setLoggerClass(AnalysisLogger)

# Module-level logger cache
_loggers: dict[str, AnalysisLogger] = {}
_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None,
    use_colors: bool = True,
    include_timestamps: bool = True,
) -> None:
    """Configure logging for the application.

    Should be called once at application startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional filename for file logging
        log_dir: Directory for log files (default: ./logs)
        use_colors: Enable ANSI colors in console output
        include_timestamps: Include timestamps in log messages
    """
    global _configured

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers
    root.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = BioinformaticsFormatter(use_colors=use_colors)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        if log_dir is None:
            log_dir = Path("./logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        file_path = log_dir / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = BioinformaticsFormatter(use_colors=False)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str = "biotools") -> AnalysisLogger:
    """Get a logger for the specified module/component.

    Args:
        name: Logger name (typically module name)

    Returns:
        Configured AnalysisLogger instance
    """
    global _configured

    if not _configured:
        # Default configuration if not explicitly setup
        setup_logging(level="INFO", use_colors=True)

    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger

    return _loggers[name]


class LogContext:
    """Context manager for temporary logging configuration.

    Useful for temporarily changing log level or adding context.

    Usage:
        with LogContext(level="DEBUG"):
            # Debug logging enabled here
            process_data()
        # Returns to original level
    """

    def __init__(
        self,
        level: Optional[str] = None,
        extra_handler: Optional[logging.Handler] = None,
    ):
        """Initialize context.

        Args:
            level: Temporary log level
            extra_handler: Additional handler to add temporarily
        """
        self.level = level
        self.extra_handler = extra_handler
        self._original_level = None
        self._root = logging.getLogger()

    def __enter__(self):
        """Enter context - apply temporary settings."""
        if self.level:
            self._original_level = self._root.level
            self._root.setLevel(getattr(logging, self.level.upper()))

        if self.extra_handler:
            self._root.addHandler(self.extra_handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original settings."""
        if self._original_level is not None:
            self._root.setLevel(self._original_level)

        if self.extra_handler:
            self._root.removeHandler(self.extra_handler)

        return False


def log_function_call(logger: Optional[AnalysisLogger] = None):
    """Decorator to log function entry and exit.

    Args:
        logger: Logger to use (default: module logger)

    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            log.debug(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                log.debug(f"Exiting {func.__name__}")
                return result
            except Exception as e:
                log.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


# Convenience exports
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
