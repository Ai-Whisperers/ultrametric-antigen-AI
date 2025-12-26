"""
Domain exceptions - custom exceptions for the HIV analysis system.

All exceptions inherit from DomainError for easy catching.
"""


class DomainError(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(DomainError):
    """Raised when validation fails."""

    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value


class EncodingError(DomainError):
    """Raised when encoding fails."""

    def __init__(self, message: str, sequence: str = None, position: int = None):
        super().__init__(message, {"sequence": sequence, "position": position})
        self.sequence = sequence
        self.position = position


class SequenceError(DomainError):
    """Raised for sequence-related errors."""

    def __init__(self, message: str, sequence_id: str = None):
        super().__init__(message, {"sequence_id": sequence_id})
        self.sequence_id = sequence_id


class MutationError(DomainError):
    """Raised for mutation-related errors."""

    def __init__(self, message: str, mutation: str = None):
        super().__init__(message, {"mutation": mutation})
        self.mutation = mutation


class RepositoryError(DomainError):
    """Raised for data access errors."""

    def __init__(self, message: str, repository: str = None, query: dict = None):
        super().__init__(message, {"repository": repository, "query": query})
        self.repository = repository
        self.query = query


class PredictionError(DomainError):
    """Raised when prediction fails."""

    def __init__(self, message: str, model: str = None, input_data=None):
        super().__init__(message, {"model": model})
        self.model = model
        self.input_data = input_data


class ConfigurationError(DomainError):
    """Raised for configuration errors."""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, {"config_key": config_key})
        self.config_key = config_key
