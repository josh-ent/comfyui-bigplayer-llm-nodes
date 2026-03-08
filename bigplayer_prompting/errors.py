class BigPlayerError(Exception):
    """Base error for user-facing node failures."""


class ModelNameExtractionError(BigPlayerError):
    """Raised when the connected MODEL object does not expose a usable name."""


class ProviderError(BigPlayerError):
    """Raised when the upstream LLM provider request fails."""


class MalformedProviderResponseError(ProviderError):
    """Raised when the provider returns data that cannot be parsed or validated."""


class UnsupportedOperationError(ProviderError):
    """Raised when a provider cannot execute the requested operation."""
