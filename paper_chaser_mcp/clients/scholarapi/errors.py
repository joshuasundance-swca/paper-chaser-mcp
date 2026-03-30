"""ScholarAPI-specific error types."""


class ScholarApiError(RuntimeError):
    """Base error for ScholarAPI failures."""


class ScholarApiKeyMissingError(ScholarApiError):
    """Raised when ScholarAPI is used without an API key."""


class ScholarApiQuotaError(ScholarApiError):
    """Raised when ScholarAPI reports insufficient credits or rate limiting."""


class ScholarApiUpstreamError(ScholarApiError):
    """Raised for transient ScholarAPI transport or upstream failures."""
