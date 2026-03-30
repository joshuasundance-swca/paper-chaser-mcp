"""ScholarAPI client subpackage."""

from .client import ScholarApiClient
from .errors import (
    ScholarApiError,
    ScholarApiKeyMissingError,
    ScholarApiQuotaError,
    ScholarApiUpstreamError,
)

__all__ = [
    "ScholarApiClient",
    "ScholarApiError",
    "ScholarApiKeyMissingError",
    "ScholarApiQuotaError",
    "ScholarApiUpstreamError",
]
