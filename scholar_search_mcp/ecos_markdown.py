"""MarkItDown-backed ECOS document conversion helpers."""

from __future__ import annotations

from .document_markdown import (
    DocumentConversionError,
    DocumentConversionTimeoutError,
    DocumentMarkdownConverter,
    UnsupportedDocumentTypeError,
)
from .document_markdown import (
    guess_document_title as _guess_document_title,
)

guess_document_title = _guess_document_title


class EcosDocumentConversionError(DocumentConversionError):
    """Base error for ECOS document conversion failures."""


class EcosUnsupportedDocumentTypeError(EcosDocumentConversionError, UnsupportedDocumentTypeError):
    """Raised when a fetched ECOS document type is not supported in v1."""


class EcosMarkdownConverter(DocumentMarkdownConverter):
    """Compatibility wrapper over the shared document converter."""

    def convert(
        self,
        *,
        content: bytes,
        source_url: str,
        content_type: str | None,
        filename: str | None = None,
    ) -> str:
        try:
            return super().convert(
                content=content,
                source_url=source_url,
                content_type=content_type,
                filename=filename,
            )
        except UnsupportedDocumentTypeError as exc:
            raise EcosUnsupportedDocumentTypeError(str(exc)) from exc
        except DocumentConversionError as exc:
            raise EcosDocumentConversionError(str(exc)) from exc

    def convert_with_timeout(
        self,
        *,
        content: bytes,
        source_url: str,
        content_type: str | None,
        timeout_seconds: float,
        filename: str | None = None,
    ) -> str:
        try:
            return super().convert_with_timeout(
                content=content,
                source_url=source_url,
                content_type=content_type,
                timeout_seconds=timeout_seconds,
                filename=filename,
            )
        except UnsupportedDocumentTypeError as exc:
            raise EcosUnsupportedDocumentTypeError(str(exc)) from exc
        except DocumentConversionTimeoutError as exc:
            raise EcosDocumentConversionError(str(exc)) from exc
        except DocumentConversionError as exc:
            raise EcosDocumentConversionError(str(exc)) from exc
