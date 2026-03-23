"""MarkItDown-backed ECOS document conversion helpers."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


class EcosDocumentConversionError(RuntimeError):
    """Base error for ECOS document conversion failures."""


class EcosUnsupportedDocumentTypeError(EcosDocumentConversionError):
    """Raised when a fetched ECOS document type is not supported in v1."""


class EcosMarkdownConverter:
    """Thin adapter around MarkItDown with plugins disabled."""

    _CONTENT_TYPE_TO_EXTENSION = {
        "application/pdf": ".pdf",
        "text/html": ".html",
        "application/xhtml+xml": ".html",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "application/json": ".json",
        "application/xml": ".xml",
        "text/xml": ".xml",
    }

    def __init__(self) -> None:
        self._converter: Any | None = None

    def _get_converter(self) -> Any:
        if self._converter is None:
            try:
                from markitdown import MarkItDown
            except ImportError as exc:  # pragma: no cover - exercised via tests
                raise EcosDocumentConversionError(
                    "MarkItDown is not installed. Install the core dependency "
                    "set with markitdown[pdf] to enable ECOS document extraction."
                ) from exc
            self._converter = MarkItDown(enable_plugins=False)
        return self._converter

    @classmethod
    def infer_extension(
        cls,
        *,
        source_url: str,
        content_type: str | None,
        filename: str | None = None,
    ) -> str | None:
        normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
        if normalized_type in cls._CONTENT_TYPE_TO_EXTENSION:
            return cls._CONTENT_TYPE_TO_EXTENSION[normalized_type]

        for candidate in (filename, urlparse(source_url).path):
            if not candidate:
                continue
            extension = Path(unquote(str(candidate))).suffix.lower()
            if extension in {
                ".pdf",
                ".html",
                ".htm",
                ".txt",
                ".csv",
                ".json",
                ".xml",
            }:
                return ".html" if extension == ".htm" else extension
        return None

    def convert(
        self,
        *,
        content: bytes,
        source_url: str,
        content_type: str | None,
        filename: str | None = None,
    ) -> str:
        extension = self.infer_extension(
            source_url=source_url,
            content_type=content_type,
            filename=filename,
        )
        if extension is None:
            raise EcosUnsupportedDocumentTypeError(
                f"Unsupported ECOS document type: {content_type or 'unknown'}"
            )

        converter = self._get_converter()
        result = converter.convert_stream(
            io.BytesIO(content),
            file_extension=extension,
            url=source_url,
        )
        markdown = str(getattr(result, "text_content", "") or "").replace("\r\n", "\n")
        return markdown.strip()


def guess_document_title(source_url: str, filename: str | None = None) -> str:
    """Return a human-readable title guess from the response filename or URL."""

    candidate = filename or os.path.basename(urlparse(source_url).path)
    normalized = unquote(candidate or "").strip()
    stem = Path(normalized).stem if normalized else ""
    return stem or "ECOS document"
