"""Shared MarkItDown-backed document conversion helpers."""

from __future__ import annotations

import io
import json
import multiprocessing
import os
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


class DocumentConversionError(RuntimeError):
    """Base error for document conversion failures."""


class UnsupportedDocumentTypeError(DocumentConversionError):
    """Raised when a fetched document type is not supported."""


class DocumentConversionTimeoutError(DocumentConversionError):
    """Raised when document conversion exceeds the configured hard timeout."""


class _HtmlToMarkdownParser(HTMLParser):
    _BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "ol",
        "p",
        "section",
        "table",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        normalized = " ".join(unescape(data).split())
        if normalized:
            self._parts.append(normalized)

    def markdown(self) -> str:
        text = " ".join(self._parts)
        lines = [" ".join(line.split()) for line in text.splitlines()]
        filtered = [line for line in lines if line]
        return "\n\n".join(filtered).strip()


def _run_markitdown_conversion(
    *,
    content: bytes,
    source_url: str,
    file_extension: str,
) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise DocumentConversionError(
            "MarkItDown is not installed. Install the core dependency "
            "set with markitdown[pdf] to enable document extraction."
        ) from exc
    converter = MarkItDown(enable_plugins=False)
    result = converter.convert_stream(
        io.BytesIO(content),
        file_extension=file_extension,
        url=source_url,
    )
    markdown = str(getattr(result, "text_content", "") or "").replace("\r\n", "\n")
    return markdown.strip()


def _markitdown_worker(queue: Any, content: bytes, source_url: str, file_extension: str) -> None:
    try:
        queue.put(
            ("ok", _run_markitdown_conversion(content=content, source_url=source_url, file_extension=file_extension))
        )
    except Exception as exc:  # pragma: no cover - subprocess exception forwarding
        queue.put(("error", f"{type(exc).__name__}:{exc}"))


class DocumentMarkdownConverter:
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
                raise DocumentConversionError(
                    "MarkItDown is not installed. Install the core dependency "
                    "set with markitdown[pdf] to enable document extraction."
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
            raise UnsupportedDocumentTypeError(f"Unsupported document type: {content_type or 'unknown'}")

        return self._convert_by_extension(
            content=content,
            source_url=source_url,
            extension=extension,
        )

    def convert_with_timeout(
        self,
        *,
        content: bytes,
        source_url: str,
        content_type: str | None,
        timeout_seconds: float,
        filename: str | None = None,
    ) -> str:
        extension = self.infer_extension(
            source_url=source_url,
            content_type=content_type,
            filename=filename,
        )
        if extension is None:
            raise UnsupportedDocumentTypeError(f"Unsupported document type: {content_type or 'unknown'}")

        if extension != ".pdf":
            return self._convert_by_extension(
                content=content,
                source_url=source_url,
                extension=extension,
            )

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(
            target=_markitdown_worker,
            args=(queue, bytes(content), source_url, extension),
        )
        process.start()
        process.join(timeout=max(float(timeout_seconds), 0.001))
        if process.is_alive():
            process.terminate()
            process.join(5)
            raise DocumentConversionTimeoutError(
                "Document conversion exceeded the configured timeout before Markdown extraction finished."
            )
        if queue.empty():
            raise DocumentConversionError("Document conversion failed without returning a result.")
        status, payload = queue.get()
        if status == "ok":
            return str(payload)
        if isinstance(payload, str) and payload.startswith("DocumentConversionError:"):
            raise DocumentConversionError(payload.split(":", 1)[1])
        raise DocumentConversionError(str(payload))

    def _convert_by_extension(
        self,
        *,
        content: bytes,
        source_url: str,
        extension: str,
    ) -> str:
        if extension in {".txt", ".csv"}:
            return self._decode_text(content)
        if extension == ".json":
            return self._convert_json(content)
        if extension in {".html", ".xml"}:
            return self._convert_markup(content)

        return _run_markitdown_conversion(
            content=content,
            source_url=source_url,
            file_extension=extension,
        )

    @staticmethod
    def _decode_text(content: bytes) -> str:
        return content.decode("utf-8", errors="replace").replace("\r\n", "\n").strip()

    @classmethod
    def _convert_json(cls, content: bytes) -> str:
        decoded = cls._decode_text(content)
        try:
            return json.dumps(json.loads(decoded), indent=2, sort_keys=True)
        except json.JSONDecodeError:
            return decoded

    @classmethod
    def _convert_markup(cls, content: bytes) -> str:
        decoded = cls._decode_text(content)
        parser = _HtmlToMarkdownParser()
        parser.feed(decoded)
        parser.close()
        extracted = parser.markdown()
        return extracted or decoded


def guess_document_title(source_url: str, filename: str | None = None) -> str:
    """Return a human-readable title guess from the response filename or URL."""

    candidate = filename or os.path.basename(urlparse(source_url).path)
    normalized = unquote(candidate or "").strip()
    stem = Path(normalized).stem if normalized else ""
    return stem or "Document"
