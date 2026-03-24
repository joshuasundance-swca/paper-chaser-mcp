from __future__ import annotations

import sys
import types

import pytest

from scholar_search_mcp.document_markdown import DocumentMarkdownConverter


def test_document_markdown_converter_extracts_html_without_markitdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingMarkItDown:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("MarkItDown should not be constructed for HTML extraction")

    monkeypatch.setitem(sys.modules, "markitdown", types.SimpleNamespace(MarkItDown=FailingMarkItDown))

    converter = DocumentMarkdownConverter()
    markdown = converter.convert(
        content=b"<html><body><h1>Notice</h1><p>Full text extraction works.</p></body></html>",
        source_url="https://example.com/notice.html",
        content_type="text/html",
        filename="notice.html",
    )

    assert "Notice" in markdown
    assert "Full text extraction works." in markdown


def test_document_markdown_converter_extracts_plain_text_without_markitdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingMarkItDown:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("MarkItDown should not be constructed for plain text extraction")

    monkeypatch.setitem(sys.modules, "markitdown", types.SimpleNamespace(MarkItDown=FailingMarkItDown))

    converter = DocumentMarkdownConverter()
    markdown = converter.convert_with_timeout(
        content=b"Line one\n\nLine two",
        source_url="https://example.com/document.txt",
        content_type="text/plain",
        timeout_seconds=0.01,
        filename="document.txt",
    )

    assert markdown == "Line one\n\nLine two"
