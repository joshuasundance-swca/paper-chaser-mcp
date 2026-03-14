"""Parsing helpers shared across provider clients."""

import re
from typing import Any, Optional


def _arxiv_id_from_url(id_url: str) -> str:
    """Extract an arXiv id from an abstract URL and strip any version suffix."""
    if not id_url:
        return ""
    match = re.search(r"arxiv\.org/abs/([\w.-]+)", id_url, re.I)
    if not match:
        return id_url
    raw = match.group(1)
    return re.sub(r"v\d+$", "", raw)


def _text(el: Optional[Any]) -> str:
    return (el.text or "").strip() if el is not None else ""