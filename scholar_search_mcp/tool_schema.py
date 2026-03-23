"""Published MCP tool schema helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def sanitize_published_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove client-hostile noise from advertised MCP tool schemas."""
    sanitized = _sanitize_schema_node(deepcopy(schema))
    if isinstance(sanitized, dict) and "$defs" in sanitized:
        if not _schema_uses_refs(sanitized):
            sanitized.pop("$defs", None)
    return sanitized


def _sanitize_schema_node(node: Any) -> Any:
    if isinstance(node, list):
        return [_sanitize_schema_node(item) for item in node]
    if not isinstance(node, dict):
        return node

    if "anyOf" in node:
        non_null = [
            option
            for option in node["anyOf"]
            if not (isinstance(option, dict) and option.get("type") == "null")
        ]
        if len(non_null) == 1:
            merged = _sanitize_schema_node(non_null[0])
            if isinstance(merged, dict):
                for key in ("description", "examples", "enum"):
                    if key not in merged and key in node:
                        merged[key] = node[key]
            return merged

    sanitized: dict[str, Any] = {}
    for key, value in node.items():
        if key in {"default", "examples"}:
            continue
        sanitized[key] = _sanitize_schema_node(value)
    return sanitized


def _schema_uses_refs(node: Any) -> bool:
    if isinstance(node, dict):
        if "$ref" in node:
            return True
        return any(_schema_uses_refs(value) for value in node.values())
    if isinstance(node, list):
        return any(_schema_uses_refs(item) for item in node)
    return False
