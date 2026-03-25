"""MCP tool definitions."""

from mcp.types import Tool

from .tool_schema import sanitize_published_schema
from .tool_specs import iter_tool_specs
from .tool_specs.descriptions import OPAQUE_CURSOR_CONTRACT, TOOL_DESCRIPTIONS


def get_tool_definitions() -> list[Tool]:
    """Return the MCP tool schema exposed by the server."""
    return [
        Tool(
            name=spec.name,
            description=spec.description,
            inputSchema=sanitize_published_schema(spec.input_model.model_json_schema()),
        )
        for spec in iter_tool_specs()
    ]


__all__ = ["OPAQUE_CURSOR_CONTRACT", "TOOL_DESCRIPTIONS", "get_tool_definitions"]
