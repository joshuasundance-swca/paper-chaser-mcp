"""MCP tool definitions."""

from collections.abc import Mapping

from mcp.types import Tool

from .settings import ToolProfile
from .tool_schema import sanitize_published_schema
from .tool_specs import iter_visible_tool_specs
from .tool_specs.descriptions import OPAQUE_CURSOR_CONTRACT, TOOL_DESCRIPTIONS


def get_tool_definitions(
    *,
    tool_profile: ToolProfile = "guided",
    hide_disabled_tools: bool = False,
    enabled_flags: Mapping[str, bool] | None = None,
) -> list[Tool]:
    """Return the MCP tool schema exposed by the server."""
    return [
        Tool(
            name=spec.name,
            description=spec.description,
            inputSchema=sanitize_published_schema(spec.input_model.model_json_schema()),
        )
        for spec in iter_visible_tool_specs(
            tool_profile=tool_profile,
            hide_disabled_tools=hide_disabled_tools,
            enabled_flags=enabled_flags,
        )
    ]


__all__ = ["OPAQUE_CURSOR_CONTRACT", "TOOL_DESCRIPTIONS", "get_tool_definitions"]
