"""Pure helpers for FastMCP tool registration used by the Paper Chaser server."""

from __future__ import annotations

from inspect import Parameter, Signature
from typing import Any, cast

from fastmcp import Context
from pydantic import Field
from pydantic_core import PydanticUndefined

from ..tool_schema import sanitize_published_schema
from ..tool_specs import get_tool_spec


def _format_tool_display_name(name: str) -> str:
    return name.replace("_", " ").title()


def _tool_tags(name: str) -> set[str]:
    return set(get_tool_spec(name).tags)


def _parameter_name(field_name: str, alias: str | None) -> str:
    return alias or field_name


def _parameter_default(model_field: Any) -> Any:
    if model_field.is_required():
        return Parameter.empty
    default = model_field.default
    if default is PydanticUndefined:
        default = None
    return Field(default=default, description=model_field.description)


def _build_signature(model: Any) -> tuple[Signature, dict[str, Any]]:
    parameters: list[Parameter] = []
    annotations: dict[str, Any] = {"return": dict[str, Any]}
    for field_name, model_field in model.model_fields.items():
        parameter_name = _parameter_name(field_name, model_field.alias)
        annotations[parameter_name] = model_field.annotation
        parameters.append(
            Parameter(
                parameter_name,
                Parameter.KEYWORD_ONLY,
                annotation=model_field.annotation,
                default=_parameter_default(model_field),
            )
        )
    annotations["ctx"] = Context
    parameters.append(
        Parameter(
            "ctx",
            Parameter.KEYWORD_ONLY,
            annotation=Context,
            default=None,
        )
    )
    return Signature(parameters=parameters), annotations


def _sanitize_registered_tool_schema(app: Any, tool_name: str) -> None:
    tool = cast(Any, app.local_provider._components[f"tool:{tool_name}@"])
    tool.parameters = sanitize_published_schema(tool.parameters)


__all__ = [
    "_build_signature",
    "_format_tool_display_name",
    "_parameter_default",
    "_parameter_name",
    "_sanitize_registered_tool_schema",
    "_tool_tags",
]
