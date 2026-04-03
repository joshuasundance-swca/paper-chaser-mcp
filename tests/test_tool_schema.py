from paper_chaser_mcp.tool_schema import sanitize_published_schema


def test_sanitize_published_schema_drops_unused_defs_and_client_hostile_fields() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "Unused": {
                "type": "string",
                "default": "ignored",
            }
        },
        "properties": {
            "query": {
                "anyOf": [
                    {"type": "string", "default": "ignored"},
                    {"type": "null"},
                ],
                "description": "Search query",
                "examples": ["rag"],
            }
        },
        "additionalProperties": False,
    }

    sanitized = sanitize_published_schema(schema)

    assert "$defs" not in sanitized
    assert "additionalProperties" not in sanitized
    assert sanitized["properties"]["query"] == {
        "type": "string",
        "description": "Search query",
        "examples": ["rag"],
    }


def test_sanitize_published_schema_keeps_defs_when_refs_remain() -> None:
    schema = {
        "type": "object",
        "$defs": {"Query": {"type": "string"}},
        "properties": {
            "query": {
                "$ref": "#/$defs/Query",
            }
        },
    }

    sanitized = sanitize_published_schema(schema)

    assert sanitized["$defs"] == {"Query": {"type": "string"}}
    assert sanitized["properties"]["query"] == {"$ref": "#/$defs/Query"}
