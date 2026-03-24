from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_ASSET = REPO_ROOT / "mcp-tools.core.json"
FULL_ASSET = REPO_ROOT / "mcp-tools.full.json"
PLUGIN_SAMPLE = REPO_ROOT / "microsoft-plugin.sample.json"


def _read_json(path: Path) -> dict:
    assert path.exists(), f"Missing asset: {path.name}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path.name} must contain a JSON object."
    return payload


def test_core_and_full_tool_assets_exist_and_use_streamable_http() -> None:
    core = _read_json(CORE_ASSET)
    full = _read_json(FULL_ASSET)

    assert core["transport"]["type"] == "streamable-http"
    assert full["transport"]["type"] == "streamable-http"
    assert core["transport"]["path"] == "/mcp"
    assert full["transport"]["path"] == "/mcp"

    assert core["tools"] == [
        "search_papers",
        "resolve_citation",
        "search_papers_match",
        "get_paper_details",
        "get_paper_metadata_crossref",
        "get_paper_open_access_unpaywall",
        "enrich_paper",
        "search_papers_bulk",
        "get_paper_citations",
        "get_paper_references",
        "search_authors",
        "get_author_info",
        "get_author_papers",
    ]
    assert "search_papers_smart" in full["tools"]
    assert "resolve_citation" in full["tools"]
    assert "get_paper_metadata_crossref" in full["tools"]
    assert "get_paper_open_access_unpaywall" in full["tools"]
    assert "enrich_paper" in full["tools"]
    assert "ask_result_set" in full["tools"]
    assert "map_research_landscape" in full["tools"]
    assert "expand_research_graph" in full["tools"]


def test_microsoft_plugin_sample_points_at_full_tool_asset() -> None:
    sample = _read_json(PLUGIN_SAMPLE)

    assert sample["tool_package"] == "mcp-tools.full.json"
    assert len(sample["conversation_starters"]) >= 5
    assert any("citation" in starter.lower() for starter in sample["conversation_starters"])
    assert any("searchSessionId" in note for note in sample["notes"])
