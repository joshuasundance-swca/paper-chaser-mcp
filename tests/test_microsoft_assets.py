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
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
    ]
    assert "research" in full["tools"]
    assert "follow_up_research" in full["tools"]
    assert "resolve_reference" in full["tools"]
    assert "inspect_source" in full["tools"]
    assert "get_runtime_status" in full["tools"]
    assert "search_papers" in full["tools"]
    assert "search_papers_smart" in full["tools"]
    assert "resolve_citation" in full["tools"]
    assert "get_paper_metadata_crossref" in full["tools"]
    assert "get_paper_open_access_unpaywall" in full["tools"]
    assert "enrich_paper" in full["tools"]
    assert "search_federal_register" in full["tools"]
    assert "get_federal_register_document" in full["tools"]
    assert "get_cfr_text" in full["tools"]
    assert "ask_result_set" in full["tools"]
    assert "map_research_landscape" in full["tools"]
    assert "expand_research_graph" in full["tools"]
    assert "search_entities_openalex" in full["tools"]
    assert "search_papers_openalex_by_entity" in full["tools"]
    assert "search_papers_scholarapi" in full["tools"]
    assert "list_papers_scholarapi" in full["tools"]
    assert "get_paper_text_scholarapi" in full["tools"]
    assert "get_paper_texts_scholarapi" in full["tools"]
    assert "get_paper_pdf_scholarapi" in full["tools"]
    assert "search_species_ecos" in full["tools"]
    assert "get_species_profile_ecos" in full["tools"]
    assert "list_species_documents_ecos" in full["tools"]
    assert "get_document_text_ecos" in full["tools"]


def test_microsoft_plugin_sample_points_at_guided_tool_asset() -> None:
    sample = _read_json(PLUGIN_SAMPLE)

    assert sample["tool_package"] == "mcp-tools.core.json"
    assert len(sample["conversation_starters"]) >= 5
    assert "research" in sample["description_for_model"]
    assert "follow_up_research" in sample["description_for_model"]
    assert "resolve_reference" in sample["description_for_model"]
    assert "inspect_source" in sample["description_for_model"]
    assert any(
        "citation" in starter.lower() or "reference" in starter.lower() for starter in sample["conversation_starters"]
    )
    assert any(
        "regulatory" in starter.lower() or "cfr" in starter.lower() for starter in sample["conversation_starters"]
    )
    assert any("PAPER_CHASER_TOOL_PROFILE=expert" in note for note in sample["notes"])
    assert any("PAPER_CHASER_TOOL_PROFILE=guided" in note for note in sample["notes"])
    assert any("searchSessionId" in note for note in sample["notes"])
