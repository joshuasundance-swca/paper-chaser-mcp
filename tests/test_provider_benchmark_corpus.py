from __future__ import annotations

import json
from pathlib import Path

from paper_chaser_mcp.models.tools import TOOL_INPUT_MODELS


def test_provider_benchmark_corpus_covers_required_scenarios() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "provider_benchmark_corpus.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert payload["version"] == 2
    cases = payload["cases"]
    assert cases

    categories = {case["category"] for case in cases}
    assert {
        "guided_default_profile",
        "safe_abstention",
        "regulatory_correctness",
        "runtime_summary_truth",
        "doi_lookup",
        "title_citation_repair",
        "author_disambiguation",
        "broad_topical_discovery",
        "citation_expansion",
        "smart_landscape_mapping",
        "provider_outage",
    }.issubset(categories)

    case_ids = {case["id"] for case in cases}
    assert "guided_default_research_entrypoint" in case_ids
    assert "guided_runtime_truth_status" in case_ids
    assert "safe_abstention_unsupported_follow_up" in case_ids
    assert "regulatory_condor_primary_source_correctness" in case_ids
    assert "stress_core_science_microplastics_discovery" in case_ids
    assert "stress_data_drought_mariana_trench_amphipods" in case_ids
    assert "stress_regulatory_6ppd_primary_source" in case_ids
    assert "stress_pseudoscience_healing_crystals_abstention" in case_ids
    assert "stress_mixed_intent_guided_blend" in case_ids
    assert "stress_follow_up_answered_but_abstaining_fu4b" in case_ids

    outage_providers = {case["provider"] for case in cases if case["category"] == "provider_outage"}
    assert outage_providers == {
        "semantic_scholar",
        "openalex",
        "core",
        "serpapi_google_scholar",
        "arxiv",
        "openai",
    }

    for case in cases:
        assert case["id"]
        assert case["tool"]
        assert isinstance(case["inputs"], dict)
        assert isinstance(case["expected"], dict)
        assert case["tool"] in TOOL_INPUT_MODELS
        TOOL_INPUT_MODELS[case["tool"]].model_validate(case["inputs"])
