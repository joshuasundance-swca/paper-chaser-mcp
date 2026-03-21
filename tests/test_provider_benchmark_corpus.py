from __future__ import annotations

import json
from pathlib import Path


def test_provider_benchmark_corpus_covers_required_scenarios() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "provider_benchmark_corpus.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert payload["version"] == 1
    cases = payload["cases"]
    assert cases

    categories = {case["category"] for case in cases}
    assert {
        "doi_lookup",
        "title_citation_repair",
        "author_disambiguation",
        "broad_topical_discovery",
        "citation_expansion",
        "smart_landscape_mapping",
        "provider_outage",
    }.issubset(categories)

    outage_providers = {
        case["provider"]
        for case in cases
        if case["category"] == "provider_outage"
    }
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
