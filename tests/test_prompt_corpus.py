from __future__ import annotations

import json
from pathlib import Path


def test_ux_prompt_corpus_covers_primary_benchmark_domains() -> None:
    corpus_path = Path(__file__).resolve().parent / "fixtures" / "ux_prompt_corpus.json"
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))

    assert set(payload) >= {
        "environmental_science_and_ecology",
        "environmental_science_follow_up_quality",
        "environmental_science_inspect_rationale",
        "consulting_and_due_diligence",
        "citation_repair",
        "guided_low_context_success",
        "safe_abstention_and_clarification",
        "regulatory_primary_source_correctness",
        "runtime_summary_truth",
        "cross_discipline_generalization",
    }
    assert len(payload["environmental_science_and_ecology"]) >= 10
    assert len(payload["environmental_science_follow_up_quality"]) >= 6
    assert len(payload["environmental_science_inspect_rationale"]) >= 6
    assert len(payload["consulting_and_due_diligence"]) >= 8
    assert len(payload["citation_repair"]) >= 8
    assert len(payload["guided_low_context_success"]) >= 8
    assert len(payload["safe_abstention_and_clarification"]) >= 8
    assert len(payload["regulatory_primary_source_correctness"]) >= 8
    assert len(payload["runtime_summary_truth"]) >= 8
    assert len(payload["cross_discipline_generalization"]) >= 8
    assert any("PFAS remediation" in prompt for prompt in payload["environmental_science_and_ecology"])
    assert any("wetland restoration strategies" in prompt for prompt in payload["environmental_science_and_ecology"])
    assert any("eDNA" in prompt for prompt in payload["environmental_science_follow_up_quality"])
    assert any("desert tortoise" in prompt for prompt in payload["environmental_science_inspect_rationale"])
    assert any("wildfire risk mitigation" in prompt for prompt in payload["consulting_and_due_diligence"])
    assert any(
        "prescribed fire versus mechanical thinning" in prompt for prompt in payload["consulting_and_due_diligence"]
    )
    assert any(
        "Rockstrom et al planetary boundaries 2009 Nature 461 472" in prompt for prompt in payload["citation_repair"]
    )
    assert any("trustworthy findings" in prompt for prompt in payload["guided_low_context_success"])
    assert any("no evidence" in prompt for prompt in payload["safe_abstention_and_clarification"])
    assert any("California condor" in prompt for prompt in payload["regulatory_primary_source_correctness"])
    assert any("active profile" in prompt for prompt in payload["runtime_summary_truth"])
    assert any("solid-state battery degradation" in prompt for prompt in payload["cross_discipline_generalization"])


def test_ux_prompt_corpus_includes_workstream_g_cross_domain_slices() -> None:
    corpus_path = Path(__file__).resolve().parent / "fixtures" / "ux_prompt_corpus.json"
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))

    assert "cross_domain_slices" in payload
    slices = payload["cross_domain_slices"]
    assert isinstance(slices, dict)

    required_slices = {
        "natural_science",
        "human_dimensions",
        "heritage_archaeology",
        "regulation",
        "known_item_recovery",
    }
    assert required_slices.issubset(set(slices) - {"description"})

    required_buckets = {"positive_controls", "ambiguous", "mixed_mode", "failure_expected"}
    for slice_name in required_slices:
        slice_payload = slices[slice_name]
        assert isinstance(slice_payload, dict), slice_name
        assert required_buckets.issubset(slice_payload.keys()), slice_name
        assert slice_payload.get("description"), slice_name
        for bucket in required_buckets:
            items = slice_payload[bucket]
            assert isinstance(items, list) and items, f"{slice_name}.{bucket}"
            assert all(isinstance(item, str) and item.strip() for item in items), f"{slice_name}.{bucket}"

    # Positive-control anchors: ensure each slice references a realistic domain signal.
    assert any("PFAS" in prompt for prompt in slices["natural_science"]["positive_controls"])
    assert any(
        "Indigenous" in prompt or "co-management" in prompt
        for prompt in slices["human_dimensions"]["positive_controls"]
    )
    assert any(
        "archaeological" in prompt or "Section 106" in prompt
        for prompt in slices["heritage_archaeology"]["positive_controls"]
    )
    assert any("50 CFR" in prompt for prompt in slices["regulation"]["positive_controls"])
    assert any(
        "Planetary boundaries" in prompt or "Attention Is All You Need" in prompt
        for prompt in slices["known_item_recovery"]["positive_controls"]
    )

    # Failure-expected buckets should include abstention-probing prompts, not filler.
    assert any("Medicare" in prompt for prompt in slices["heritage_archaeology"]["failure_expected"])
    assert any(
        "never finalized" in prompt or "never been listed" in prompt
        for prompt in slices["regulation"]["failure_expected"]
    )
