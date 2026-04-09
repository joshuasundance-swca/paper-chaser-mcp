from __future__ import annotations

import json
from pathlib import Path


def test_ux_prompt_corpus_covers_primary_benchmark_domains() -> None:
    corpus_path = Path(__file__).resolve().parent / "fixtures" / "ux_prompt_corpus.json"
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))

    assert set(payload) == {
        "environmental_science_and_ecology",
        "consulting_and_due_diligence",
        "citation_repair",
        "guided_low_context_success",
        "safe_abstention_and_clarification",
        "regulatory_primary_source_correctness",
        "runtime_summary_truth",
        "cross_discipline_generalization",
    }
    assert len(payload["environmental_science_and_ecology"]) >= 10
    assert len(payload["consulting_and_due_diligence"]) >= 8
    assert len(payload["citation_repair"]) >= 8
    assert len(payload["guided_low_context_success"]) >= 8
    assert len(payload["safe_abstention_and_clarification"]) >= 8
    assert len(payload["regulatory_primary_source_correctness"]) >= 8
    assert len(payload["runtime_summary_truth"]) >= 8
    assert len(payload["cross_discipline_generalization"]) >= 8
    assert any("PFAS remediation" in prompt for prompt in payload["environmental_science_and_ecology"])
    assert any("wetland restoration strategies" in prompt for prompt in payload["environmental_science_and_ecology"])
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
