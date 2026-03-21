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
        "cross_discipline_generalization",
    }
    assert len(payload["environmental_science_and_ecology"]) >= 10
    assert len(payload["consulting_and_due_diligence"]) >= 8
    assert len(payload["citation_repair"]) >= 8
    assert len(payload["cross_discipline_generalization"]) >= 8
    assert any(
        "PFAS remediation" in prompt
        for prompt in payload["environmental_science_and_ecology"]
    )
    assert any(
        "wildfire risk mitigation" in prompt
        for prompt in payload["consulting_and_due_diligence"]
    )
    assert any(
        "Rockstrom et al planetary boundaries 2009 Nature 461 472" in prompt
        for prompt in payload["citation_repair"]
    )
    assert any(
        "solid-state battery degradation" in prompt
        for prompt in payload["cross_discipline_generalization"]
    )
