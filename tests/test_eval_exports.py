from pathlib import Path

from paper_chaser_mcp.eval_exports import (
    export_foundry_eval_rows,
    export_hf_dataset_rows,
    export_training_chat_rows,
    load_jsonl_rows,
)


def test_eval_exports_build_foundry_and_hf_rows() -> None:
    sample_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "trace-promotion.sample.jsonl"
    rows = load_jsonl_rows(sample_path)

    foundry_rows = export_foundry_eval_rows(rows)
    hf_rows = export_hf_dataset_rows(rows)

    assert len(foundry_rows) == 2
    assert foundry_rows[0]["metadata"]["review_labels"]["trainingEligibility"] == "approved"
    assert len(hf_rows) == 2
    assert hf_rows[1]["review_labels"]["trainingObjective"] == "grounded_synthesis"


def test_eval_exports_build_training_chat_rows() -> None:
    sample_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "trace-promotion.sample.jsonl"
    rows = load_jsonl_rows(sample_path)
    training_rows = export_training_chat_rows(rows)

    assert len(training_rows) == 2
    assert training_rows[0]["messages"][0]["role"] == "user"
    assert training_rows[0]["messages"][1]["role"] == "assistant"
