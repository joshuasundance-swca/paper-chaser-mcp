from pathlib import Path

from paper_chaser_mcp.eval_trace_promotion import load_reviewed_trace_rows, promote_reviewed_traces


def test_trace_promotion_sample_promotes_rows() -> None:
    sample_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "trace-promotion.sample.jsonl"
    rows = load_reviewed_trace_rows(sample_path)
    promoted = promote_reviewed_traces(rows, dataset_version="0.1.0")

    assert len(promoted) == 2
    assert promoted[0]["meta"]["origin"] == "trace_mined"
    assert promoted[0]["meta"]["task_family"] == "planner"
    assert promoted[1]["meta"]["task_family"] == "synthesis"
    assert promoted[0]["review_labels"]["trainingEligibility"] == "approved"
