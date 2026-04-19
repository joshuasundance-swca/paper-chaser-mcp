import json
from pathlib import Path

from paper_chaser_mcp.eval_trace_promotion import (
    apply_yolo_review_defaults,
    build_live_matrix_payload,
    group_promoted_rows_by_family,
    load_reviewed_trace_rows,
    promote_reviewed_traces,
    write_live_matrix_payload,
    write_promoted_rows,
    write_promoted_rows_by_family,
)


def test_trace_promotion_sample_promotes_rows() -> None:
    sample_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "trace-promotion.sample.jsonl"
    rows = load_reviewed_trace_rows(sample_path)
    promoted = promote_reviewed_traces(rows, dataset_version="0.1.0")

    assert len(promoted) == 2
    assert promoted[0]["meta"]["origin"] == "trace_mined"
    assert promoted[0]["meta"]["task_family"] == "planner"
    assert promoted[1]["meta"]["task_family"] == "synthesis"
    assert promoted[0]["review_labels"]["trainingEligibility"] == "approved"


def test_yolo_review_defaults_auto_approve_and_infer_live_eval() -> None:
    review_queue_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "trace-promotion.sample.jsonl"

    queue_rows = load_reviewed_trace_rows(review_queue_path)
    queue_rows[1]["review"]["promote"] = False
    queue_rows[1]["review"]["expected"] = {}
    queue_rows[1]["live_eval"] = {}
    queue_rows[0]["trace"]["search_session_id"] = "ssn_sample_planner"
    queue_rows[0]["trace"]["query"] = "Research PFAS remediation in groundwater"
    queue_rows[1]["trace"]["search_session_id"] = "ssn_sample_planner"

    yolo_rows = apply_yolo_review_defaults(queue_rows)

    assert yolo_rows[1]["review"]["promote"] is True
    assert yolo_rows[1]["review"]["labels"]["trainingEligibility"] == "approved"
    assert yolo_rows[1]["live_eval"]["research_query"] == "Research PFAS remediation in groundwater"
    assert yolo_rows[1]["review"]["expected"]["expected_answer_status"] == "answered"


def test_write_promoted_rows_by_family_and_matrix_payload(tmp_path: Path) -> None:
    sample_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "trace-promotion.sample.jsonl"
    rows = load_reviewed_trace_rows(sample_path)
    promoted = promote_reviewed_traces(rows, dataset_version="0.1.0")

    written = write_promoted_rows_by_family(tmp_path, promoted)
    written_names = sorted(path.name for path in written)
    assert written_names == ["planner.seed.jsonl", "synthesis.seed.jsonl"]

    grouped = group_promoted_rows_by_family(promoted)
    assert sorted(grouped.keys()) == ["planner", "synthesis"]

    matrix_payload = build_live_matrix_payload(["openai", "nvidia", "azure-openai"])
    assert [scenario["name"] for scenario in matrix_payload["scenarios"]] == [
        "openai-live",
        "nvidia-live",
        "azure-openai-live",
    ]

    matrix_path = tmp_path / "provider-matrix.json"
    write_live_matrix_payload(matrix_path, matrix_payload)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert len(payload["scenarios"]) == 3


def test_write_promoted_rows_creates_parent_directories(tmp_path: Path) -> None:
    sample_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "trace-promotion.sample.jsonl"
    rows = load_reviewed_trace_rows(sample_path)
    promoted = promote_reviewed_traces(rows, dataset_version="0.1.0")

    output_path = tmp_path / "nested" / "eval-dataset" / "promoted.jsonl"
    write_promoted_rows(output_path, promoted)

    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8")
    assert "trace_promoted_planner_pfas_001" in contents
