from pathlib import Path

from paper_chaser_mcp.eval_canary import run_eval_canary


def test_eval_seed_datasets_validate_cleanly() -> None:
    dataset_root = Path(__file__).resolve().parent / "fixtures" / "evals"
    report = run_eval_canary(dataset_root)

    assert report["summary"]["failedItems"] == 0
    family_names = {entry["family"] for entry in report["families"]}
    assert {"planner", "synthesis", "abstention", "provenance", "runtime", "misc"}.issubset(family_names)


def test_eval_seed_datasets_have_multiple_items_per_family() -> None:
    dataset_root = Path(__file__).resolve().parent / "fixtures" / "evals"
    report = run_eval_canary(dataset_root)
    family_counts = {entry["family"]: entry["itemCount"] for entry in report["families"]}

    assert family_counts["planner"] >= 4
    assert family_counts["synthesis"] >= 4
    assert family_counts["abstention"] >= 4
    assert family_counts["provenance"] >= 4
    assert family_counts["runtime"] >= 4
    assert family_counts["misc"] >= 4
