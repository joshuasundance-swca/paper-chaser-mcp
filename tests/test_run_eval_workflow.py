import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eval_workflow.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_eval_workflow_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_workflow_paths_uses_expected_layout(tmp_path: Path) -> None:
    module = _load_module()

    paths = module.build_workflow_paths(tmp_path)

    assert paths.batch_report == tmp_path / "expert-batch-report.json"
    assert paths.review_queue == tmp_path / "review-queue.jsonl"
    assert paths.merged_review_input == tmp_path / "merged-review-input.jsonl"
    assert paths.promoted_rows == tmp_path / "dataset" / "promoted.jsonl"
    assert paths.matrix_results == tmp_path / "dataset" / "matrix-results.json"


def test_build_promote_command_uses_yolo_for_non_ui_review_mode(tmp_path: Path) -> None:
    module = _load_module()
    args = argparse.Namespace(
        review_mode="yolo",
        providers=["openai", "nvidia", "anthropic"],
        dataset_version="0.1.0",
        latency_profile="balanced",
    )
    paths = module.build_workflow_paths(tmp_path)

    command = module.build_promote_command(args, paths)

    assert str(paths.review_queue) in command
    assert "--yolo" in command
    assert command[0] == module.sys.executable
    provider_index = command.index("--providers") + 1
    assert command[provider_index : provider_index + 3] == [
        "openai",
        "nvidia",
        "anthropic",
    ]


def test_build_review_ui_and_viewer_commands_use_same_python(tmp_path: Path) -> None:
    module = _load_module()
    args = argparse.Namespace(
        merge_review_inputs=None,
        review_ui_host="127.0.0.1",
        review_ui_port=8765,
        matrix_viewer_host="127.0.0.1",
        matrix_viewer_port=8766,
    )
    paths = module.build_workflow_paths(tmp_path)

    review_command = module.build_review_ui_command(args, paths)
    viewer_command = module.build_matrix_viewer_command(args, paths)

    assert review_command[0] == module.sys.executable
    assert viewer_command[0] == module.sys.executable
    assert str(paths.review_queue) in review_command
    assert str(paths.matrix_results) in viewer_command


def test_merge_review_rows_prefers_latest_duplicate_trace(tmp_path: Path) -> None:
    module = _load_module()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        '{"trace_id": "trace_a", "review": {"promote": false}, "trace": {"source_tool": "research"}}\n',
        encoding="utf-8",
    )
    second.write_text(
        (
            '{"trace_id": "trace_a", "review": {"promote": true}, "trace": {"source_tool": "research"}}\n'
            '{"trace_id": "trace_b", "review": {"promote": true}, "trace": {"source_tool": "ask_result_set"}}\n'
        ),
        encoding="utf-8",
    )

    merged = module.merge_review_rows([first, second])

    assert [row["trace_id"] for row in merged] == ["trace_a", "trace_b"]
    assert merged[0]["review"]["promote"] is True


def test_build_custom_matrix_payload_supports_model_ladder() -> None:
    module = _load_module()
    args = argparse.Namespace(
        latency_profile="balanced",
        matrix_preset=None,
        matrix_scenario=[
            "openai-best|openai|gpt-5.4-mini|gpt-5.4",
            "openai-small|openai|gpt-4.1-mini|gpt-4.1",
        ],
    )

    payload = module.build_custom_matrix_payload(args)

    scenarios = payload["scenarios"]
    assert scenarios[0]["name"] == "openai-best"
    assert scenarios[1]["env"]["PAPER_CHASER_PLANNER_MODEL"] == "gpt-4.1-mini"
    assert scenarios[1]["env"]["PAPER_CHASER_SYNTHESIS_MODEL"] == "gpt-4.1"


def test_resolve_matrix_scenario_specs_combines_presets_and_explicit_values() -> None:
    module = _load_module()
    args = argparse.Namespace(
        matrix_preset=["openai-lower-bound"],
        matrix_scenario=["custom|openai|gpt-4.1-mini|gpt-4.1"],
    )

    specs = module.resolve_matrix_scenario_specs(args)

    assert specs[0].startswith("openai-best|openai|")
    assert specs[-1] == "custom|openai|gpt-4.1-mini|gpt-4.1"


def test_resolve_matrix_scenario_specs_supports_multiple_presets() -> None:
    module = _load_module()
    args = argparse.Namespace(
        matrix_preset=["openai-lower-bound", "google-lower-bound"],
        matrix_scenario=None,
    )

    specs = module.resolve_matrix_scenario_specs(args)

    assert any(spec.startswith("openai-best|") for spec in specs)
    assert any(spec.startswith("google-best|") for spec in specs)


def test_cross_provider_lower_bound_preset_contains_multiple_providers_and_rungs() -> None:
    module = _load_module()
    args = argparse.Namespace(
        matrix_preset=["cross-provider-lower-bound"],
        matrix_scenario=None,
    )

    specs = module.resolve_matrix_scenario_specs(args)

    assert any(spec.startswith("openai-best|") for spec in specs)
    assert any(spec.startswith("openai-small|") for spec in specs)
    assert any(spec.startswith("anthropic-best|") for spec in specs)
    assert any(spec.startswith("google-small|") for spec in specs)
    assert any(spec.startswith("nvidia-best|") for spec in specs)
