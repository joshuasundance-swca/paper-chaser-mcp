"""Run the trace-first eval workflow end to end."""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 - trusted local CLI wrapper for local repo scripts
import sys
from pathlib import Path
from typing import NamedTuple

from paper_chaser_mcp.eval_trace_promotion import (
    build_live_matrix_payload,
    load_reviewed_trace_rows,
    write_live_matrix_payload,
    write_promoted_rows,
)

MATRIX_PRESETS: dict[str, list[str]] = {
    "openai-lower-bound": [
        "openai-best|openai|gpt-5.4-mini|gpt-5.4",
        "openai-mid|openai|gpt-4.1-mini|gpt-4.1",
        "openai-small|openai|gpt-4o-mini|gpt-4o-mini",
    ],
    "anthropic-lower-bound": [
        "anthropic-best|anthropic|claude-haiku-4-5|claude-sonnet-4-6",
        "anthropic-small|anthropic|claude-haiku-4-5|claude-haiku-4-5",
    ],
    "google-lower-bound": [
        "google-best|google|gemini-2.5-flash|gemini-2.5-pro",
        "google-small|google|gemini-2.5-flash|gemini-2.5-flash",
    ],
    "nvidia-lower-bound": [
        "nvidia-best|nvidia|nvidia/nemotron-3-nano-30b-a3b|nvidia/nemotron-3-super-120b-a12b",
        "nvidia-small|nvidia|nvidia/nemotron-3-nano-30b-a3b|nvidia/nemotron-3-nano-30b-a3b",
    ],
    "cross-provider-best": [
        "openai-best|openai|gpt-5.4-mini|gpt-5.4",
        "anthropic-best|anthropic|claude-haiku-4-5|claude-sonnet-4-6",
        "google-best|google|gemini-2.5-flash|gemini-2.5-pro",
        "nvidia-best|nvidia|nvidia/nemotron-3-nano-30b-a3b|nvidia/nemotron-3-super-120b-a12b",
    ],
    "cross-provider-lower-bound": [
        "openai-best|openai|gpt-5.4-mini|gpt-5.4",
        "openai-small|openai|gpt-4o-mini|gpt-4o-mini",
        "anthropic-best|anthropic|claude-haiku-4-5|claude-sonnet-4-6",
        "anthropic-small|anthropic|claude-haiku-4-5|claude-haiku-4-5",
        "google-best|google|gemini-2.5-flash|gemini-2.5-pro",
        "google-small|google|gemini-2.5-flash|gemini-2.5-flash",
        "nvidia-best|nvidia|nvidia/nemotron-3-nano-30b-a3b|nvidia/nemotron-3-super-120b-a12b",
        "nvidia-small|nvidia|nvidia/nemotron-3-nano-30b-a3b|nvidia/nemotron-3-nano-30b-a3b",
    ],
}


class WorkflowPaths(NamedTuple):
    batch_report: Path
    capture_path: Path
    review_queue: Path
    merged_review_input: Path
    reviewed_traces: Path
    promoted_rows: Path
    dataset_root: Path
    matrix_config: Path
    matrix_results: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run expert batch capture, optional review, promotion, seed splitting, matrix evaluation, "
            "and optional matrix viewing in one workflow."
        )
    )
    parser.add_argument("--scenario-file", required=True, help="Path to expert scenario JSON file.")
    parser.add_argument(
        "--artifact-root",
        default=str(Path(__file__).resolve().parent.parent / "build" / "eval-workflow"),
        help="Directory where workflow artifacts should be written.",
    )
    parser.add_argument(
        "--dotenv-path",
        default=str(Path(__file__).resolve().parent.parent / ".env"),
        help="Path to the shared .env file used by the batch and matrix runners.",
    )
    parser.add_argument(
        "--review-mode",
        choices=["yolo", "ui", "skip"],
        default="yolo",
        help="How to handle the review step before promotion.",
    )
    parser.add_argument(
        "--merge-review-inputs",
        nargs="*",
        default=None,
        help=(
            "Optional review queue or reviewed-trace JSONL files to merge with the current run before "
            "review or promotion."
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=["openai", "nvidia"],
        help="Providers to include in the generated live matrix.",
    )
    parser.add_argument(
        "--matrix-scenario",
        action="append",
        default=None,
        help=(
            "Optional custom matrix scenario in the form name|provider or "
            "name|provider|planner_model|synthesis_model. Repeat to build a model ladder."
        ),
    )
    parser.add_argument(
        "--matrix-preset",
        action="append",
        choices=sorted(MATRIX_PRESETS),
        default=None,
        help="Optional canned matrix preset to expand into one or more model-ladder scenarios.",
    )
    parser.add_argument("--dataset-version", default="0.1.0", help="Dataset version for promoted rows.")
    parser.add_argument(
        "--latency-profile",
        default="balanced",
        help="Latency profile to stamp into the generated provider matrix.",
    )
    parser.add_argument(
        "--launch-matrix-viewer",
        action="store_true",
        help="Launch the local matrix viewer after matrix evaluation completes.",
    )
    parser.add_argument(
        "--matrix-viewer-host",
        default="127.0.0.1",
        help="Host to bind the optional matrix viewer to.",
    )
    parser.add_argument(
        "--matrix-viewer-port",
        type=int,
        default=8766,
        help="Port to bind the optional matrix viewer to.",
    )
    parser.add_argument(
        "--review-ui-host",
        default="127.0.0.1",
        help="Host to bind the review UI to when --review-mode ui is used.",
    )
    parser.add_argument(
        "--review-ui-port",
        type=int,
        default=8765,
        help="Port to bind the review UI to when --review-mode ui is used.",
    )
    return parser


def build_workflow_paths(artifact_root: Path) -> WorkflowPaths:
    return WorkflowPaths(
        batch_report=artifact_root / "expert-batch-report.json",
        capture_path=artifact_root / "captured-events.jsonl",
        review_queue=artifact_root / "review-queue.jsonl",
        merged_review_input=artifact_root / "merged-review-input.jsonl",
        reviewed_traces=artifact_root / "reviewed-traces.jsonl",
        promoted_rows=artifact_root / "dataset" / "promoted.jsonl",
        dataset_root=artifact_root / "dataset" / "seed-set",
        matrix_config=artifact_root / "dataset" / "provider-matrix.json",
        matrix_results=artifact_root / "dataset" / "matrix-results.json",
    )


def _script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def build_batch_command(args: argparse.Namespace, paths: WorkflowPaths) -> list[str]:
    return [
        sys.executable,
        _script_path("run_expert_eval_batch.py"),
        "--dotenv-path",
        args.dotenv_path,
        "--scenario-file",
        args.scenario_file,
        "--output",
        str(paths.batch_report),
        "--capture-path",
        str(paths.capture_path),
        "--review-queue-output",
        str(paths.review_queue),
    ]


def build_review_ui_command(args: argparse.Namespace, paths: WorkflowPaths) -> list[str]:
    return [
        sys.executable,
        _script_path("review_eval_traces.py"),
        "--input",
        str(review_input_path(args, paths)),
        "--output",
        str(paths.reviewed_traces),
        "--host",
        args.review_ui_host,
        "--port",
        str(args.review_ui_port),
    ]


def build_promote_command(args: argparse.Namespace, paths: WorkflowPaths) -> list[str]:
    input_path = paths.reviewed_traces if args.review_mode == "ui" else review_input_path(args, paths)
    command = [
        sys.executable,
        _script_path("promote_eval_traces.py"),
        "--input",
        str(input_path),
        "--output",
        str(paths.promoted_rows),
        "--dataset-root",
        str(paths.dataset_root),
        "--matrix-output",
        str(paths.matrix_config),
        "--dataset-version",
        args.dataset_version,
        "--latency-profile",
        args.latency_profile,
    ]
    if args.providers:
        command.extend(["--providers", *args.providers])
    if args.review_mode == "yolo":
        command.append("--yolo")
    return command


def review_input_path(args: argparse.Namespace, paths: WorkflowPaths) -> Path:
    return paths.merged_review_input if getattr(args, "merge_review_inputs", None) else paths.review_queue


def merge_review_rows(input_paths: list[Path]) -> list[dict[str, object]]:
    merged_by_key: dict[str, dict[str, object]] = {}
    ordered_keys: list[str] = []
    for path in input_paths:
        for row in load_reviewed_trace_rows(path):
            trace_id = str(row.get("trace_id") or "").strip()
            key = trace_id or json.dumps(row, sort_keys=True)
            if key not in merged_by_key:
                ordered_keys.append(key)
            merged_by_key[key] = row
    return [merged_by_key[key] for key in ordered_keys]


def resolve_matrix_scenario_specs(args: argparse.Namespace) -> list[str]:
    specs: list[str] = []
    for preset in getattr(args, "matrix_preset", None) or []:
        specs.extend(MATRIX_PRESETS[preset])
    specs.extend(getattr(args, "matrix_scenario", None) or [])
    return specs


def build_custom_matrix_payload(args: argparse.Namespace) -> dict[str, object]:
    scenarios: list[dict[str, object]] = []
    for raw_spec in resolve_matrix_scenario_specs(args):
        parts = [part.strip() for part in str(raw_spec).split("|")]
        if len(parts) not in {2, 4}:
            raise ValueError("matrix scenarios must use name|provider or name|provider|planner_model|synthesis_model")
        name, provider = parts[0], parts[1]
        if not name or not provider:
            raise ValueError("matrix scenario name and provider are required")
        scenario = build_live_matrix_payload([provider], latency_profile=args.latency_profile)["scenarios"][0]
        scenario["name"] = name
        env = dict(scenario.get("env") or {})
        if len(parts) == 4:
            planner_model, synthesis_model = parts[2], parts[3]
            if planner_model:
                env["PAPER_CHASER_PLANNER_MODEL"] = planner_model
            if synthesis_model:
                env["PAPER_CHASER_SYNTHESIS_MODEL"] = synthesis_model
        scenario["env"] = env
        scenarios.append(scenario)
    return {"scenarios": scenarios}


def build_matrix_command(args: argparse.Namespace, paths: WorkflowPaths) -> list[str]:
    return [
        sys.executable,
        _script_path("run_eval_canaries.py"),
        "--dataset-root",
        str(paths.dataset_root),
        "--matrix-config",
        str(paths.matrix_config),
        "--dotenv-path",
        args.dotenv_path,
        "--live",
        "--output",
        str(paths.matrix_results),
        "--summary",
    ]


def build_matrix_viewer_command(args: argparse.Namespace, paths: WorkflowPaths) -> list[str]:
    return [
        sys.executable,
        _script_path("view_eval_matrix.py"),
        "--input",
        str(paths.matrix_results),
        "--host",
        args.matrix_viewer_host,
        "--port",
        str(args.matrix_viewer_port),
    ]


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)  # nosec B603 - fixed local script entrypoints only


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths = build_workflow_paths(artifact_root)

    run_command(build_batch_command(args, paths))

    if args.merge_review_inputs:
        merged_rows = merge_review_rows([paths.review_queue, *[Path(path) for path in args.merge_review_inputs]])
        write_promoted_rows(paths.merged_review_input, merged_rows)

    if args.review_mode == "ui":
        print(
            ("Opening review UI. Save your reviewed rows, then stop the server with Ctrl+C to continue the workflow.")
        )
        run_command(build_review_ui_command(args, paths))

    run_command(build_promote_command(args, paths))

    if resolve_matrix_scenario_specs(args):
        write_live_matrix_payload(paths.matrix_config, build_custom_matrix_payload(args))

    run_command(build_matrix_command(args, paths))

    print("Workflow artifacts")
    print(f"batch report: {paths.batch_report}")
    print(f"review queue: {paths.review_queue}")
    if args.merge_review_inputs:
        print(f"merged review input: {paths.merged_review_input}")
    if args.review_mode == "ui":
        print(f"reviewed traces: {paths.reviewed_traces}")
    print(f"promoted rows: {paths.promoted_rows}")
    print(f"dataset root: {paths.dataset_root}")
    print(f"matrix config: {paths.matrix_config}")
    print(f"matrix results: {paths.matrix_results}")

    if args.launch_matrix_viewer:
        print("Opening matrix viewer. Stop the server with Ctrl+C when done.")
        run_command(build_matrix_viewer_command(args, paths))
    else:
        viewer_command = " ".join(build_matrix_viewer_command(args, paths))
        print(f"matrix viewer command: {viewer_command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
