"""Run deterministic validation over role-based LLM evaluation seed datasets."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess  # nosec B404 - trusted local CLI wrapper for spawned self-runs
import sys
import tempfile
from pathlib import Path

from paper_chaser_mcp.eval_canary import (
    TASK_FAMILIES,
    render_canary_report,
    run_eval_canary,
    run_live_eval_canary,
)


def _load_dotenv(dotenv_path: Path) -> int:
    loaded = 0
    if not dotenv_path.exists():
        return loaded

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
        loaded += 1
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Validate role-based evaluation seed datasets and emit a structured offline canary report.")
    )
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "evals"),
        help="Directory containing *.jsonl evaluation seed datasets.",
    )
    parser.add_argument(
        "--family",
        choices=["all", *sorted(TASK_FAMILIES)],
        default="all",
        help="Limit validation to one task family.",
    )
    parser.add_argument(
        "--item-id",
        default=None,
        help="Limit validation to one specific evaluation item id.",
    )
    parser.add_argument(
        "--matrix-config",
        default=None,
        help="Path to a JSON matrix config describing multiple live runtime scenarios to execute.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Path to write JSON output, or '-' for stdout.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact human-readable summary to stderr.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return a non-zero exit code on warnings as well as failures.",
    )
    parser.add_argument(
        "--dotenv-path",
        default=str(Path(__file__).resolve().parent.parent / ".env"),
        help="Path to a .env file to load before live evaluation. Existing environment variables are preserved.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Run live validation against the currently configured Paper Chaser runtime for supported families. "
            "This evaluates the under-the-hood runtime behavior, not an external MCP client model."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.matrix_config:
        if not args.live:
            parser.error("--matrix-config requires --live")
        return _run_matrix(parser, args)

    dataset_root = Path(args.dataset_root)
    family_filter = None if args.family == "all" else args.family
    if args.live:
        dotenv_loaded = _load_dotenv(Path(args.dotenv_path))
        report = asyncio.run(
            run_live_eval_canary(dataset_root, family_filter=family_filter, item_id_filter=args.item_id)
        )
        report.setdefault("liveSummary", {})["dotenvPath"] = str(Path(args.dotenv_path))
        report.setdefault("liveSummary", {})["dotenvLoadedKeys"] = dotenv_loaded
    else:
        report = run_eval_canary(dataset_root, family_filter=family_filter, item_id_filter=args.item_id)

    if args.summary:
        print(render_canary_report(report), file=sys.stderr)

    output_text = json.dumps(report, indent=2)
    if args.output == "-":
        print(output_text)
    else:
        Path(args.output).write_text(output_text, encoding="utf-8")

    failed = report["summary"]["failedItems"]
    warnings = report["summary"]["warningItems"]
    live_failed = report.get("liveSummary", {}).get("failed", 0)
    if failed:
        return 1
    if live_failed:
        return 1
    if args.fail_on_warning and warnings:
        return 1
    return 0


def _run_matrix(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    matrix_path = Path(args.matrix_config)
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    scenarios = matrix_payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        parser.error("matrix config must contain a non-empty 'scenarios' list")

    matrix_results: list[dict[str, object]] = []
    exit_code = 0
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            parser.error("each matrix scenario must be an object")
        scenario_name = str(scenario.get("name") or "").strip()
        env_overrides = scenario.get("env") or {}
        if not scenario_name:
            parser.error("each matrix scenario requires a name")
        if not isinstance(env_overrides, dict):
            parser.error(f"matrix scenario {scenario_name!r} has non-object env overrides")

        child_env = os.environ.copy()
        child_env.update({str(key): str(value) for key, value in env_overrides.items()})
        scenario_family = str(scenario.get("family") or args.family)
        scenario_item_id = scenario.get("itemId") or args.item_id

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as handle:
            output_path = Path(handle.name)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset-root",
            args.dataset_root,
            "--family",
            scenario_family,
            "--output",
            str(output_path),
            "--dotenv-path",
            args.dotenv_path,
            "--live",
        ]
        if scenario_item_id:
            command.extend(["--item-id", str(scenario_item_id)])

        result = subprocess.run(  # nosec B603 - command is built from fixed args and self path
            command,
            env=child_env,
            capture_output=True,
            text=True,
        )
        scenario_report: dict[str, object]
        if output_path.exists():
            scenario_report = json.loads(output_path.read_text(encoding="utf-8"))
            output_path.unlink(missing_ok=True)
        else:
            scenario_report = {
                "summary": {"failedItems": 1, "warningItems": 0},
                "liveSummary": {"failed": 1},
                "error": result.stderr or result.stdout,
            }

        scenario_result = {
            "name": scenario_name,
            "env": env_overrides,
            "family": scenario_family,
            "itemId": scenario_item_id,
            "exitCode": result.returncode,
            "report": scenario_report,
        }
        matrix_results.append(scenario_result)
        if result.returncode != 0:
            exit_code = 1

    matrix_summary = {
        "scenarioCount": len(matrix_results),
        "failedScenarios": sum(1 for item in matrix_results if item["exitCode"] != 0),
        "succeededScenarios": sum(1 for item in matrix_results if item["exitCode"] == 0),
    }
    payload = {"matrixSummary": matrix_summary, "scenarios": matrix_results}
    output_text = json.dumps(payload, indent=2)
    if args.output == "-":
        print(output_text)
    else:
        Path(args.output).write_text(output_text, encoding="utf-8")
    if args.summary:
        print(
            (
                "Eval Matrix Summary\n"
                f"scenarios: {matrix_summary['scenarioCount']}\n"
                f"passed: {matrix_summary['succeededScenarios']}\n"
                f"failed: {matrix_summary['failedScenarios']}"
            ),
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
