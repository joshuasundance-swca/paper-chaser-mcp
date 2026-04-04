"""Execute expert-tool scenarios and optionally materialize a review queue from captured events."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from paper_chaser_mcp.eval_curation import (
    build_batch_ledger_rows,
    build_batch_summary,
    build_review_queue_rows,
    load_captured_eval_events,
    render_review_queue,
    write_batch_ledger_csv,
    write_review_queue,
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


def _resolve_placeholders(value: Any, prior_results: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, str) and value.startswith("$result."):
        _, scenario_name, field_name = value.split(".", 2)
        return (prior_results.get(scenario_name) or {}).get(field_name)
    if isinstance(value, dict):
        return {key: _resolve_placeholders(child, prior_results) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(child, prior_results) for child in value]
    return value


async def _run_batch(args: argparse.Namespace) -> dict[str, Any]:
    from paper_chaser_mcp.server import _execute_tool

    scenarios_payload = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))
    scenarios = scenarios_payload.get("scenarios") or []
    prior_results: dict[str, dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    for scenario in scenarios:
        name = str(scenario.get("name") or "").strip()
        tool = str(scenario.get("tool") or "").strip()
        arguments = _resolve_placeholders(scenario.get("arguments") or {}, prior_results)
        result = await _execute_tool(tool, arguments, ctx=None)
        prior_results[name] = result if isinstance(result, dict) else {"value": result}
        runs.append({"name": name, "tool": tool, "arguments": arguments, "result": result})
    return {"runs": runs}


def _derive_batch_id(scenario_file: str) -> str:
    digest = hashlib.sha256(scenario_file.encode("utf-8")).hexdigest()[:8]
    date_prefix = time.strftime("%Y%m%d")
    return f"batch_{date_prefix}_{digest}"


def _derive_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def _default_artifact_path(output_path: str, artifact_name: str) -> Path | None:
    if output_path == "-":
        return None
    return Path(output_path).resolve().parent / artifact_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run expert-tool scenarios and optionally build a review queue from captured events."
    )
    parser.add_argument("--scenario-file", required=True, help="Path to expert scenario JSON file.")
    parser.add_argument("--output", required=True, help="Path to write the batch report JSON.")
    parser.add_argument(
        "--dotenv-path", default=str(Path(__file__).resolve().parent.parent / ".env"), help="Path to .env file."
    )
    parser.add_argument("--capture-path", default=None, help="Path to captured eval event JSONL file.")
    parser.add_argument(
        "--review-queue-output", default=None, help="Optional review queue output path built from captured events."
    )
    parser.add_argument(
        "--batch-summary-output",
        default=None,
        help="Optional batch summary JSON path. Defaults next to --output when writing to a file.",
    )
    parser.add_argument(
        "--batch-ledger-output",
        default=None,
        help="Optional batch ledger CSV path. Defaults next to --output when writing to a file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _load_dotenv(Path(args.dotenv_path))
    batch_id = _derive_batch_id(args.scenario_file)
    run_id = _derive_run_id()
    os.environ["PAPER_CHASER_EVAL_BATCH_ID"] = batch_id
    os.environ["PAPER_CHASER_EVAL_RUN_ID"] = run_id
    if args.capture_path:
        os.environ["PAPER_CHASER_ENABLE_EVAL_TRACE_CAPTURE"] = "true"
        os.environ["PAPER_CHASER_EVAL_TRACE_PATH"] = args.capture_path
    report = asyncio.run(_run_batch(args))
    generated_at = int(time.time() * 1000)
    report = {
        "batchId": batch_id,
        "runId": run_id,
        "generatedAt": generated_at,
        "scenarioFile": args.scenario_file,
        **report,
    }
    if args.output == "-":
        sys.stdout.write(json.dumps(report, indent=2))
        sys.stdout.write("\n")
    else:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_output = (
        Path(args.batch_summary_output)
        if args.batch_summary_output
        else _default_artifact_path(args.output, "batch-summary.json")
    )
    ledger_output = (
        Path(args.batch_ledger_output)
        if args.batch_ledger_output
        else _default_artifact_path(args.output, "batch-ledger.csv")
    )
    if args.review_queue_output and args.capture_path:
        events = load_captured_eval_events(Path(args.capture_path))
        queue = build_review_queue_rows(events)
        if args.review_queue_output == "-":
            sys.stdout.write(render_review_queue(queue))
        else:
            write_review_queue(Path(args.review_queue_output), queue)
        summary = build_batch_summary(
            report,
            events,
            queue,
            batch_id=batch_id,
            run_id=run_id,
            scenario_file=args.scenario_file,
        )
        if summary_output is not None:
            summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if ledger_output is not None:
            ledger_rows = build_batch_ledger_rows(report, events, queue)
            write_batch_ledger_csv(ledger_output, ledger_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
