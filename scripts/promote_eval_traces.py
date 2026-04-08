"""Promote reviewed live traces into evaluation seed rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_chaser_mcp.eval_trace_promotion import (
    apply_yolo_review_defaults,
    build_live_matrix_payload,
    load_reviewed_trace_rows,
    promote_reviewed_traces,
    render_promoted_rows,
    write_live_matrix_payload,
    write_promoted_rows,
    write_promoted_rows_by_family,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote reviewed live traces into evaluation dataset rows.")
    parser.add_argument("--input", required=True, help="Path to reviewed trace JSONL.")
    parser.add_argument("--output", required=True, help="Path to write promoted evaluation JSONL rows.")
    parser.add_argument("--dataset-version", default="0.1.0", help="Dataset version to stamp onto promoted rows.")
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Auto-approve all trace rows with inferred defaults before promotion.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional directory to also write family-specific *.seed.jsonl files.",
    )
    parser.add_argument(
        "--matrix-output",
        default=None,
        help="Optional path to also write a live provider matrix JSON file.",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Providers to include in the optional live matrix output, e.g. openai nvidia azure-openai.",
    )
    parser.add_argument(
        "--latency-profile",
        default="balanced",
        help="Latency profile to stamp into the optional live provider matrix.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = load_reviewed_trace_rows(Path(args.input))
    if args.yolo:
        rows = apply_yolo_review_defaults(rows)
    promoted = promote_reviewed_traces(rows, dataset_version=args.dataset_version)
    if args.output == "-":
        sys.stdout.write(render_promoted_rows(promoted))
    else:
        write_promoted_rows(Path(args.output), promoted)
    if args.dataset_root:
        write_promoted_rows_by_family(Path(args.dataset_root), promoted)
    if args.matrix_output:
        payload = build_live_matrix_payload(
            args.providers or ["openai", "nvidia"],
            latency_profile=args.latency_profile,
        )
        write_live_matrix_payload(Path(args.matrix_output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
