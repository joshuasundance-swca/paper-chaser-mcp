"""Promote reviewed live traces into evaluation seed rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_chaser_mcp.eval_trace_promotion import (
    load_reviewed_trace_rows,
    promote_reviewed_traces,
    render_promoted_rows,
    write_promoted_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote reviewed live traces into evaluation dataset rows.")
    parser.add_argument("--input", required=True, help="Path to reviewed trace JSONL.")
    parser.add_argument("--output", required=True, help="Path to write promoted evaluation JSONL rows.")
    parser.add_argument("--dataset-version", default="0.1.0", help="Dataset version to stamp onto promoted rows.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = load_reviewed_trace_rows(Path(args.input))
    promoted = promote_reviewed_traces(rows, dataset_version=args.dataset_version)
    if args.output == "-":
        sys.stdout.write(render_promoted_rows(promoted))
    else:
        write_promoted_rows(Path(args.output), promoted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
