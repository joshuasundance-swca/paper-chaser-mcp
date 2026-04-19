"""Build a review queue from captured live eval events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_chaser_mcp.eval_curation import (
    build_review_queue_rows,
    load_captured_eval_events,
    render_review_queue,
    write_review_queue,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a review queue from captured live eval events.")
    parser.add_argument("--input", required=True, help="Path to captured eval event JSONL.")
    parser.add_argument("--output", required=True, help="Path to write review-queue JSONL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    events = load_captured_eval_events(Path(args.input))
    queue = build_review_queue_rows(events)
    if args.output == "-":
        sys.stdout.write(render_review_queue(queue))
    else:
        write_review_queue(Path(args.output), queue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
