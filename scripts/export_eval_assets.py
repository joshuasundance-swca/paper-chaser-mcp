"""Export promoted eval rows or reviewed traces into portable Foundry/HF/training formats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_chaser_mcp.eval_exports import (
    export_foundry_eval_rows,
    export_hf_dataset_rows,
    export_training_chat_rows,
    load_jsonl_rows,
    render_jsonl,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export eval assets into portable Foundry, Hugging Face, or training formats."
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL rows.")
    parser.add_argument("--output", required=True, help="Path to write exported JSONL rows.")
    parser.add_argument(
        "--format",
        required=True,
        choices=["foundry-eval", "hf-dataset", "training-chat"],
        help="Export format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = load_jsonl_rows(Path(args.input))
    if args.format == "foundry-eval":
        exported = export_foundry_eval_rows(rows)
    elif args.format == "hf-dataset":
        exported = export_hf_dataset_rows(rows)
    else:
        exported = export_training_chat_rows(rows)
    if args.output == "-":
        sys.stdout.write(render_jsonl(exported))
    else:
        write_jsonl(Path(args.output), exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
