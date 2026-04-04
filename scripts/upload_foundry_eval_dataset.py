"""Upload exported eval artifacts into a Microsoft Foundry project dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from paper_chaser_mcp.eval_publish import MissingOptionalDependencyError, upload_foundry_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload eval artifacts to a Microsoft Foundry project dataset.")
    parser.add_argument("--input", required=True, help="Path to the exported eval JSONL file or folder.")
    parser.add_argument(
        "--project-endpoint",
        default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT"),
        help="Foundry project endpoint. Defaults to AZURE_AI_PROJECT_ENDPOINT.",
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset name to create or update in Foundry.")
    parser.add_argument("--dataset-version", required=True, help="Dataset version to register in Foundry.")
    parser.add_argument("--connection-name", default=None, help="Optional Foundry storage connection override.")
    parser.add_argument(
        "--file-pattern",
        default=None,
        help="Optional regex pattern used only when uploading a folder through datasets.upload_folder().",
    )
    parser.add_argument(
        "--expected-format",
        choices=["foundry-eval"],
        default="foundry-eval",
        help="Validate the local file against the expected export shape before upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize locally without calling Foundry.",
    )
    parser.add_argument("--output", default="-", help="Path to write a JSON summary, or '-' for stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.project_endpoint:
        parser.error("--project-endpoint is required unless AZURE_AI_PROJECT_ENDPOINT is set")

    try:
        result = upload_foundry_dataset(
            source=Path(args.input),
            project_endpoint=args.project_endpoint,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            connection_name=args.connection_name,
            file_pattern=args.file_pattern,
            expected_format=args.expected_format,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, MissingOptionalDependencyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = json.dumps(result, indent=2)
    if args.output == "-":
        print(payload)
    else:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
