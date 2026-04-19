"""Upload exported eval artifacts to Hugging Face dataset repos or buckets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from paper_chaser_mcp.eval_publish import MissingOptionalDependencyError, upload_hf_bucket, upload_hf_dataset_repo


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Path to the exported eval artifact file or folder.")
    parser.add_argument(
        "--expected-format",
        choices=["foundry-eval", "hf-dataset", "training-chat"],
        default=None,
        help="Optional local validation against one of the repo export formats before upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize locally without calling the Hub.",
    )
    parser.add_argument("--output", default="-", help="Path to write a JSON summary, or '-' for stdout.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload eval artifacts to Hugging Face dataset repos or buckets.")
    subparsers = parser.add_subparsers(dest="target", required=True)

    repo_parser = subparsers.add_parser("dataset-repo", help="Upload to a Hugging Face dataset repository.")
    _add_common_arguments(repo_parser)
    repo_parser.add_argument("--repo-id", required=True, help="Dataset repo id in owner/name form.")
    repo_parser.add_argument("--path-in-repo", default=None, help="Destination path in the dataset repo.")
    repo_parser.add_argument("--private", action="store_true", help="Create the dataset repo as private if missing.")
    repo_parser.add_argument("--revision", default=None, help="Optional branch or revision to upload against.")
    repo_parser.add_argument("--create-pr", action="store_true", help="Create a PR instead of committing directly.")
    repo_parser.add_argument("--commit-message", default=None, help="Optional commit message override.")
    repo_parser.add_argument("--commit-description", default=None, help="Optional commit description override.")
    repo_parser.add_argument(
        "--delete-pattern",
        action="append",
        default=None,
        help="Remote delete pattern used only with directory uploads via upload_folder().",
    )

    bucket_parser = subparsers.add_parser("bucket", help="Upload to a Hugging Face bucket.")
    _add_common_arguments(bucket_parser)
    bucket_parser.add_argument("--bucket-id", required=True, help="Bucket id in owner/name form.")
    bucket_parser.add_argument(
        "--remote-path",
        default=None,
        help="Destination file path or directory prefix inside the bucket.",
    )
    bucket_parser.add_argument("--private", action="store_true", help="Create the bucket as private if missing.")
    bucket_parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete unmatched remote files when syncing a folder.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    try:
        if args.target == "dataset-repo":
            result = upload_hf_dataset_repo(
                source=Path(args.input),
                repo_id=args.repo_id,
                path_in_repo=args.path_in_repo,
                token=token,
                private=True if args.private else None,
                revision=args.revision,
                create_pr=args.create_pr,
                commit_message=args.commit_message,
                commit_description=args.commit_description,
                delete_patterns=args.delete_pattern,
                expected_format=args.expected_format,
                dry_run=args.dry_run,
            )
        else:
            result = upload_hf_bucket(
                source=Path(args.input),
                bucket_id=args.bucket_id,
                remote_path=args.remote_path,
                token=token,
                private=True if args.private else None,
                delete=args.delete,
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
