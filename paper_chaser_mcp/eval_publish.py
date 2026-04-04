"""Helpers for validating and publishing eval artifacts to external platforms."""

from __future__ import annotations

import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Sequence

from . import __version__
from .eval_exports import load_jsonl_rows


class MissingOptionalDependencyError(RuntimeError):
    """Raised when an optional SDK required by a publish helper is missing."""


EXPORT_FORMATS = ("foundry-eval", "hf-dataset", "training-chat")


def validate_export_rows(rows: list[dict[str, Any]], expected_format: str) -> dict[str, Any]:
    if expected_format not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {expected_format}")
    if not rows:
        return {
            "format": expected_format,
            "rowCount": 0,
            "taskFamilies": [],
            "sampleKeys": [],
        }

    sample_keys = sorted(rows[0].keys())
    task_families = sorted(
        {str(row.get("task_family") or "").strip() for row in rows if str(row.get("task_family") or "").strip()}
    )

    required_fields = {
        "foundry-eval": ("id", "task_family", "input", "expected", "metadata"),
        "hf-dataset": (
            "id",
            "task_family",
            "input",
            "expected",
            "evaluation_target",
            "tags",
            "review_labels",
            "lineage",
        ),
        "training-chat": ("messages", "metadata"),
    }[expected_format]

    for index, row in enumerate(rows, start=1):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(f"Row {index} is missing required fields for {expected_format}: {', '.join(missing)}")
        if expected_format == "training-chat":
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Row {index} must contain at least two chat messages")
            roles = [str(message.get("role") or "") for message in messages[:2] if isinstance(message, dict)]
            if roles != ["user", "assistant"]:
                raise ValueError(f"Row {index} must start with user then assistant messages")
        else:
            if not isinstance(row.get("input"), dict):
                raise ValueError(f"Row {index} has a non-object input field")
            if not isinstance(row.get("expected"), dict):
                raise ValueError(f"Row {index} has a non-object expected field")

    return {
        "format": expected_format,
        "rowCount": len(rows),
        "taskFamilies": task_families,
        "sampleKeys": sample_keys,
    }


def summarize_export_source(source: Path, *, expected_format: str | None = None) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(f"Eval artifact source does not exist: {source}")

    summary: dict[str, Any] = {
        "path": str(source),
        "pathKind": "directory" if source.is_dir() else "file",
    }
    if source.is_file():
        summary["sizeBytes"] = source.stat().st_size
        if source.suffix.lower() == ".jsonl":
            rows = load_jsonl_rows(source)
            summary["rowCount"] = len(rows)
            summary["sampleKeys"] = sorted(rows[0].keys()) if rows else []
            summary["taskFamilies"] = sorted(
                {str(row.get("task_family") or "").strip() for row in rows if str(row.get("task_family") or "").strip()}
            )
            if expected_format:
                summary["validation"] = validate_export_rows(rows, expected_format)
        return summary

    summary["fileCount"] = sum(1 for child in source.rglob("*") if child.is_file())
    return summary


def _create_default_credential() -> Any:
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:  # pragma: no cover - exercised via helper patching in tests
        raise MissingOptionalDependencyError(
            "Azure eval publishing requires the eval-foundry extra. "
            'Install it with `pip install -e ".[eval-foundry]"` or `pip install azure-identity azure-ai-projects`.'
        ) from exc
    return DefaultAzureCredential()


def _create_ai_project_client(endpoint: str, credential: Any) -> Any:
    try:
        from azure.ai.projects import AIProjectClient  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised via helper patching in tests
        raise MissingOptionalDependencyError(
            "Azure eval publishing requires the eval-foundry extra. "
            'Install it with `pip install -e ".[eval-foundry]"` or `pip install azure-ai-projects azure-identity`.'
        ) from exc
    return AIProjectClient(endpoint=endpoint, credential=credential)


def _create_hf_api(token: str | None = None) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised via helper patching in tests
        raise MissingOptionalDependencyError(
            "Hugging Face eval publishing requires the eval-huggingface extra. "
            'Install it with `pip install -e ".[eval-huggingface]"` or `pip install huggingface_hub`.'
        ) from exc
    return HfApi(token=token, library_name="paper-chaser-mcp", library_version=__version__)


def _enter_if_context(stack: ExitStack, value: Any) -> Any:
    if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
        return stack.enter_context(value)
    return value


def _compile_file_pattern(pattern: str | None) -> re.Pattern[str] | None:
    if not pattern:
        return None
    return re.compile(pattern)


def _extract_fields(value: Any, *fields: str) -> dict[str, Any]:
    return {field: getattr(value, field, None) for field in fields}


def _normalize_repo_path(source: Path, path_in_repo: str | None) -> str:
    if path_in_repo:
        return path_in_repo.replace("\\", "/")
    return source.name


def _normalize_bucket_path(source: Path, remote_path: str | None) -> str:
    if not remote_path:
        return source.name
    normalized = remote_path.replace("\\", "/").strip("/")
    if source.is_file() and remote_path.endswith(("/", "\\")):
        return f"{normalized}/{source.name}"
    return normalized


def _bucket_handle(bucket_id: str, remote_path: str | None = None) -> str:
    normalized = (remote_path or "").replace("\\", "/").strip("/")
    if normalized:
        return f"hf://buckets/{bucket_id}/{normalized}"
    return f"hf://buckets/{bucket_id}"


def upload_foundry_dataset(
    *,
    source: Path,
    project_endpoint: str,
    dataset_name: str,
    dataset_version: str,
    connection_name: str | None = None,
    file_pattern: str | None = None,
    expected_format: str | None = "foundry-eval",
    dry_run: bool = False,
    credential_factory: Callable[[], Any] | None = None,
    client_factory: Callable[[str, Any], Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_export_source(source, expected_format=expected_format if source.is_file() else None)
    result: dict[str, Any] = {
        "target": "azure-ai-foundry",
        "datasetName": dataset_name,
        "datasetVersion": dataset_version,
        "projectEndpoint": project_endpoint,
        "source": summary,
        "dryRun": dry_run,
    }
    if dry_run:
        result["dryRunScope"] = "local-only"
        return result

    credential_factory = credential_factory or _create_default_credential
    client_factory = client_factory or _create_ai_project_client
    with ExitStack() as stack:
        credential = _enter_if_context(stack, credential_factory())
        client = _enter_if_context(stack, client_factory(project_endpoint, credential))
        if source.is_dir():
            dataset = client.datasets.upload_folder(
                name=dataset_name,
                version=dataset_version,
                folder=str(source),
                connection_name=connection_name,
                file_pattern=_compile_file_pattern(file_pattern),
            )
            result["uploadedKind"] = "folder"
        else:
            dataset = client.datasets.upload_file(
                name=dataset_name,
                version=dataset_version,
                file_path=str(source),
                connection_name=connection_name,
            )
            result["uploadedKind"] = "file"

    result["dataset"] = _extract_fields(dataset, "id", "name", "version", "path", "data_uri")
    return result


def upload_hf_dataset_repo(
    *,
    source: Path,
    repo_id: str,
    path_in_repo: str | None = None,
    token: str | None = None,
    private: bool | None = None,
    revision: str | None = None,
    create_pr: bool = False,
    commit_message: str | None = None,
    commit_description: str | None = None,
    delete_patterns: Sequence[str] | None = None,
    expected_format: str | None = None,
    dry_run: bool = False,
    api_factory: Callable[[str | None], Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_export_source(source, expected_format=expected_format if source.is_file() else None)
    repo_path = _normalize_repo_path(source, path_in_repo)
    result: dict[str, Any] = {
        "target": "huggingface-dataset-repo",
        "repoId": repo_id,
        "pathInRepo": repo_path,
        "source": summary,
        "dryRun": dry_run,
    }
    if dry_run:
        result["dryRunScope"] = "local-only"
        return result

    api_factory = api_factory or _create_hf_api
    api = api_factory(token)
    repo_url = api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    if source.is_dir():
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(source),
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
            revision=revision,
            create_pr=create_pr,
            delete_patterns=list(delete_patterns) if delete_patterns else None,
        )
        uploaded_exists = None
    else:
        commit = api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr,
        )
        uploaded_exists = api.file_exists(repo_id=repo_id, filename=repo_path, repo_type="dataset", revision=revision)

    result["repo"] = {
        "repoUrl": str(repo_url),
        "commit": _extract_fields(commit, "oid", "commit_url", "commit_message", "pr_url", "pr_num"),
        "uploadedPathExists": uploaded_exists,
    }
    return result


def upload_hf_bucket(
    *,
    source: Path,
    bucket_id: str,
    remote_path: str | None = None,
    token: str | None = None,
    private: bool | None = None,
    delete: bool = False,
    expected_format: str | None = None,
    dry_run: bool = False,
    api_factory: Callable[[str | None], Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_export_source(source, expected_format=expected_format if source.is_file() else None)
    destination = _normalize_bucket_path(source, remote_path)
    result: dict[str, Any] = {
        "target": "huggingface-bucket",
        "bucketId": bucket_id,
        "remotePath": destination,
        "source": summary,
        "dryRun": dry_run,
    }
    if dry_run:
        result["dryRunScope"] = "local-only"
        if source.is_dir():
            result["plan"] = {
                "destinationHandle": _bucket_handle(bucket_id, remote_path),
                "delete": delete,
            }
        else:
            result["plan"] = {
                "uploads": [{"local": str(source), "remote": destination}],
                "delete": False,
            }
        return result

    api_factory = api_factory or _create_hf_api
    api = api_factory(token)
    bucket_url = api.create_bucket(bucket_id=bucket_id, private=private, exist_ok=True)
    if source.is_dir():
        plan = api.sync_bucket(str(source), _bucket_handle(bucket_id, remote_path), delete=delete)
        plan_summary = plan.summary() if hasattr(plan, "summary") else None
        result["bucket"] = {
            "bucketUrl": str(bucket_url),
            "syncSummary": plan_summary,
        }
        return result

    api.batch_bucket_files(bucket_id, add=[(str(source), destination)])
    result["bucket"] = {
        "bucketUrl": str(bucket_url),
        "uploadedPaths": [destination],
    }
    return result
