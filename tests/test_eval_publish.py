import json
from builtins import __import__ as builtin_import
from pathlib import Path
from typing import Any, cast

import pytest

from paper_chaser_mcp.eval_exports import (
    export_foundry_eval_rows,
    export_hf_dataset_rows,
    export_training_chat_rows,
    load_jsonl_rows,
)
from paper_chaser_mcp.eval_publish import (
    MissingOptionalDependencyError,
    _create_ai_project_client,
    _create_default_credential,
    _create_hf_api,
    summarize_export_source,
    upload_foundry_dataset,
    upload_hf_bucket,
    upload_hf_dataset_repo,
    validate_export_rows,
)


def _trace_rows() -> list[dict[str, object]]:
    sample_path = Path(__file__).resolve().parent / "fixtures" / "evals" / "trace-promotion.sample.jsonl"
    return load_jsonl_rows(sample_path)


def test_validate_export_rows_accepts_all_supported_formats(tmp_path: Path) -> None:
    trace_rows = _trace_rows()

    foundry_rows = export_foundry_eval_rows(trace_rows)
    hf_rows = export_hf_dataset_rows(trace_rows)
    training_rows = export_training_chat_rows(trace_rows)

    assert validate_export_rows(foundry_rows, "foundry-eval")["rowCount"] == 2
    assert validate_export_rows(hf_rows, "hf-dataset")["taskFamilies"] == ["planner", "synthesis"]
    assert validate_export_rows(training_rows, "training-chat")["rowCount"] == 2


def test_summarize_export_source_validates_jsonl_file(tmp_path: Path) -> None:
    foundry_path = tmp_path / "foundry.jsonl"
    foundry_rows = export_foundry_eval_rows(_trace_rows())
    foundry_path.write_text("".join(json.dumps(row) + "\n" for row in foundry_rows), encoding="utf-8")

    summary = summarize_export_source(foundry_path, expected_format="foundry-eval")

    assert summary["rowCount"] == 2
    assert summary["validation"]["format"] == "foundry-eval"


def test_upload_foundry_dataset_uses_documented_upload_file_shape(tmp_path: Path) -> None:
    source = tmp_path / "foundry.jsonl"
    source.write_text(
        '{"id":"row-1","task_family":"planner","input":{},"expected":{},"metadata":{}}\n',
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    class FakeCredential:
        def __enter__(self) -> "FakeCredential":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeDatasets:
        def upload_file(self, **kwargs):
            calls["upload"] = kwargs
            return type("Dataset", (), {"id": "dataset-123", "name": kwargs["name"], "version": kwargs["version"]})()

    class FakeClient:
        def __init__(self) -> None:
            self.datasets = FakeDatasets()

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    result = upload_foundry_dataset(
        source=source,
        project_endpoint="https://example.services.ai.azure.com/api/projects/demo",
        dataset_name="paper-chaser-evals",
        dataset_version="2026.04.04",
        credential_factory=FakeCredential,
        client_factory=lambda endpoint, credential: FakeClient(),
        dry_run=False,
    )

    assert calls["upload"] == {
        "name": "paper-chaser-evals",
        "version": "2026.04.04",
        "file_path": str(source),
        "connection_name": None,
    }
    assert result["dataset"]["id"] == "dataset-123"


def test_upload_hf_dataset_repo_uses_create_repo_then_upload_file(tmp_path: Path) -> None:
    source = tmp_path / "hf.jsonl"
    source.write_text(
        '{"id":"row-1","task_family":"planner","input":{},"expected":{},"evaluation_target":"internal_llm_role","tags":[],"review_labels":{},"lineage":{}}\n',
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    class FakeApi:
        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs
            return "https://huggingface.co/datasets/example/paper-chaser"

        def upload_file(self, **kwargs):
            calls["upload_file"] = kwargs
            return type(
                "CommitInfo",
                (),
                {
                    "oid": "abc123",
                    "commit_url": "https://huggingface.co/commit/abc123",
                    "commit_message": kwargs.get("commit_message"),
                    "pr_url": None,
                    "pr_num": None,
                },
            )()

        def file_exists(self, **kwargs):
            calls["file_exists"] = kwargs
            return True

    result = upload_hf_dataset_repo(
        source=source,
        repo_id="example/paper-chaser-evals",
        expected_format="hf-dataset",
        api_factory=lambda token: FakeApi(),
        dry_run=False,
    )

    create_repo_call = cast(dict[str, Any], calls["create_repo"])
    upload_file_call = cast(dict[str, Any], calls["upload_file"])

    assert create_repo_call["repo_type"] == "dataset"
    assert upload_file_call["path_in_repo"] == "hf.jsonl"
    assert result["repo"]["uploadedPathExists"] is True


def test_upload_hf_bucket_uses_batch_bucket_files_for_single_file(tmp_path: Path) -> None:
    source = tmp_path / "training.jsonl"
    source.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}],"metadata":{}}\n',
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    class FakeApi:
        def create_bucket(self, **kwargs):
            calls["create_bucket"] = kwargs
            return "https://huggingface.co/buckets/example/paper-chaser"

        def batch_bucket_files(self, bucket_id, add):
            calls["batch_bucket_files"] = {"bucket_id": bucket_id, "add": add}

    result = upload_hf_bucket(
        source=source,
        bucket_id="example/paper-chaser-eval-artifacts",
        remote_path="training/",
        expected_format="training-chat",
        api_factory=lambda token: FakeApi(),
        dry_run=False,
    )

    create_bucket_call = cast(dict[str, Any], calls["create_bucket"])

    assert create_bucket_call["bucket_id"] == "example/paper-chaser-eval-artifacts"
    assert calls["batch_bucket_files"] == {
        "bucket_id": "example/paper-chaser-eval-artifacts",
        "add": [(str(source), "training/training.jsonl")],
    }
    assert result["bucket"]["uploadedPaths"] == ["training/training.jsonl"]


def test_create_default_credential_raises_helpful_error_when_azure_identity_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "azure.identity":
            raise ImportError("azure.identity unavailable")
        return builtin_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        _create_default_credential()

    assert ".[eval-foundry]" in str(exc_info.value)


def test_create_ai_project_client_raises_helpful_error_when_azure_projects_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "azure.ai.projects":
            raise ImportError("azure.ai.projects unavailable")
        return builtin_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        _create_ai_project_client("https://example.services.ai.azure.com/api/projects/demo", object())

    assert ".[eval-foundry]" in str(exc_info.value)


def test_create_hf_api_raises_helpful_error_when_huggingface_hub_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        if name == "huggingface_hub":
            raise ImportError("huggingface_hub unavailable")
        return builtin_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(MissingOptionalDependencyError) as exc_info:
        _create_hf_api()

    assert ".[eval-huggingface]" in str(exc_info.value)
