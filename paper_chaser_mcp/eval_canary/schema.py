"""Schema constants and low-level helpers for eval_canary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

SCHEMA_VERSION = 1

TASK_FAMILIES = {"planner", "synthesis", "abstention", "provenance", "runtime", "misc"}
ORIGINS = {"human", "synthetic_reviewed", "trace_mined", "adversarial"}
REVIEW_STATUSES = {"draft", "validated", "needs_review"}

PLANNER_INTENTS = {
    "discovery",
    "review",
    "known_item",
    "author",
    "citation",
    "regulatory",
}
ANSWER_STATUSES = {"answered", "abstained", "insufficient_evidence"}
ABSTENTION_BEHAVIORS = {"answer", "abstain", "clarify", "answer_with_caveats"}
PROVENANCE_SOURCE_TYPES = {
    "scholarly_article",
    "primary_regulatory",
    "secondary_regulatory",
    "unknown",
}
PROVENANCE_TRUST_STATES = {"verified_primary_source", "verified_metadata", "unverified"}
PROVENANCE_ACCESS_STATES = {
    "full_text_verified",
    "full_text_retrieved",
    "abstract_only",
    "access_unverified",
    "restricted",
}
RUNTIME_PROFILES = {"guided", "expert"}


def _validate_structured_model(name: str, model_cls: Any, payload: Any, errors: list[str]) -> None:
    try:
        model_cls.model_validate(payload)
    except ValidationError as exc:
        errors.append(f"{name} failed structured validation: {exc.errors()[0]['msg']}")


def _jsonl_files(dataset_root: Path, family_filter: str | None = None) -> list[Path]:
    files = sorted(path for path in dataset_root.glob("*.seed.jsonl") if path.is_file())
    if family_filter is None:
        return files
    prefix = f"{family_filter}."
    return [path for path in files if path.name.startswith(prefix)]


def _iter_raw_eval_rows(
    dataset_root: Path,
    family_filter: str | None = None,
    item_id_filter: str | None = None,
) -> list[tuple[Path, int, dict[str, Any]]]:
    rows: list[tuple[Path, int, dict[str, Any]]] = []
    for file_path in _jsonl_files(dataset_root, family_filter=family_filter):
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                raw_item = json.loads(line)
                if item_id_filter is not None:
                    item_id = str(raw_item.get("meta", {}).get("id") or "")
                    if item_id != item_id_filter:
                        continue
                rows.append((file_path, line_number, raw_item))
    return rows


def _result_item(
    *,
    item_id: str,
    family: str,
    path: str,
    line_number: int,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    status = "failed" if errors else ("warning" if warnings else "passed")
    return {
        "id": item_id,
        "family": family,
        "path": path,
        "lineNumber": line_number,
        "status": status,
        "errors": errors,
        "warnings": warnings,
    }
