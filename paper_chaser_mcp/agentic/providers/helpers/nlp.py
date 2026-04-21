"""Tokenization, similarity, and normalization primitives for provider helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from .schemas import _AnswerSchema

_ResponseModelT = TypeVar("_ResponseModelT", bound=BaseModel)

TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
COMMON_QUERY_WORDS = {
    "paper",
    "papers",
    "research",
    "study",
    "studies",
    "review",
    "recent",
    "latest",
    "work",
    "works",
}
MAX_EMBED_TEXT_LENGTH = 6_000
THEME_LABEL_STOPWORDS = COMMON_QUERY_WORDS | {
    "effect",
    "effects",
    "impact",
    "impacts",
    "response",
    "responses",
    "change",
    "changes",
    "documenting",
    "evidence",
    "findings",
    "highlighted",
    "main",
}


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _normalized_embedding_text(text: str) -> str:
    return " ".join(text.split())[:MAX_EMBED_TEXT_LENGTH]


def _top_terms(texts: list[str], *, limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(token for token in _tokenize(text) if token not in COMMON_QUERY_WORDS)
    return [term for term, _ in counts.most_common(limit)]


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens: Counter[str] = Counter(_tokenize(left))
    right_tokens: Counter[str] = Counter(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = set(left_tokens) & set(right_tokens)
    numerator = sum(left_tokens[token] * right_tokens[token] for token in intersection)
    left_norm = math.sqrt(sum(value * value for value in left_tokens.values()))
    right_norm = math.sqrt(sum(value * value for value in right_tokens.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    try:
        import numpy as np
    except ImportError:
        numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    left_array = np.array(left)
    right_array = np.array(right)
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))


def _normalize_confidence_label(value: Any) -> Literal["high", "medium", "low"]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"high", "medium", "low"}:
            return normalized  # type: ignore[return-value]
        if normalized in {"strong", "very_high", "very high"}:
            return "high"
        if normalized in {"moderate", "mid", "mixed"}:
            return "medium"
        if normalized in {"weak", "uncertain", "insufficient"}:
            return "low"
        try:
            numeric = float(normalized)
        except ValueError:
            numeric = None
        if numeric is not None:
            if numeric >= 0.8:
                return "high"
            if numeric >= 0.5:
                return "medium"
            return "low"
    if isinstance(value, (int, float)):
        if value >= 0.8:
            return "high"
        if value >= 0.5:
            return "medium"
        return "low"
    return "medium"


def _langchain_message_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    if isinstance(response, BaseModel):
        return response.model_dump_json()
    return str(response).strip()


def _extract_json_object(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None
    start = text.find("{")
    depth = 0
    for index, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    return None


def _normalize_label_text(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^(theme|label)\s*:\s*", "", line, flags=re.IGNORECASE)
        line = line.strip().strip('"').strip("'")
        if line:
            return line
    return ""


def _normalize_theme_label_output(text: str) -> str:
    label = _normalize_label_text(text)
    if label:
        return label
    return text.strip().strip('"').strip("'")


def _coerce_langchain_structured_response(
    response: Any,
    response_model: type[_ResponseModelT],
) -> _ResponseModelT:
    if isinstance(response, response_model):
        return response
    if isinstance(response, BaseModel):
        return response_model.model_validate(response.model_dump())
    if isinstance(response, dict):
        return response_model.model_validate(response)
    text = _langchain_message_text(response)
    if text:
        try:
            return response_model.model_validate_json(text)
        except Exception:
            json_payload = _extract_json_object(text)
            if json_payload:
                return response_model.model_validate_json(json_payload)
    raise ValueError("LangChain provider did not return structured output.")


def _extract_seed_identifiers(query: str) -> list[str]:
    identifiers: list[str] = []
    for pattern in (
        r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
        r"(arxiv:\d{4}\.\d{4,5}(?:v\d+)?)",
        r"((?:https?://)[^\s]+)",
        r"(\d{4}\.\d{4,5}(?:v\d+)?)",
    ):
        for match in re.findall(pattern, query, flags=re.IGNORECASE):
            identifiers.append(str(match))
    seen: set[str] = set()
    deduped: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        deduped.append(identifier)
    return deduped


def _collect_evidence_ids(evidence_papers: list[dict[str, Any]]) -> set[str]:
    valid_ids: set[str] = set()
    for paper in evidence_papers:
        if not isinstance(paper, dict):
            continue
        for key in ("paperId", "sourceId", "canonicalId"):
            value = str(paper.get(key) or "").strip()
            if value:
                valid_ids.add(value)
    return valid_ids


def _normalize_answer_schema_output(
    *,
    parsed_answer: "_AnswerSchema",
    evidence_papers: list[dict[str, Any]],
    confidence_normalizer: Any,
) -> dict[str, Any]:
    payload = parsed_answer.model_dump()
    payload["confidence"] = confidence_normalizer(payload.get("confidence"))
    valid_evidence_ids = _collect_evidence_ids(evidence_papers)
    selected_evidence_ids = [
        str(identifier).strip()
        for identifier in payload.get("selectedEvidenceIds") or []
        if str(identifier).strip() in valid_evidence_ids
    ]
    payload["selectedEvidenceIds"] = selected_evidence_ids
    payload["selectedLeadIds"] = [str(identifier).strip() for identifier in payload.get("selectedLeadIds") or []]
    cited_paper_ids = [
        str(identifier).strip()
        for identifier in payload.get("citedPaperIds") or []
        if str(identifier).strip() in valid_evidence_ids
    ]
    if not cited_paper_ids and selected_evidence_ids:
        cited_paper_ids = list(selected_evidence_ids[:3])
    payload["citedPaperIds"] = cited_paper_ids
    answer_text = str(payload.get("answer") or "").strip()
    if answer_text and not str(payload.get("evidenceSummary") or "").strip():
        payload["evidenceSummary"] = answer_text.split("\n", 1)[0][:240]
    if (
        payload.get("answerability") == "insufficient"
        and not str(payload.get("missingEvidenceDescription") or "").strip()
    ):
        payload["missingEvidenceDescription"] = "The supplied papers did not contain enough direct evidence."
    if payload.get("answerability") == "grounded" and not selected_evidence_ids and evidence_papers:
        payload["answerability"] = "limited"
    return payload
