"""Generate eval topic and query candidates using the configured planner model."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import subprocess  # nosec B404 - trusted local CLI wrapper for local repo scripts
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from paper_chaser_mcp.agentic.config import AgenticConfig
from paper_chaser_mcp.agentic.planner import normalize_query
from paper_chaser_mcp.agentic.providers import resolve_provider_bundle
from paper_chaser_mcp.settings import AppSettings

DEFAULT_TOPIC_FAMILY_BY_INTENT = {
    "discovery": "literature_discovery",
    "review": "evidence_review",
    "known_item": "known_item_lookup",
    "author": "author_lookup",
    "citation": "citation_repair",
    "regulatory": "regulatory_research",
}

SEED_PRESETS = {
    "balanced-science": (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "fixtures"
        / "evals"
        / "topic-seeds.balanced-science.sample.txt"
    ),
    "environmental-consulting": (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "fixtures"
        / "evals"
        / "topic-seeds.environmental-consulting.sample.txt"
    ),
}
DEFAULT_SEED_PRESET = "balanced-science"

TAXONOMY_PRESETS = {
    "balanced-science": (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "fixtures"
        / "evals"
        / "topic-taxonomy.balanced-science.sample.json"
    ),
    "environmental-consulting": (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "fixtures"
        / "evals"
        / "topic-taxonomy.environmental-consulting.sample.json"
    ),
}
DEFAULT_TAXONOMY_PRESET = "balanced-science"
DEFAULT_TAXONOMY_FILE = TAXONOMY_PRESETS[DEFAULT_TAXONOMY_PRESET]


def _script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def _load_dotenv(dotenv_path: Path) -> int:
    loaded = 0
    if not dotenv_path.exists():
        return loaded
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in __import__("os").environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        __import__("os").environ[key] = value
        loaded += 1
    return loaded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate realistic eval topic and query candidates using the configured planner model, "
            "and optionally emit an expert batch scenario file for trace capture."
        )
    )
    parser.add_argument(
        "--easy-button",
        action="store_true",
        help="Apply integrated defaults for outputs, ranking, follow-ups, and review artifacts.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(Path(__file__).resolve().parent.parent / "build" / "eval-workflow"),
        help="Base directory for generated artifacts when using integrated defaults.",
    )
    parser.add_argument(
        "--seed-query",
        action="append",
        default=None,
        help="Seed research ask to expand into more realistic eval queries. Repeat for multiple seeds.",
    )
    parser.add_argument(
        "--seed-file",
        default=None,
        help="Optional newline-delimited file of seed research asks.",
    )
    parser.add_argument(
        "--seed-preset",
        choices=[*sorted(SEED_PRESETS), "none"],
        default=None,
        help=(
            "Optional checked-in seed preset. When no explicit seed query or seed file is provided, "
            f"the default starter preset '{DEFAULT_SEED_PRESET}' is used automatically."
        ),
    )
    parser.add_argument(
        "--context-text",
        action="append",
        default=None,
        help="Optional contextual text or user-language snippets to ground the expansions.",
    )
    parser.add_argument(
        "--single-seed-diversification",
        action="store_true",
        help=(
            "When a run starts from a single seed, ask the planner for additional review, regulatory, "
            "and methods-oriented variants to improve intent and family coverage."
        ),
    )
    parser.add_argument(
        "--taxonomy-preset",
        choices=sorted(TAXONOMY_PRESETS),
        default=DEFAULT_TAXONOMY_PRESET,
        help="Checked-in taxonomy preset to use when --taxonomy-file is not provided.",
    )
    parser.add_argument(
        "--taxonomy-file",
        default=None,
        help=(
            "Optional JSON taxonomy file with a top-level 'rules' array. Each rule may define "
            "family, tags, keywords, and intents."
        ),
    )
    parser.add_argument(
        "--latency-profile",
        default="balanced",
        choices=["fast", "balanced", "deep"],
        help="Latency profile used for planner-side generation and classification.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=3,
        help="Maximum LLM-generated variants per seed query.",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.0,
        help="Minimum topic quality score to keep after ranking.",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Optional maximum number of ranked topics to keep after filtering.",
    )
    parser.add_argument(
        "--ai-prune-mode",
        choices=["off", "rewrite", "rewrite-or-drop"],
        default="off",
        help="Optionally rewrite or drop weak topics with another planner-assisted pass before workflow handoff.",
    )
    parser.add_argument(
        "--ai-prune-below-score",
        type=float,
        default=35.0,
        help="Topics scoring below this threshold are candidates for AI-assisted rewrite or drop.",
    )
    parser.add_argument(
        "--domain-balance-mode",
        choices=["off", "round-robin"],
        default="off",
        help="Optionally rebalance the final ranked pool across topic families instead of pure score order.",
    )
    parser.add_argument(
        "--domain-balance-max-share",
        type=float,
        default=0.4,
        help="Maximum preferred share for any one family in the final pool when balancing is enabled.",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include the original seed query as a candidate alongside generated variants.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write generated topic JSON, or '-' for stdout.",
    )
    parser.add_argument(
        "--scenario-output",
        default=None,
        help="Optional path to also write an expert batch scenario JSON file from the generated topics.",
    )
    parser.add_argument(
        "--jsonl-output",
        default=None,
        help="Optional path to also export ranked topics as JSONL for review workflows.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to also export ranked topics as CSV for spreadsheet review.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional path to also export ranked topics as a markdown table for editor or PR review.",
    )
    parser.add_argument(
        "--merge-inputs",
        nargs="*",
        default=None,
        help="Optional prior generated-topic JSON files to merge and deduplicate with the current run.",
    )
    parser.add_argument(
        "--emit-follow-up",
        action="store_true",
        help="When writing a scenario file, also emit a synthesis follow-up scenario for each generated search.",
    )
    parser.add_argument(
        "--dotenv-path",
        default=str(Path(__file__).resolve().parent.parent / ".env"),
        help="Path to .env file used to load provider credentials and model settings.",
    )
    parser.add_argument(
        "--launch-viewer",
        action="store_true",
        help="Launch the local generated-topic viewer after writing outputs.",
    )
    parser.add_argument(
        "--launch-workflow",
        action="store_true",
        help="Run the full eval workflow immediately after generating the batch scenario file.",
    )
    parser.add_argument(
        "--workflow-review-mode",
        choices=["yolo", "ui", "skip"],
        default="yolo",
        help="Review mode to use when --launch-workflow is enabled.",
    )
    parser.add_argument(
        "--workflow-autopilot-policy",
        choices=["safe", "review", "blocked"],
        default="review",
        help="Guardrail policy for automatic workflow handoff when --launch-workflow is enabled.",
    )
    parser.add_argument(
        "--workflow-matrix-preset",
        action="append",
        default=None,
        help="Matrix preset(s) to pass through to run_eval_workflow.py when --launch-workflow is enabled.",
    )
    parser.add_argument(
        "--workflow-launch-matrix-viewer",
        action="store_true",
        help="Also launch the eval matrix viewer from run_eval_workflow.py.",
    )
    parser.add_argument(
        "--force-launch-workflow",
        action="store_true",
        help="Allow workflow handoff even when the generated topic pool recommends human review.",
    )
    parser.add_argument(
        "--viewer-host",
        default="127.0.0.1",
        help="Host to bind the optional generated-topic viewer to.",
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=8767,
        help="Port to bind the optional generated-topic viewer to.",
    )
    return parser


def load_seed_queries(args: argparse.Namespace) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()
    seed_files: list[Path] = []

    preset_name = getattr(args, "seed_preset", None)
    if preset_name is None and not (args.seed_query or args.seed_file):
        preset_name = DEFAULT_SEED_PRESET
    if preset_name and preset_name != "none":
        seed_files.append(SEED_PRESETS[preset_name])

    for query in args.seed_query or []:
        normalized = normalize_query(str(query))
        lowered = normalized.lower()
        if normalized and lowered not in seen:
            seen.add(lowered)
            seeds.append(normalized)
    if args.seed_file:
        seed_files.append(Path(args.seed_file))
    for seed_file in seed_files:
        for raw_line in seed_file.read_text(encoding="utf-8").splitlines():
            normalized = normalize_query(raw_line)
            lowered = normalized.lower()
            if normalized and lowered not in seen:
                seen.add(lowered)
                seeds.append(normalized)
    if not seeds:
        raise ValueError("Provide at least one --seed-query or --seed-file entry.")
    return seeds


def _settings_from_env(dotenv_path: Path) -> AppSettings:
    _load_dotenv(dotenv_path)
    return AppSettings.from_env()


def apply_easy_button_defaults(args: argparse.Namespace) -> None:
    if not getattr(args, "easy_button", False):
        return
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    if not args.output:
        args.output = str(artifact_root / "generated-topics.json")
    if not args.jsonl_output:
        args.jsonl_output = str(artifact_root / "generated-topics.jsonl")
    if not args.csv_output:
        args.csv_output = str(artifact_root / "generated-topics.csv")
    if not args.markdown_output:
        args.markdown_output = str(artifact_root / "generated-topics.md")
    if not args.scenario_output:
        args.scenario_output = str(artifact_root / "generated-batch.json")
    if args.seed_preset is None:
        args.seed_preset = DEFAULT_SEED_PRESET
    if not args.include_original:
        args.include_original = True
    if not args.emit_follow_up:
        args.emit_follow_up = True
    if float(args.min_quality_score) < 30.0:
        args.min_quality_score = 30.0
    if args.max_topics is None:
        args.max_topics = 25
    if getattr(args, "ai_prune_mode", "off") == "off":
        args.ai_prune_mode = "rewrite-or-drop"
    if getattr(args, "domain_balance_mode", "off") == "off":
        args.domain_balance_mode = "round-robin"
    if not args.launch_workflow:
        args.launch_viewer = True
    if args.launch_workflow and not args.workflow_matrix_preset:
        args.workflow_matrix_preset = ["cross-provider-lower-bound"]
    if args.launch_workflow and getattr(args, "workflow_autopilot_policy", "review") == "review":
        args.workflow_autopilot_policy = "safe"


def resolve_taxonomy_file(args: argparse.Namespace) -> Path:
    if args.taxonomy_file:
        return Path(args.taxonomy_file)
    return TAXONOMY_PRESETS[args.taxonomy_preset]


def load_taxonomy_rules(taxonomy_file: str | None) -> list[dict[str, Any]]:
    if not taxonomy_file:
        return []
    payload = json.loads(Path(taxonomy_file).read_text(encoding="utf-8"))
    rules = payload.get("rules") if isinstance(payload, dict) else payload
    if not isinstance(rules, list):
        raise ValueError("taxonomy file must be a JSON array or an object containing a 'rules' array")
    return [rule for rule in rules if isinstance(rule, dict)]


def write_topics_jsonl(path: Path, topics: list[dict[str, Any]]) -> None:
    lines = [json.dumps(topic, ensure_ascii=True) for topic in topics]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_topics_csv(path: Path, topics: list[dict[str, Any]]) -> None:
    fieldnames = [
        "query",
        "seedQuery",
        "intent",
        "family",
        "tags",
        "qualityScore",
        "qualityTier",
        "followUpMode",
        "providerPlan",
        "candidateConcepts",
        "successCriteria",
        "rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for topic in topics:
            writer.writerow(
                {
                    "query": topic.get("query") or "",
                    "seedQuery": topic.get("seedQuery") or "",
                    "intent": topic.get("intent") or "",
                    "family": topic.get("family") or "",
                    "tags": "; ".join(str(tag) for tag in topic.get("tags") or []),
                    "qualityScore": topic.get("qualityScore") or "",
                    "qualityTier": topic.get("qualityTier") or "",
                    "followUpMode": topic.get("followUpMode") or "",
                    "providerPlan": "; ".join(str(item) for item in topic.get("providerPlan") or []),
                    "candidateConcepts": "; ".join(str(item) for item in topic.get("candidateConcepts") or []),
                    "successCriteria": "; ".join(str(item) for item in topic.get("successCriteria") or []),
                    "rationale": topic.get("rationale") or "",
                }
            )


def write_topics_markdown(path: Path, topics: list[dict[str, Any]]) -> None:
    lines = [
        "| Query | Family | Intent | Score | Tier | Tags |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for topic in topics:
        lines.append(
            "| {query} | {family} | {intent} | {score} | {tier} | {tags} |".format(
                query=str(topic.get("query") or "").replace("|", "\\|"),
                family=str(topic.get("family") or "").replace("|", "\\|"),
                intent=str(topic.get("intent") or "").replace("|", "\\|"),
                score=str(topic.get("qualityScore") or ""),
                tier=str(topic.get("qualityTier") or ""),
                tags=", ".join(str(tag) for tag in topic.get("tags") or []).replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _topic_family(topic: dict[str, Any]) -> str:
    return str(topic.get("family") or "unknown")


def _taxonomy_match_candidates(topic: dict[str, Any], taxonomy_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query = normalize_query(str(topic.get("query") or ""))
    intent = str(topic.get("intent") or "")
    rationale = str(topic.get("rationale") or "")
    candidate_concepts = [str(item).strip() for item in topic.get("candidateConcepts") or [] if str(item).strip()]
    tags = {str(item).strip().lower() for item in topic.get("tags") or [] if str(item).strip()}
    haystack = " ".join([query, rationale, *candidate_concepts]).lower()

    candidates: list[dict[str, Any]] = []
    for rule in taxonomy_rules:
        family = str(rule.get("family") or "").strip()
        if not family:
            continue
        matched_keywords = [
            keyword
            for keyword in [str(item).strip().lower() for item in rule.get("keywords") or [] if str(item).strip()]
            if keyword and keyword in haystack
        ]
        matched_tags = [
            tag
            for tag in [str(item).strip().lower() for item in rule.get("tags") or [] if str(item).strip()]
            if tag and tag in tags
        ]
        rule_intents = {str(item).strip() for item in rule.get("intents") or [] if str(item).strip()}
        score = float(len(matched_keywords))
        score += 0.5 * len(matched_tags)
        if rule_intents and intent in rule_intents:
            score += 0.5
        if score <= 0.0:
            continue
        candidates.append(
            {
                "family": family,
                "score": round(score, 2),
                "matchedKeywords": matched_keywords,
                "matchedTags": matched_tags,
                "intentMatched": intent in rule_intents if rule_intents else False,
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["family"])))
    return candidates


def build_family_cross_check(topic: dict[str, Any], taxonomy_rules: list[dict[str, Any]]) -> dict[str, Any]:
    assigned_family = _topic_family(topic)
    candidates = _taxonomy_match_candidates(topic, taxonomy_rules)
    top_candidate = candidates[0]["family"] if candidates else assigned_family
    disagreement = bool(candidates) and top_candidate != assigned_family
    return {
        "assignedFamily": assigned_family,
        "topCandidateFamily": top_candidate,
        "disagreement": disagreement,
        "candidateCount": len(candidates),
        "candidates": candidates[:5],
        "warning": (
            f"assigned family '{assigned_family}' disagrees with cross-check candidate '{top_candidate}'"
            if disagreement
            else None
        ),
    }


def annotate_family_cross_checks(
    topics: list[dict[str, Any]],
    taxonomy_rules: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    disagreement_count = 0
    no_signal_count = 0
    for topic in topics:
        enriched = dict(topic)
        cross_check = build_family_cross_check(enriched, taxonomy_rules)
        enriched["familyCrossCheck"] = cross_check
        if cross_check["disagreement"]:
            disagreement_count += 1
        if cross_check["candidateCount"] == 0:
            no_signal_count += 1
        annotated.append(enriched)
    return annotated, {
        "disagreementCount": disagreement_count,
        "noSignalCount": no_signal_count,
        "topicCount": len(annotated),
    }


def summarize_topic_pool(topics: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    intents: dict[str, int] = {}
    quality_tiers: dict[str, int] = {}
    total_score = 0.0
    for topic in topics:
        family = str(topic.get("family") or "unknown")
        intent = str(topic.get("intent") or "unknown")
        tier = str(topic.get("qualityTier") or "unscored")
        families[family] = families.get(family, 0) + 1
        intents[intent] = intents.get(intent, 0) + 1
        quality_tiers[tier] = quality_tiers.get(tier, 0) + 1
        total_score += float(topic.get("qualityScore") or 0.0)
    topic_count = len(topics)
    average_quality = round(total_score / topic_count, 1) if topic_count else 0.0
    max_family_share = 0.0
    max_intent_share = 0.0
    if topic_count:
        max_family_share = max(count / topic_count for count in families.values()) if families else 0.0
        max_intent_share = max(count / topic_count for count in intents.values()) if intents else 0.0
    return {
        "topicCount": topic_count,
        "averageQualityScore": average_quality,
        "families": families,
        "intents": intents,
        "qualityTiers": quality_tiers,
        "familyCount": len(families),
        "intentCount": len(intents),
        "maxFamilyShare": round(max_family_share, 3),
        "maxIntentShare": round(max_intent_share, 3),
        "familyDisagreementCount": sum(
            1
            for topic in topics
            if isinstance(topic.get("familyCrossCheck"), dict) and bool(topic["familyCrossCheck"].get("disagreement"))
        ),
    }


def add_prune_metrics_to_summary(summary: dict[str, Any], prune_summary: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(summary)
    topic_count = int(summary.get("topicCount") or 0)
    rewritten = int(prune_summary.get("rewritten") or 0)
    dropped = int(prune_summary.get("dropped") or 0)
    enriched["rewrittenCount"] = rewritten
    enriched["droppedCount"] = dropped
    enriched["rewrittenShare"] = round((rewritten / topic_count), 3) if topic_count else 0.0
    return enriched


def _hard_blocker_threshold(args: argparse.Namespace | None, name: str, default: int | float) -> int | float:
    if args is None:
        return default
    value = getattr(args, name, default)
    return default if value is None else value


def identify_hard_blockers(
    summary: dict[str, Any],
    *,
    args: argparse.Namespace | None = None,
) -> list[str]:
    blockers: list[str] = []
    min_topic_count = int(_hard_blocker_threshold(args, "hard_blocker_min_topic_count", 5))
    max_family_share = float(_hard_blocker_threshold(args, "hard_blocker_max_family_share", 0.7))
    max_family_disagreement_count = int(_hard_blocker_threshold(args, "hard_blocker_max_family_disagreement_count", 3))
    max_rewritten_share = float(_hard_blocker_threshold(args, "hard_blocker_max_rewritten_share", 0.5))

    if int(summary.get("topicCount") or 0) < min_topic_count:
        min_topic_label = "five" if min_topic_count == 5 else str(min_topic_count)
        blockers.append(f"topic pool contains fewer than {min_topic_label} topics")
    if int((summary.get("qualityTiers") or {}).get("high") or 0) == 0:
        blockers.append("topic pool contains no high-quality topics")
    if float(summary.get("maxFamilyShare") or 0.0) > max_family_share:
        blockers.append("topic pool is dominated by a single family beyond the hard blocker threshold")
    if int(summary.get("familyDisagreementCount") or 0) > max_family_disagreement_count:
        blockers.append("family cross-check disagreement count exceeds the hard blocker threshold")
    if float(summary.get("rewrittenShare") or 0.0) > max_rewritten_share:
        blockers.append("rewritten topic share exceeds the hard blocker threshold")
    return blockers


def _fallback_plan(query: str, error: Exception) -> dict[str, Any]:
    return {
        "intent": "discovery",
        "followUpMode": "qa",
        "providerPlan": [],
        "candidateConcepts": [],
        "constraints": {},
        "successCriteria": [],
        "rationale": f"Planner fallback used after provider error: {type(error).__name__}",
        "query": query,
    }


def assess_topic_pool_risks(summary: dict[str, Any]) -> tuple[list[str], str]:
    warnings: list[str] = []
    topic_count = int(summary.get("topicCount") or 0)
    average_quality = float(summary.get("averageQualityScore") or 0.0)
    family_count = int(summary.get("familyCount") or 0)
    max_family_share = float(summary.get("maxFamilyShare") or 0.0)
    max_intent_share = float(summary.get("maxIntentShare") or 0.0)
    intents = summary.get("intents") or {}
    high_count = int((summary.get("qualityTiers") or {}).get("high") or 0)
    disagreement_count = int(summary.get("familyDisagreementCount") or 0)

    if topic_count < 10:
        warnings.append("topic pool is small; generated eval coverage may be too narrow")
    if average_quality < 35.0:
        warnings.append("average topic quality is low; generated scenarios may be noisy")
    if family_count < 4:
        warnings.append("domain diversity is low; topic pool may be overfit to a narrow family set")
    if max_family_share > 0.45:
        warnings.append("one family dominates the pool; domain balance may be poor")
    if max_intent_share > 0.7:
        warnings.append("one intent dominates the pool; eval routing coverage may be weak")
    if len(intents) <= 1 or set(intents) == {"discovery"}:
        warnings.append(
            "non-discovery intents are underrepresented; known-item, review, and regulatory coverage may be weak"
        )
    if high_count == 0:
        warnings.append(
            "no high-quality topics remain after ranking; human review is recommended before workflow handoff"
        )
    if disagreement_count > 0:
        warnings.append("family cross-check disagrees with assigned family for one or more topics")

    recommendation = "ai-assisted-ok" if len(warnings) <= 1 else "human-review-required"
    return warnings, recommendation


def _follow_up_question(follow_up_mode: str) -> str:
    if follow_up_mode == "claim_check":
        return "Which recurring claims in this result set are strongly supported versus weakly supported?"
    if follow_up_mode == "comparison":
        return "What comparisons or tradeoffs show up across the strongest papers in this result set?"
    return "What evaluation tradeoffs, evidence gaps, or practical implications show up in this result set?"


def _slugify(value: str) -> str:
    tokens = [token.lower() for token in normalize_query(value).split() if token]
    return "_".join(tokens[:6]) or "query"


def score_topic_candidate(topic: dict[str, Any]) -> tuple[float, str, list[str]]:
    score = 0.0
    signals: list[str] = []
    query = normalize_query(str(topic.get("query") or ""))
    seed_query = normalize_query(str(topic.get("seedQuery") or ""))
    intent = str(topic.get("intent") or "")
    provider_plan = [str(item).strip() for item in topic.get("providerPlan") or [] if str(item).strip()]
    candidate_concepts = [str(item).strip() for item in topic.get("candidateConcepts") or [] if str(item).strip()]
    success_criteria = [str(item).strip() for item in topic.get("successCriteria") or [] if str(item).strip()]
    tags = [str(item).strip() for item in topic.get("tags") or [] if str(item).strip()]
    family = str(topic.get("family") or "").strip()
    token_count = len(query.split())

    if query and query != seed_query:
        score += 18.0
        signals.append("novel_from_seed")
    elif query:
        score += 8.0
        signals.append("literal_seed")

    if 3 <= token_count <= 12:
        score += 12.0
        signals.append("query_length_good")
    elif token_count >= 2:
        score += 6.0
        signals.append("query_length_ok")

    if intent:
        score += 12.0
        signals.append(f"intent:{intent}")
    if intent in {"known_item", "review", "regulatory"}:
        score += 6.0
        signals.append("non_default_intent")

    if provider_plan:
        score += min(len(provider_plan), 3) * 4.0
        signals.append("provider_plan")
    if candidate_concepts:
        score += min(len(candidate_concepts), 3) * 4.0
        signals.append("candidate_concepts")
    if success_criteria:
        score += min(len(success_criteria), 3) * 4.0
        signals.append("success_criteria")
    if family and family not in DEFAULT_TOPIC_FAMILY_BY_INTENT.values():
        score += 10.0
        signals.append("taxonomy_family")
    if tags:
        score += min(len(tags), 4) * 2.0
        signals.append("taxonomy_tags")

    if score >= 55.0:
        tier = "high"
    elif score >= 35.0:
        tier = "medium"
    else:
        tier = "low"
    return round(score, 1), tier, signals


def rank_topics(
    topics: list[dict[str, Any]],
    *,
    min_quality_score: float,
    max_topics: int | None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for topic in topics:
        enriched = dict(topic)
        score, tier, signals = score_topic_candidate(enriched)
        enriched["qualityScore"] = score
        enriched["qualityTier"] = tier
        enriched["qualitySignals"] = signals
        if score < min_quality_score:
            continue
        ranked.append(enriched)
    ranked.sort(
        key=lambda item: (
            -float(item.get("qualityScore") or 0.0),
            str(item.get("intent") or ""),
            str(item.get("query") or "").lower(),
        )
    )
    if max_topics is not None:
        return ranked[:max_topics]
    return ranked


def rebalance_topics(
    topics: list[dict[str, Any]],
    *,
    mode: str,
    max_topics: int | None,
    max_family_share: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode == "off" or not topics:
        limited = topics[:max_topics] if max_topics is not None else list(topics)
        return limited, {"mode": mode, "applied": False, "selectedCount": len(limited)}

    ordered_topics = list(topics)
    family_order: list[str] = []
    family_queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for topic in ordered_topics:
        family = _topic_family(topic)
        if family not in family_queues:
            family_order.append(family)
        family_queues[family].append(topic)

    target_count = max_topics if max_topics is not None else len(ordered_topics)
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = defaultdict(int)

    # First pass: guarantee one topic per family while capacity remains.
    for family in family_order:
        if len(selected) >= target_count:
            break
        if family_queues[family]:
            topic = family_queues[family].popleft()
            selected.append(topic)
            family_counts[family] += 1

    made_progress = True
    while len(selected) < target_count and made_progress:
        made_progress = False
        for family in family_order:
            if len(selected) >= target_count:
                break
            queue = family_queues[family]
            if not queue:
                continue
            projected_share = (family_counts[family] + 1) / (len(selected) + 1)
            if projected_share > max_family_share and len(selected) >= len(family_order):
                continue
            topic = queue.popleft()
            selected.append(topic)
            family_counts[family] += 1
            made_progress = True

    # If caps left unused slots, fill the rest in original ranked order.
    if len(selected) < target_count:
        remaining: list[dict[str, Any]] = []
        for family in family_order:
            remaining.extend(list(family_queues[family]))
        selected.extend(remaining[: max(0, target_count - len(selected))])

    return selected, {
        "mode": mode,
        "applied": True,
        "selectedCount": len(selected),
        "familyCounts": dict(family_counts),
        "maxFamilyShare": max_family_share,
    }


async def ai_assisted_prune_topics(
    topics: list[dict[str, Any]],
    *,
    mode: str,
    prune_below_score: float,
    provider_bundle: Any,
    taxonomy_rules: list[dict[str, Any]],
    context_texts: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode == "off":
        return list(topics), {"mode": mode, "rewritten": 0, "dropped": 0, "kept": len(topics), "audit": []}

    kept: list[dict[str, Any]] = []
    rewritten = 0
    dropped = 0
    audit: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for topic in topics:
        original_score, _, _ = score_topic_candidate(topic)
        normalized_query = normalize_query(str(topic.get("query") or ""))
        lowered_query = normalized_query.lower()
        original_intent = str(topic.get("intent") or "")
        if original_score >= prune_below_score:
            if lowered_query not in seen_queries:
                kept.append(topic)
                seen_queries.add(lowered_query)
            continue

        seed_query = normalize_query(str(topic.get("seedQuery") or normalized_query))
        evidence_texts = [
            value
            for value in [
                seed_query,
                normalized_query,
                str(topic.get("rationale") or ""),
                *context_texts,
                *[str(item) for item in topic.get("candidateConcepts") or []],
            ]
            if normalize_query(str(value))
        ]
        expansions = await provider_bundle.asuggest_speculative_expansions(
            query=seed_query,
            evidence_texts=evidence_texts,
            max_variants=3,
        )
        best_rewrite: dict[str, Any] | None = None
        best_score = original_score
        for candidate in expansions:
            candidate_query = normalize_query(candidate.variant)
            candidate_key = candidate_query.lower()
            if not candidate_query or candidate_key in seen_queries or candidate_key == lowered_query:
                continue
            try:
                plan = await provider_bundle.aplan_search(query=candidate_query, mode="auto")
            except Exception as error:
                audit.append(
                    {
                        "action": "rewrite_error",
                        "originalQuery": normalized_query,
                        "candidateQuery": candidate_query,
                        "originalScore": original_score,
                        "reason": f"Rewrite planning failed: {type(error).__name__}",
                    }
                )
                continue
            if str(plan.intent or "") != original_intent:
                audit.append(
                    {
                        "action": "rewrite_rejected",
                        "originalQuery": normalized_query,
                        "candidateQuery": candidate_query,
                        "originalIntent": original_intent,
                        "candidateIntent": plan.intent,
                        "reason": "Rewrite rejected because the inferred intent changed.",
                    }
                )
                continue
            family, tags = assign_topic_taxonomy(
                query=candidate_query,
                intent=plan.intent,
                candidate_concepts=plan.candidate_concepts,
                taxonomy_rules=taxonomy_rules,
            )
            rewritten_topic = {
                **topic,
                "query": candidate_query,
                "intent": plan.intent,
                "family": family,
                "tags": tags,
                "followUpMode": plan.follow_up_mode,
                "providerPlan": plan.provider_plan,
                "candidateConcepts": plan.candidate_concepts,
                "constraints": plan.constraints,
                "successCriteria": plan.success_criteria,
                "rationale": f"AI rewrite of weak topic: {candidate.rationale or topic.get('rationale') or ''}".strip(),
                "rewriteSource": normalized_query,
            }
            candidate_score, _, _ = score_topic_candidate(rewritten_topic)
            if candidate_score > best_score + 2.0:
                best_score = candidate_score
                best_rewrite = rewritten_topic

        if best_rewrite is not None:
            kept.append(best_rewrite)
            seen_queries.add(normalize_query(str(best_rewrite.get("query") or "")).lower())
            rewritten += 1
            audit.append(
                {
                    "action": "rewritten",
                    "originalQuery": normalized_query,
                    "rewrittenQuery": str(best_rewrite.get("query") or ""),
                    "originalScore": original_score,
                    "rewrittenScore": best_score,
                    "reason": best_rewrite.get("rationale") or "AI rewrite improved the topic.",
                }
            )
            continue

        if mode == "rewrite-or-drop":
            dropped += 1
            audit.append(
                {
                    "action": "dropped",
                    "originalQuery": normalized_query,
                    "originalScore": original_score,
                    "reason": "No stronger rewrite cleared the prune threshold.",
                }
            )
            continue

        if lowered_query not in seen_queries:
            kept.append(topic)
            seen_queries.add(lowered_query)

    return kept, {"mode": mode, "rewritten": rewritten, "dropped": dropped, "kept": len(kept), "audit": audit}


def assign_topic_taxonomy(
    *,
    query: str,
    intent: str,
    candidate_concepts: list[str],
    taxonomy_rules: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    family = DEFAULT_TOPIC_FAMILY_BY_INTENT.get(intent, "general_research")
    tags = [intent]
    haystack = " ".join([query, *candidate_concepts]).lower()
    for rule in taxonomy_rules:
        rule_intents = {str(item).strip() for item in rule.get("intents") or [] if str(item).strip()}
        if rule_intents and intent not in rule_intents:
            continue
        keywords = [str(item).strip().lower() for item in rule.get("keywords") or [] if str(item).strip()]
        if keywords and not any(keyword in haystack for keyword in keywords):
            continue
        rule_family = str(rule.get("family") or "").strip()
        if rule_family:
            family = rule_family
        for tag in rule.get("tags") or []:
            normalized = str(tag).strip()
            if normalized and normalized not in tags:
                tags.append(normalized)
    if family not in tags:
        tags.append(family)
    return family, tags


def merge_generated_topic_payloads(
    current_payload: dict[str, Any],
    merge_inputs: list[str] | None,
) -> dict[str, Any]:
    if not merge_inputs:
        return current_payload

    merged_by_query: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    seed_queries: list[str] = []
    seen_seeds: set[str] = set()

    def _remember_seed(value: str) -> None:
        lowered = value.lower()
        if value and lowered not in seen_seeds:
            seen_seeds.add(lowered)
            seed_queries.append(value)

    def _merge_payload(payload: dict[str, Any]) -> None:
        for seed in payload.get("seedQueries") or []:
            normalized = normalize_query(str(seed))
            if normalized:
                _remember_seed(normalized)
        for topic in payload.get("topics") or []:
            if not isinstance(topic, dict):
                continue
            query = normalize_query(str(topic.get("query") or ""))
            if not query:
                continue
            key = query.lower()
            if key not in merged_by_query:
                ordered_keys.append(key)
            merged_by_query[key] = topic

    for input_path in merge_inputs:
        _merge_payload(json.loads(Path(input_path).read_text(encoding="utf-8")))
    _merge_payload(current_payload)

    merged_payload = dict(current_payload)
    merged_payload["seedQueries"] = seed_queries
    merged_payload["topics"] = [merged_by_query[key] for key in ordered_keys]
    merged_payload["mergedInputs"] = merge_inputs
    return merged_payload


def _single_seed_diversification_briefs(seed_query: str) -> list[dict[str, str]]:
    return [
        {
            "angle": "review",
            "instruction": (
                "Generate one evidence-review or synthesis-oriented search variant for this seed. "
                "Prefer literature review, evidence synthesis, or meta-analysis framing."
            ),
        },
        {
            "angle": "regulatory",
            "instruction": (
                "Generate one regulatory, policy, guidance, or compliance-oriented search variant for this seed. "
                "Prefer standards, guidelines, or policy implications framing."
            ),
        },
        {
            "angle": "methods",
            "instruction": (
                "Generate one methods, measurement, monitoring, modeling, or exposure-assessment-oriented "
                "search variant for this seed. Prefer instrumentation, workflow, or analytic-method framing."
            ),
        },
    ]


async def maybe_add_single_seed_diversifications(
    *,
    args: argparse.Namespace,
    provider_bundle: Any,
    taxonomy_rules: list[dict[str, Any]],
    context_texts: list[str],
    seed_queries: list[str],
    topics: list[dict[str, Any]],
    seen_queries: set[str],
    generation_warnings: list[str],
) -> None:
    if not getattr(args, "single_seed_diversification", False):
        return
    if len(seed_queries) != 1:
        return

    seed_query = seed_queries[0]
    for brief in _single_seed_diversification_briefs(seed_query):
        try:
            expansions = await provider_bundle.asuggest_speculative_expansions(
                query=seed_query,
                evidence_texts=[*context_texts, brief["instruction"], f"Diversification angle: {brief['angle']}"],
                max_variants=1,
            )
        except Exception as error:
            generation_warnings.append(
                (
                    f"Single-seed diversification failed for angle '{brief['angle']}': {type(error).__name__}; "
                    "skipping diversification candidate."
                )
            )
            continue

        for candidate in expansions:
            normalized = normalize_query(candidate.variant)
            lowered = normalized.lower()
            if not normalized or lowered in seen_queries:
                continue
            seen_queries.add(lowered)
            try:
                plan = await provider_bundle.aplan_search(query=normalized, mode="auto")
                plan_data = {
                    "intent": plan.intent,
                    "followUpMode": plan.follow_up_mode,
                    "providerPlan": plan.provider_plan,
                    "candidateConcepts": plan.candidate_concepts,
                    "constraints": plan.constraints,
                    "successCriteria": plan.success_criteria,
                }
            except Exception as error:
                plan_data = _fallback_plan(normalized, error)
                generation_warnings.append(
                    (
                        f"Planner failed for diversification candidate '{normalized}': {type(error).__name__}; "
                        "using discovery fallback metadata."
                    )
                )
            family, tags = assign_topic_taxonomy(
                query=normalized,
                intent=str(plan_data["intent"]),
                candidate_concepts=list(plan_data["candidateConcepts"]),
                taxonomy_rules=taxonomy_rules,
            )
            topics.append(
                {
                    "seedQuery": seed_query,
                    "query": normalized,
                    "intent": plan_data["intent"],
                    "family": family,
                    "tags": tags,
                    "followUpMode": plan_data["followUpMode"],
                    "providerPlan": plan_data["providerPlan"],
                    "candidateConcepts": plan_data["candidateConcepts"],
                    "constraints": plan_data["constraints"],
                    "successCriteria": plan_data["successCriteria"],
                    "rationale": (
                        f"Single-seed diversification candidate ({brief['angle']}): "
                        f"{candidate.rationale or 'Generated to improve coverage.'}"
                    ),
                    "diversificationAngle": brief["angle"],
                }
            )
            break


def build_scenario_payload(
    topics: list[dict[str, Any]],
    *,
    latency_profile: str,
    emit_follow_up: bool,
) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for index, topic in enumerate(topics, start=1):
        search_name = f"topic_{index}_{_slugify(str(topic['query']))}"
        scenarios.append(
            {
                "name": search_name,
                "tool": "search_papers_smart",
                "arguments": {
                    "query": topic["query"],
                    "latencyProfile": latency_profile,
                },
            }
        )
        if not emit_follow_up:
            continue
        scenarios.append(
            {
                "name": f"{search_name}_follow_up",
                "tool": "ask_result_set",
                "arguments": {
                    "searchSessionId": f"$result.{search_name}.searchSessionId",
                    "question": _follow_up_question(str(topic.get("followUpMode") or "qa")),
                    "topK": 6,
                    "answerMode": topic.get("followUpMode") or "qa",
                    "latencyProfile": latency_profile,
                },
            }
        )
    return {"scenarios": scenarios}


def build_viewer_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        _script_path("view_generated_topics.py"),
        "--input",
        args.output,
        "--host",
        args.viewer_host,
        "--port",
        str(args.viewer_port),
    ]


def build_workflow_command(args: argparse.Namespace) -> list[str]:
    if not args.scenario_output:
        raise ValueError("--launch-workflow requires --scenario-output or --easy-button")
    command = [
        sys.executable,
        _script_path("run_eval_workflow.py"),
        "--scenario-file",
        args.scenario_output,
        "--artifact-root",
        args.artifact_root,
        "--dotenv-path",
        args.dotenv_path,
        "--review-mode",
        args.workflow_review_mode,
    ]
    for preset in args.workflow_matrix_preset or []:
        command.extend(["--matrix-preset", preset])
    if args.workflow_launch_matrix_viewer:
        command.append("--launch-matrix-viewer")
    return command


def _workflow_threshold(args: argparse.Namespace, name: str, default: int | float) -> int | float:
    value = getattr(args, name, default)
    return default if value is None else value


def build_workflow_launch_decision(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    policy = getattr(args, "workflow_autopilot_policy", "review")
    summary = payload.get("summary") or {}
    risk_warnings = list(payload.get("riskWarnings") or [])
    hard_blockers = identify_hard_blockers(summary, args=args) if summary else list(payload.get("hardBlockers") or [])
    thresholds = {
        "safeMinAverageQuality": float(_workflow_threshold(args, "safe_min_average_quality", 45.0)),
        "safeMinTopicCount": int(_workflow_threshold(args, "safe_min_topic_count", 5)),
        "safeMinFamilyCount": int(_workflow_threshold(args, "safe_min_family_count", 4)),
        "safeMinIntentCount": int(_workflow_threshold(args, "safe_min_intent_count", 2)),
        "safeMaxFamilyShare": float(_workflow_threshold(args, "safe_max_family_share", 0.4)),
        "safeMaxRiskWarnings": int(_workflow_threshold(args, "safe_max_risk_warnings", 0)),
        "safeMaxFamilyDisagreementCount": int(_workflow_threshold(args, "safe_max_family_disagreement_count", 0)),
        "safeMaxRewrittenShare": float(_workflow_threshold(args, "safe_max_rewritten_share", 0.35)),
        "hardBlockerMinTopicCount": int(_hard_blocker_threshold(args, "hard_blocker_min_topic_count", 5)),
        "hardBlockerMaxFamilyShare": float(_hard_blocker_threshold(args, "hard_blocker_max_family_share", 0.7)),
        "hardBlockerMaxFamilyDisagreementCount": int(
            _hard_blocker_threshold(args, "hard_blocker_max_family_disagreement_count", 3)
        ),
        "hardBlockerMaxRewrittenShare": float(_hard_blocker_threshold(args, "hard_blocker_max_rewritten_share", 0.5)),
    }
    trace: list[dict[str, Any]] = []

    def _record(check: str, passed: bool, message: str, *, actual: Any = None, expected: Any = None) -> None:
        trace.append(
            {
                "check": check,
                "passed": passed,
                "message": message,
                "actual": actual,
                "expected": expected,
            }
        )

    if not args.launch_workflow:
        _record("launch_requested", False, "workflow launch not requested")
        return {
            "shouldLaunch": False,
            "reason": "workflow launch not requested",
            "policy": policy,
            "thresholds": thresholds,
            "hardBlockers": [],
            "trace": trace,
        }

    _record("launch_requested", True, "workflow launch requested")

    if getattr(args, "force_launch_workflow", False):
        _record("force_override", True, "workflow handoff forced by caller")
        return {
            "shouldLaunch": True,
            "reason": "workflow handoff forced by caller",
            "policy": policy,
            "thresholds": thresholds,
            "hardBlockers": [],
            "trace": trace,
        }

    if policy == "blocked":
        _record("policy", False, "workflow autopilot policy is blocked", actual=policy, expected="review|safe")
        return {
            "shouldLaunch": False,
            "reason": "workflow autopilot policy is blocked",
            "policy": policy,
            "thresholds": thresholds,
            "hardBlockers": [],
            "trace": trace,
        }

    if hard_blockers:
        _record("hard_blockers", False, str(hard_blockers[0]), actual=hard_blockers, expected=[])
        return {
            "shouldLaunch": False,
            "reason": str(hard_blockers[0]),
            "policy": policy,
            "thresholds": thresholds,
            "hardBlockers": hard_blockers,
            "trace": trace,
        }

    _record("hard_blockers", True, "no hard blockers", actual=[], expected=[])

    if policy == "review":
        passed = payload.get("reviewRecommendation") != "human-review-required"
        _record(
            "review_recommendation",
            passed,
            "generated topic pool recommends human review" if not passed else "review recommendation allows handoff",
            actual=payload.get("reviewRecommendation"),
            expected="!= human-review-required",
        )
        return {
            "shouldLaunch": passed,
            "reason": "workflow handoff allowed" if passed else "generated topic pool recommends human review",
            "policy": policy,
            "thresholds": thresholds,
            "hardBlockers": [],
            "trace": trace,
        }

    safe_checks = [
        (
            "review_recommendation",
            payload.get("reviewRecommendation") == "ai-assisted-ok",
            "safe autopilot requires ai-assisted-ok review recommendation",
            payload.get("reviewRecommendation"),
            "ai-assisted-ok",
        ),
        (
            "topic_count",
            int(summary.get("topicCount") or 0) >= thresholds["safeMinTopicCount"],
            f"safe autopilot requires at least {thresholds['safeMinTopicCount']} topics",
            int(summary.get("topicCount") or 0),
            f">= {thresholds['safeMinTopicCount']}",
        ),
        (
            "average_quality",
            float(summary.get("averageQualityScore") or 0.0) >= thresholds["safeMinAverageQuality"],
            f"safe autopilot requires average quality score of at least {thresholds['safeMinAverageQuality']}",
            float(summary.get("averageQualityScore") or 0.0),
            f">= {thresholds['safeMinAverageQuality']}",
        ),
        (
            "family_count",
            int(summary.get("familyCount") or 0) >= thresholds["safeMinFamilyCount"],
            f"safe autopilot requires at least {thresholds['safeMinFamilyCount']} represented families",
            int(summary.get("familyCount") or 0),
            f">= {thresholds['safeMinFamilyCount']}",
        ),
        (
            "intent_count",
            int(summary.get("intentCount") or 0) >= thresholds["safeMinIntentCount"],
            f"safe autopilot requires at least {thresholds['safeMinIntentCount']} represented intents",
            int(summary.get("intentCount") or 0),
            f">= {thresholds['safeMinIntentCount']}",
        ),
        (
            "max_family_share",
            float(summary.get("maxFamilyShare") or 1.0) <= thresholds["safeMaxFamilyShare"],
            f"safe autopilot requires no family share above {thresholds['safeMaxFamilyShare']}",
            float(summary.get("maxFamilyShare") or 1.0),
            f"<= {thresholds['safeMaxFamilyShare']}",
        ),
        (
            "risk_warning_count",
            len(risk_warnings) <= thresholds["safeMaxRiskWarnings"],
            f"safe autopilot requires no more than {thresholds['safeMaxRiskWarnings']} risk warnings",
            len(risk_warnings),
            f"<= {thresholds['safeMaxRiskWarnings']}",
        ),
        (
            "family_disagreement_count",
            int(summary.get("familyDisagreementCount") or 0) <= thresholds["safeMaxFamilyDisagreementCount"],
            (
                "safe autopilot requires family cross-check disagreements to stay at or below "
                f"{thresholds['safeMaxFamilyDisagreementCount']}"
            ),
            int(summary.get("familyDisagreementCount") or 0),
            f"<= {thresholds['safeMaxFamilyDisagreementCount']}",
        ),
        (
            "rewritten_share",
            float(summary.get("rewrittenShare") or 0.0) <= thresholds["safeMaxRewrittenShare"],
            f"safe autopilot requires rewritten share at or below {thresholds['safeMaxRewrittenShare']}",
            float(summary.get("rewrittenShare") or 0.0),
            f"<= {thresholds['safeMaxRewrittenShare']}",
        ),
    ]

    failure_reason = "workflow handoff allowed"
    should_launch = True
    for check, passed, message, actual, expected in safe_checks:
        _record(check, passed, message if not passed else f"{check} passed", actual=actual, expected=expected)
        if not passed and should_launch:
            should_launch = False
            failure_reason = message

    return {
        "shouldLaunch": should_launch,
        "reason": failure_reason,
        "policy": policy,
        "thresholds": thresholds,
        "hardBlockers": [],
        "trace": trace,
    }


def should_launch_workflow(args: argparse.Namespace, payload: dict[str, Any]) -> bool:
    return bool(build_workflow_launch_decision(args, payload)["shouldLaunch"])


def workflow_suppression_reason(args: argparse.Namespace, payload: dict[str, Any]) -> str:
    return str(build_workflow_launch_decision(args, payload)["reason"])


async def generate_topics(args: argparse.Namespace) -> dict[str, Any]:
    settings = _settings_from_env(Path(args.dotenv_path))
    config = AgenticConfig.from_settings(settings)
    provider_bundle = resolve_provider_bundle(
        config,
        openai_api_key=settings.openai_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
        openrouter_http_referer=settings.openrouter_http_referer,
        openrouter_title=settings.openrouter_title,
        azure_openai_api_key=settings.azure_openai_api_key,
        azure_openai_endpoint=settings.azure_openai_endpoint,
        azure_openai_api_version=settings.azure_openai_api_version,
        azure_openai_planner_deployment=settings.azure_openai_planner_deployment,
        azure_openai_synthesis_deployment=settings.azure_openai_synthesis_deployment,
        anthropic_api_key=settings.anthropic_api_key,
        nvidia_api_key=settings.nvidia_api_key,
        nvidia_nim_base_url=settings.nvidia_nim_base_url,
        google_api_key=settings.google_api_key,
        mistral_api_key=settings.mistral_api_key,
        huggingface_api_key=settings.huggingface_api_key,
        huggingface_base_url=settings.huggingface_base_url,
    )

    topics: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    generation_warnings: list[str] = []
    taxonomy_path = resolve_taxonomy_file(args)
    taxonomy_rules = load_taxonomy_rules(str(taxonomy_path))
    context_texts = [normalize_query(text) for text in (args.context_text or []) if normalize_query(text)]
    seed_queries = load_seed_queries(args)
    for seed_query in seed_queries:
        try:
            expansions = await provider_bundle.asuggest_speculative_expansions(
                query=seed_query,
                evidence_texts=context_texts,
                max_variants=args.max_variants,
            )
        except Exception as error:
            expansions = []
            generation_warnings.append(
                (
                    f"Speculative expansion failed for '{seed_query}': {type(error).__name__}; "
                    "falling back to literal seed."
                )
            )
        candidate_queries = [candidate.variant for candidate in expansions]
        if args.include_original:
            candidate_queries.insert(0, seed_query)
        if not candidate_queries:
            candidate_queries = [seed_query]
        for candidate_query in candidate_queries:
            normalized = normalize_query(candidate_query)
            lowered = normalized.lower()
            if not normalized or lowered in seen_queries:
                continue
            seen_queries.add(lowered)
            try:
                plan = await provider_bundle.aplan_search(query=normalized, mode="auto")
                plan_data = {
                    "intent": plan.intent,
                    "followUpMode": plan.follow_up_mode,
                    "providerPlan": plan.provider_plan,
                    "candidateConcepts": plan.candidate_concepts,
                    "constraints": plan.constraints,
                    "successCriteria": plan.success_criteria,
                }
            except Exception as error:
                plan_data = _fallback_plan(normalized, error)
                generation_warnings.append(
                    f"Planner failed for '{normalized}': {type(error).__name__}; using discovery fallback metadata."
                )
            family, tags = assign_topic_taxonomy(
                query=normalized,
                intent=str(plan_data["intent"]),
                candidate_concepts=list(plan_data["candidateConcepts"]),
                taxonomy_rules=taxonomy_rules,
            )
            topics.append(
                {
                    "seedQuery": seed_query,
                    "query": normalized,
                    "intent": plan_data["intent"],
                    "family": family,
                    "tags": tags,
                    "followUpMode": plan_data["followUpMode"],
                    "providerPlan": plan_data["providerPlan"],
                    "candidateConcepts": plan_data["candidateConcepts"],
                    "constraints": plan_data["constraints"],
                    "successCriteria": plan_data["successCriteria"],
                    "rationale": next(
                        (
                            candidate.rationale
                            for candidate in expansions
                            if normalize_query(candidate.variant).lower() == lowered
                        ),
                        (
                            plan_data["rationale"]
                            if "rationale" in plan_data
                            else "Literal seed query."
                            if normalized == seed_query
                            else "Generated topic candidate."
                        ),
                    ),
                }
            )

    await maybe_add_single_seed_diversifications(
        args=args,
        provider_bundle=provider_bundle,
        taxonomy_rules=taxonomy_rules,
        context_texts=context_texts,
        seed_queries=seed_queries,
        topics=topics,
        seen_queries=seen_queries,
        generation_warnings=generation_warnings,
    )

    selection_metadata = provider_bundle.selection_metadata()
    payload = {
        "seedQueries": seed_queries,
        "latencyProfile": args.latency_profile,
        "configuredSmartProvider": selection_metadata.get("configuredSmartProvider"),
        "activeSmartProvider": selection_metadata.get("activeSmartProvider"),
        "plannerModel": selection_metadata.get("plannerModel"),
        "taxonomyPreset": args.taxonomy_preset,
        "taxonomyFile": str(taxonomy_path),
        "generationWarnings": generation_warnings,
        "topics": topics,
    }
    merged_payload = merge_generated_topic_payloads(payload, args.merge_inputs)
    scored_topics = rank_topics(
        merged_payload["topics"],
        min_quality_score=0.0,
        max_topics=None,
    )
    pruned_topics, prune_summary = await ai_assisted_prune_topics(
        scored_topics,
        mode=args.ai_prune_mode,
        prune_below_score=args.ai_prune_below_score,
        provider_bundle=provider_bundle,
        taxonomy_rules=taxonomy_rules,
        context_texts=context_texts,
    )
    final_ranked = rank_topics(
        pruned_topics,
        min_quality_score=args.min_quality_score,
        max_topics=None,
    )
    balanced_topics, balance_summary = rebalance_topics(
        final_ranked,
        mode=args.domain_balance_mode,
        max_topics=args.max_topics,
        max_family_share=args.domain_balance_max_share,
    )
    cross_checked_topics, cross_check_summary = annotate_family_cross_checks(balanced_topics, taxonomy_rules)
    merged_payload["topics"] = cross_checked_topics
    merged_payload["pruneSummary"] = prune_summary
    merged_payload["balanceSummary"] = balance_summary
    merged_payload["familyCrossCheckSummary"] = cross_check_summary
    merged_payload["summary"] = add_prune_metrics_to_summary(
        summarize_topic_pool(merged_payload["topics"]),
        prune_summary,
    )
    merged_payload["hardBlockers"] = identify_hard_blockers(merged_payload["summary"], args=args)
    risk_warnings, review_recommendation = assess_topic_pool_risks(merged_payload["summary"])
    merged_payload["riskWarnings"] = risk_warnings
    merged_payload["reviewRecommendation"] = review_recommendation
    await provider_bundle.aclose()
    return merged_payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_easy_button_defaults(args)
    if args.launch_viewer and args.output == "-":
        parser.error("--launch-viewer requires --output to be a file path.")
    if not args.output:
        parser.error("--output is required unless --easy-button provides a default path.")
    payload = asyncio.run(generate_topics(args))
    rendered = json.dumps(payload, indent=2)
    if args.output == "-":
        sys.stdout.write(rendered)
        sys.stdout.write("\n")
    else:
        Path(args.output).write_text(rendered, encoding="utf-8")
    if args.jsonl_output:
        write_topics_jsonl(Path(args.jsonl_output), payload["topics"])
    if args.csv_output:
        write_topics_csv(Path(args.csv_output), payload["topics"])
    if args.markdown_output:
        write_topics_markdown(Path(args.markdown_output), payload["topics"])
    if args.scenario_output:
        scenario_payload = build_scenario_payload(
            payload["topics"],
            latency_profile=args.latency_profile,
            emit_follow_up=args.emit_follow_up,
        )
        Path(args.scenario_output).write_text(json.dumps(scenario_payload, indent=2), encoding="utf-8")
    if args.launch_workflow:
        if not should_launch_workflow(args, payload):
            print(f"Workflow handoff suppressed: {workflow_suppression_reason(args, payload)}.")
            print(json.dumps(payload.get("riskWarnings") or [], indent=2))
            if not args.launch_viewer:
                subprocess.run(build_viewer_command(args), check=True)  # nosec B603 - fixed local script entrypoint only
        else:
            subprocess.run(build_workflow_command(args), check=True)  # nosec B603 - fixed local script entrypoint only
    elif args.launch_viewer:
        subprocess.run(build_viewer_command(args), check=True)  # nosec B603 - fixed local script entrypoint only
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
