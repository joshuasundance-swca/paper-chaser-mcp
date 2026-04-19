"""Run profile-driven eval bootstrap and workflow handoff with immutable run bundles."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess  # nosec B404 - trusted local CLI wrapper for local repo scripts
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

DEFAULT_PROFILE_FILE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "evals" / "eval-autopilot-profiles.sample.json"
)


def _script_path(name: str) -> str:
    return str(Path(__file__).resolve().parent / name)


def _load_generate_module():
    path = Path(__file__).resolve().parent / "generate_eval_topics.py"
    spec = importlib.util.spec_from_file_location("generate_eval_topics_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load generate_eval_topics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run profile-driven topic generation, safety review, and optional workflow handoff."
    )
    parser.add_argument("--profile", required=True, help="Profile name from the autopilot profile file.")
    parser.add_argument(
        "--profile-file",
        default=str(DEFAULT_PROFILE_FILE),
        help="JSON profile file describing generation and workflow defaults.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(Path(__file__).resolve().parent.parent / "build" / "eval-autopilot"),
        help="Parent directory where immutable run bundles should be written.",
    )
    parser.add_argument(
        "--dotenv-path",
        default=str(Path(__file__).resolve().parent.parent / ".env"),
        help="Path to the shared .env file used for generation and workflow commands.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Write the run bundle and commands without executing them."
    )
    parser.add_argument("--force-launch-workflow", action="store_true", help="Override autopilot suppression rules.")
    parser.add_argument(
        "--matrix-preset",
        action="append",
        default=None,
        help="Optional matrix preset override(s) for workflow handoff.",
    )
    parser.add_argument("--seed-query", action="append", default=None, help="Optional seed query override.")
    parser.add_argument("--seed-file", default=None, help="Optional seed file override.")
    parser.add_argument(
        "--workflow-autopilot-policy",
        choices=["safe", "review", "blocked"],
        default=None,
        help="Optional autopilot policy override.",
    )
    parser.add_argument(
        "--slice",
        dest="slice_name",
        choices=["general", "env-sci"],
        default="general",
        help=(
            "Benchmark slice selector. 'general' (default) preserves the existing "
            "autopilot behavior. 'env-sci' additionally loads the environmental-science "
            "benchmark pack and judge-rubric template so downstream judges (if wired) "
            "can score env-sci-specific roles."
        ),
    )
    parser.add_argument(
        "--env-sci-pack",
        default=None,
        help="Override path to the env-sci benchmark pack JSON (only used when --slice env-sci).",
    )
    parser.add_argument(
        "--env-sci-rubric",
        default=None,
        help="Override path to the env-sci judge-rubric template JSON (only used when --slice env-sci).",
    )
    return parser


def load_profiles(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(profiles, dict):
        raise ValueError("profile file must contain a top-level 'profiles' object")
    return profiles


def resolve_profile(profiles: dict[str, Any], name: str) -> dict[str, Any]:
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        raise ValueError(f"Unknown autopilot profile: {name}")
    return profile


def build_run_directory(artifact_root: Path, profile_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_profile = profile_name.replace("/", "-").replace(" ", "-")
    run_dir = artifact_root / f"{timestamp}-{safe_profile}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "run_dir": run_dir,
        "generated_json": run_dir / "generated-topics.json",
        "generated_jsonl": run_dir / "generated-topics.jsonl",
        "generated_csv": run_dir / "generated-topics.csv",
        "generated_markdown": run_dir / "generated-topics.md",
        "scenario_json": run_dir / "generated-batch.json",
        "manifest_json": run_dir / "run-manifest.json",
        "report_json": run_dir / "autopilot-report.json",
        "report_markdown": run_dir / "autopilot-report.md",
        "holdout_report_json": run_dir / "holdout-report.json",
        "state_json": run_dir / "run-state.json",
        "generate_log": run_dir / "generate.log",
        "workflow_log": run_dir / "workflow.log",
        "workflow_artifacts": run_dir / "workflow",
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_state(path: Path, state: dict[str, Any]) -> None:
    state["updatedAt"] = _timestamp()
    _write_json(path, state)


def _normalize_overlap_text(text: str) -> str:
    lowered = text.lower()
    chars = [character if character.isalnum() else " " for character in lowered]
    return " ".join("".join(chars).split())


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_normalize_overlap_text(left).split())
    right_tokens = set(_normalize_overlap_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _sequence_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize_overlap_text(left), _normalize_overlap_text(right)).ratio()


def build_generate_command(args: argparse.Namespace, profile: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    generation = dict(profile.get("generation") or {})
    command = [
        sys.executable,
        _script_path("generate_eval_topics.py"),
        "--dotenv-path",
        args.dotenv_path,
        "--output",
        str(paths["generated_json"]),
        "--jsonl-output",
        str(paths["generated_jsonl"]),
        "--csv-output",
        str(paths["generated_csv"]),
        "--markdown-output",
        str(paths["generated_markdown"]),
        "--scenario-output",
        str(paths["scenario_json"]),
        "--taxonomy-preset",
        str(generation.get("taxonomyPreset") or "balanced-science"),
        "--latency-profile",
        str(generation.get("latencyProfile") or "balanced"),
        "--ai-prune-mode",
        str(generation.get("aiPruneMode") or "rewrite-or-drop"),
        "--ai-prune-below-score",
        str(generation.get("aiPruneBelowScore") or 35.0),
        "--domain-balance-mode",
        str(generation.get("domainBalanceMode") or "round-robin"),
        "--domain-balance-max-share",
        str(generation.get("domainBalanceMaxShare") or 0.4),
        "--min-quality-score",
        str(generation.get("minQualityScore") or 30.0),
    ]
    if generation.get("maxTopics") is not None:
        command.extend(["--max-topics", str(generation["maxTopics"])])
    if generation.get("maxVariants") is not None:
        command.extend(["--max-variants", str(generation["maxVariants"])])
    if generation.get("includeOriginal", True):
        command.append("--include-original")
    if generation.get("emitFollowUp", True):
        command.append("--emit-follow-up")
    if bool(generation.get("singleSeedDiversification")):
        command.append("--single-seed-diversification")
    for context_text in generation.get("contextText") or []:
        command.extend(["--context-text", str(context_text)])
    if args.seed_file:
        command.extend(["--seed-file", args.seed_file])
    elif generation.get("seedFile"):
        command.extend(["--seed-file", str(generation["seedFile"])])
    elif generation.get("seedPreset"):
        command.extend(["--seed-preset", str(generation["seedPreset"])])
    for seed_query in args.seed_query or generation.get("seedQueries") or []:
        command.extend(["--seed-query", str(seed_query)])
    return command


def build_workflow_command(args: argparse.Namespace, profile: dict[str, Any], paths: dict[str, Path]) -> list[str]:
    workflow = dict(profile.get("workflow") or {})
    review_mode = str(workflow.get("reviewMode") or "yolo")
    command = [
        sys.executable,
        _script_path("run_eval_workflow.py"),
        "--scenario-file",
        str(paths["scenario_json"]),
        "--artifact-root",
        str(paths["workflow_artifacts"]),
        "--dotenv-path",
        args.dotenv_path,
        "--review-mode",
        review_mode,
    ]
    matrix_presets = args.matrix_preset or workflow.get("matrixPreset") or []
    for preset in matrix_presets:
        command.extend(["--matrix-preset", str(preset)])
    if bool(workflow.get("launchMatrixViewer")):
        command.append("--launch-matrix-viewer")
    return command


def build_holdout_report(
    profile: dict[str, Any], generated_payload: dict[str, Any], profile_file: Path
) -> dict[str, Any]:
    holdout_path = profile.get("holdoutSeedFile")
    if not holdout_path:
        return {"enabled": False}
    path = Path(str(holdout_path))
    if not path.is_absolute():
        path = (profile_file.parent / path).resolve()
    seeds = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    generated_queries = [str(topic.get("query") or "").strip() for topic in generated_payload.get("topics") or []]
    generated_query_set = {query.lower() for query in generated_queries}
    exact_overlap = [seed for seed in seeds if seed.lower() in generated_query_set]
    near_overlap: list[dict[str, Any]] = []
    for seed in seeds:
        best_match: dict[str, Any] | None = None
        for query in generated_queries:
            token_jaccard = _token_jaccard(seed, query)
            sequence_ratio = _sequence_ratio(seed, query)
            if token_jaccard < 0.5 and sequence_ratio < 0.82:
                continue
            candidate = {
                "holdoutSeed": seed,
                "generatedQuery": query,
                "tokenJaccard": round(token_jaccard, 3),
                "sequenceRatio": round(sequence_ratio, 3),
            }
            if best_match is None or (
                candidate["tokenJaccard"],
                candidate["sequenceRatio"] > (best_match["tokenJaccard"], best_match["sequenceRatio"]),
            ):
                best_match = candidate
        if best_match is not None:
            near_overlap.append(best_match)
    return {
        "enabled": True,
        "holdoutSeedFile": str(path),
        "holdoutSeedCount": len(seeds),
        "exactOverlapCount": len(exact_overlap),
        "exactOverlap": exact_overlap,
        "nearOverlapCount": len(near_overlap),
        "nearOverlap": near_overlap,
    }


def build_autopilot_report(
    *,
    profile_name: str,
    profile: dict[str, Any],
    generated_payload: dict[str, Any],
    workflow_decision: dict[str, Any],
    holdout_report: dict[str, Any],
    paths: dict[str, Path],
    commands: dict[str, list[str]],
    command_statuses: dict[str, Any],
) -> dict[str, Any]:
    return {
        "profile": profile_name,
        "description": profile.get("description"),
        "forceLaunchWorkflow": bool((profile.get("workflow") or {}).get("forceLaunch")),
        "reviewRecommendation": generated_payload.get("reviewRecommendation"),
        "riskWarnings": generated_payload.get("riskWarnings") or [],
        "hardBlockers": workflow_decision.get("hardBlockers") or [],
        "generationWarnings": generated_payload.get("generationWarnings") or [],
        "summary": generated_payload.get("summary") or {},
        "pruneSummary": generated_payload.get("pruneSummary") or {},
        "balanceSummary": generated_payload.get("balanceSummary") or {},
        "familyCrossCheckSummary": generated_payload.get("familyCrossCheckSummary") or {},
        "holdoutReport": holdout_report,
        "workflowDecision": workflow_decision,
        "shouldLaunchWorkflow": workflow_decision.get("shouldLaunch"),
        "suppressionReason": workflow_decision.get("reason"),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "run_dir"},
        "commands": commands,
        "commandStatuses": command_statuses,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    decision = report.get("workflowDecision") or {}
    lines = [
        "# Eval Autopilot Report",
        "",
        f"- Profile: {report['profile']}",
        f"- Review recommendation: {report['reviewRecommendation']}",
        f"- Workflow launch: {'yes' if report['shouldLaunchWorkflow'] else 'no'}",
        f"- Reason: {report['suppressionReason']}",
        f"- Policy: {decision.get('policy')}",
        "",
        "## Risk Signals",
        "",
        f"- Risk warnings: {len(report.get('riskWarnings') or [])}",
        f"- Hard blockers: {len(report.get('hardBlockers') or [])}",
        f"- Generation warnings: {len(report.get('generationWarnings') or [])}",
        "",
        "## Summary",
        "",
        f"- Topic count: {report['summary'].get('topicCount')}",
        f"- Average quality: {report['summary'].get('averageQualityScore')}",
        f"- Family count: {report['summary'].get('familyCount')}",
        f"- Rewritten share: {report['summary'].get('rewrittenShare')}",
        "",
        "## Workflow Decision Trace",
        "",
    ]
    for item in decision.get("trace") or []:
        lines.append(f"- [{'pass' if item.get('passed') else 'fail'}] {item.get('check')}: {item.get('message')}")
    lines.extend(
        [
            "",
            "## Command Status",
            "",
        ]
    )
    for name, status in (report.get("commandStatuses") or {}).items():
        lines.append(f"- {name}: {status.get('status')} (exit={status.get('exitCode')}, log={status.get('logPath')})")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, path in report.get("artifacts", {}).items():
        lines.append(f"- {name}: {path}")
    lines.extend(["", "## Commands", ""])
    for name, command in report.get("commands", {}).items():
        lines.append(f"- {name}: {' '.join(command)}")
    return "\n".join(lines) + "\n"


def run_command(
    command: list[str],
    *,
    dry_run: bool,
    log_path: Path,
    stage_name: str,
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "stage": stage_name,
        "status": "dry-run" if dry_run else "pending",
        "exitCode": 0,
        "command": command,
        "logPath": str(log_path),
        "startedAt": _timestamp(),
        "completedAt": None,
    }
    state.setdefault("commandStatuses", {})[stage_name] = status
    state["currentStage"] = stage_name
    write_run_state(state_path, state)
    if dry_run:
        status["completedAt"] = _timestamp()
        write_run_state(state_path, state)
        return status

    print(f"starting {stage_name}; log: {log_path}")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n\n")
        try:
            completed = subprocess.run(  # nosec B603 - fixed local script entrypoints only
                command,
                check=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            status["status"] = "completed"
            status["exitCode"] = int(completed.returncode)
        except subprocess.CalledProcessError as error:
            status["status"] = "failed"
            status["exitCode"] = int(error.returncode)
            status["error"] = f"{type(error).__name__}: exit code {error.returncode}"
        finally:
            status["completedAt"] = _timestamp()
            write_run_state(state_path, state)
    print(f"completed {stage_name}; status: {status['status']}; exit: {status['exitCode']}")
    return status


def resolve_slice_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve slice-aware metadata for the run manifest.

    The ``general`` slice (default) preserves pre-existing autopilot behavior
    by emitting a minimal metadata block. The ``env-sci`` slice additionally
    loads the environmental-science benchmark pack and judge-rubric template
    so downstream judge harnesses can pick them up without having to re-discover
    the fixture paths.
    """

    slice_name = str(getattr(args, "slice_name", "general") or "general")
    metadata: dict[str, Any] = {"name": slice_name}
    if slice_name != "env-sci":
        return metadata

    # Import lazily so the general path does not require paper_chaser_mcp on sys.path.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from paper_chaser_mcp.eval_curation import (  # noqa: WPS433 - local import by design
        DEFAULT_ENV_SCI_BENCHMARK_PACK,
        DEFAULT_ENV_SCI_JUDGE_RUBRIC,
        ENV_SCI_JUDGE_ROLES,
        load_env_sci_benchmark_pack,
        load_env_sci_judge_rubric_template,
    )

    pack_path = Path(getattr(args, "env_sci_pack", None) or DEFAULT_ENV_SCI_BENCHMARK_PACK)
    rubric_path = Path(getattr(args, "env_sci_rubric", None) or DEFAULT_ENV_SCI_JUDGE_RUBRIC)
    pack = load_env_sci_benchmark_pack(pack_path)
    rubric = load_env_sci_judge_rubric_template(rubric_path)
    metadata.update(
        {
            "benchmarkPackPath": str(pack_path),
            "judgeRubricPath": str(rubric_path),
            "rowCount": len(pack.get("rows") or []),
            "failureFamilies": list(pack.get("failureFamilies") or []),
            "judgeRoles": list(ENV_SCI_JUDGE_ROLES),
            "judgeRoleIds": [role.get("id") for role in rubric.get("roles") or [] if isinstance(role, dict)],
        }
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    profile_file = Path(args.profile_file)
    profiles = load_profiles(profile_file)
    profile = resolve_profile(profiles, args.profile)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_dir = build_run_directory(artifact_root, args.profile)
    paths = build_run_paths(run_dir)
    state = {
        "profile": args.profile,
        "profileFile": str(profile_file),
        "dryRun": args.dry_run,
        "status": "running",
        "startedAt": _timestamp(),
        "currentStage": "initializing",
        "commandStatuses": {},
    }
    write_run_state(paths["state_json"], state)

    generate_command = build_generate_command(args, profile, paths)
    workflow_command = build_workflow_command(args, profile, paths)
    commands = {"generate": generate_command, "workflow": workflow_command}
    slice_metadata = resolve_slice_metadata(args)
    manifest_payload: dict[str, Any] = {
        "profile": args.profile,
        "profileFile": str(profile_file),
        "dryRun": args.dry_run,
        "commands": commands,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "slice": slice_metadata,
    }
    _write_json(paths["manifest_json"], manifest_payload)

    generate_status = run_command(
        generate_command,
        dry_run=args.dry_run,
        log_path=paths["generate_log"],
        stage_name="generate",
        state_path=paths["state_json"],
        state=state,
    )

    generated_payload = (
        json.loads(paths["generated_json"].read_text(encoding="utf-8"))
        if paths["generated_json"].exists()
        else {
            "reviewRecommendation": "unknown",
            "riskWarnings": [],
            "hardBlockers": [],
            "generationWarnings": [],
            "summary": {},
        }
    )

    generate_module = _load_generate_module()
    workflow_policy = args.workflow_autopilot_policy or str(
        (profile.get("workflow") or {}).get("autopilotPolicy") or "safe"
    )
    force_launch = bool(args.force_launch_workflow or (profile.get("workflow") or {}).get("forceLaunch"))
    workflow_thresholds = dict((profile.get("workflow") or {}).get("thresholds") or {})
    launch_args = argparse.Namespace(
        launch_workflow=True,
        force_launch_workflow=force_launch,
        workflow_autopilot_policy=workflow_policy,
        safe_min_average_quality=workflow_thresholds.get("safeMinAverageQuality"),
        safe_min_topic_count=workflow_thresholds.get("safeMinTopicCount"),
        safe_min_family_count=workflow_thresholds.get("safeMinFamilyCount"),
        safe_min_intent_count=workflow_thresholds.get("safeMinIntentCount"),
        safe_max_family_share=workflow_thresholds.get("safeMaxFamilyShare"),
        safe_max_risk_warnings=workflow_thresholds.get("safeMaxRiskWarnings"),
        safe_max_family_disagreement_count=workflow_thresholds.get("safeMaxFamilyDisagreementCount"),
        safe_max_rewritten_share=workflow_thresholds.get("safeMaxRewrittenShare"),
        hard_blocker_min_topic_count=workflow_thresholds.get("hardBlockerMinTopicCount"),
        hard_blocker_max_family_share=workflow_thresholds.get("hardBlockerMaxFamilyShare"),
        hard_blocker_max_family_disagreement_count=workflow_thresholds.get("hardBlockerMaxFamilyDisagreementCount"),
        hard_blocker_max_rewritten_share=workflow_thresholds.get("hardBlockerMaxRewrittenShare"),
    )
    workflow_decision = generate_module.build_workflow_launch_decision(launch_args, generated_payload)
    holdout_report = build_holdout_report(profile, generated_payload, profile_file)

    report = build_autopilot_report(
        profile_name=args.profile,
        profile=profile,
        generated_payload=generated_payload,
        workflow_decision=workflow_decision,
        holdout_report=holdout_report,
        paths=paths,
        commands=commands,
        command_statuses=state["commandStatuses"],
    )
    paths["holdout_report_json"].write_text(json.dumps(holdout_report, indent=2), encoding="utf-8")
    paths["report_json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["report_markdown"].write_text(render_markdown_report(report), encoding="utf-8")
    state["reportPath"] = str(paths["report_json"])
    if generate_status.get("status") == "failed":
        state["status"] = "failed"
        state["currentStage"] = "generate"
        write_run_state(paths["state_json"], state)
        print(f"autopilot run directory: {run_dir}")
        print(f"autopilot report: {paths['report_json']}")
        print("workflow launch: no")
        print(f"reason: generate stage failed with exit code {generate_status.get('exitCode')}")
        return int(generate_status.get("exitCode") or 1)

    workflow_status = run_command(
        workflow_command,
        dry_run=args.dry_run or not bool(workflow_decision.get("shouldLaunch")),
        log_path=paths["workflow_log"],
        stage_name="workflow",
        state_path=paths["state_json"],
        state=state,
    )
    report["commandStatuses"] = state["commandStatuses"]
    paths["report_json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["report_markdown"].write_text(render_markdown_report(report), encoding="utf-8")
    state["status"] = "failed" if workflow_status.get("status") == "failed" else "completed"
    state["currentStage"] = "done"
    write_run_state(paths["state_json"], state)

    print(f"autopilot run directory: {run_dir}")
    print(f"autopilot report: {paths['report_json']}")
    print(f"workflow launch: {'yes' if workflow_decision.get('shouldLaunch') and not args.dry_run else 'no'}")
    print(f"reason: {workflow_decision.get('reason')}")
    if workflow_status.get("status") == "failed":
        return int(workflow_status.get("exitCode") or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
