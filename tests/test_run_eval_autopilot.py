import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_eval_autopilot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_eval_autopilot_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_holdout_report_resolves_relative_path(tmp_path: Path) -> None:
    module = _load_module()
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("query one\nquery two\n", encoding="utf-8")
    profile_file = tmp_path / "profiles.json"
    profile_file.write_text('{"profiles": {}}', encoding="utf-8")

    report = module.build_holdout_report(
        {"holdoutSeedFile": "holdout.txt"},
        {"topics": [{"query": "query one"}]},
        profile_file,
    )

    assert report["enabled"] is True
    assert report["exactOverlapCount"] == 1


def test_build_holdout_report_detects_near_overlap(tmp_path: Path) -> None:
    module = _load_module()
    holdout = tmp_path / "holdout.txt"
    holdout.write_text("wildfire smoke respiratory outcomes\n", encoding="utf-8")
    profile_file = tmp_path / "profiles.json"
    profile_file.write_text('{"profiles": {}}', encoding="utf-8")

    report = module.build_holdout_report(
        {"holdoutSeedFile": "holdout.txt"},
        {"topics": [{"query": "wildfire smoke exposure and respiratory outcomes"}]},
        profile_file,
    )

    assert report["enabled"] is True
    assert report["nearOverlapCount"] == 1
    assert report["nearOverlap"][0]["generatedQuery"] == "wildfire smoke exposure and respiratory outcomes"


def test_main_dry_run_writes_run_bundle(tmp_path: Path) -> None:
    module = _load_module()
    profile_file = tmp_path / "profiles.json"
    profile_file.write_text(
        json.dumps(
            {
                "profiles": {
                    "demo": {
                        "description": "demo",
                        "generation": {
                            "seedQueries": ["wildfire smoke exposure and respiratory outcomes"],
                            "taxonomyPreset": "balanced-science",
                            "includeOriginal": True,
                            "emitFollowUp": True,
                            "minQualityScore": 30.0,
                        },
                        "workflow": {
                            "reviewMode": "yolo",
                            "autopilotPolicy": "safe",
                            "thresholds": {
                                "safeMinTopicCount": 3,
                                "safeMinFamilyCount": 1,
                                "safeMinIntentCount": 1,
                                "safeMaxFamilyShare": 1.0,
                                "safeMaxRiskWarnings": 0,
                                "safeMaxFamilyDisagreementCount": 0,
                                "safeMaxRewrittenShare": 0.5,
                                "hardBlockerMinTopicCount": 3,
                                "hardBlockerMaxFamilyShare": 1.0,
                                "hardBlockerMaxFamilyDisagreementCount": 1,
                                "hardBlockerMaxRewrittenShare": 0.5,
                            },
                            "matrixPreset": ["cross-provider-lower-bound"],
                        },
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--profile",
            "demo",
            "--profile-file",
            str(profile_file),
            "--artifact-root",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )

    run_dirs = list((tmp_path / "runs").iterdir())
    assert exit_code == 0
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "run-manifest.json").read_text(encoding="utf-8"))
    report = json.loads((run_dirs[0] / "autopilot-report.json").read_text(encoding="utf-8"))
    state = json.loads((run_dirs[0] / "run-state.json").read_text(encoding="utf-8"))
    assert manifest["dryRun"] is True
    assert report["shouldLaunchWorkflow"] is False
    assert report["workflowDecision"]["policy"] == "safe"
    assert report["workflowDecision"]["thresholds"]["hardBlockerMinTopicCount"] == 3
    assert state["status"] == "completed"
    assert report["commandStatuses"]["generate"]["status"] == "dry-run"
    assert (run_dirs[0] / "generate.log").exists() is False


def test_build_generate_command_includes_single_seed_diversification_flag(tmp_path: Path) -> None:
    module = _load_module()
    paths = module.build_run_paths(tmp_path)
    args = type("Args", (), {"dotenv_path": ".env", "seed_file": None, "seed_query": None})()
    profile = {
        "generation": {
            "taxonomyPreset": "balanced-science",
            "singleSeedDiversification": True,
        }
    }

    command = module.build_generate_command(args, profile, paths)

    assert "--single-seed-diversification" in command


def test_force_launch_profile_sets_force_override_in_report(tmp_path: Path) -> None:
    module = _load_module()
    profile_file = tmp_path / "profiles.json"
    profile_file.write_text(
        json.dumps(
            {
                "profiles": {
                    "demo": {
                        "description": "demo",
                        "generation": {
                            "seedQueries": ["wildfire smoke exposure and respiratory outcomes"],
                            "taxonomyPreset": "balanced-science",
                        },
                        "workflow": {
                            "autopilotPolicy": "safe",
                            "forceLaunch": True,
                        },
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--profile",
            "demo",
            "--profile-file",
            str(profile_file),
            "--artifact-root",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )

    run_dirs = list((tmp_path / "runs").iterdir())
    report = json.loads((run_dirs[0] / "autopilot-report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["forceLaunchWorkflow"] is True
    assert report["workflowDecision"]["reason"] == "workflow handoff forced by caller"
