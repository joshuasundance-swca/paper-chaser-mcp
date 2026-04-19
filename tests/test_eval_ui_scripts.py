import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEW_SCRIPT = REPO_ROOT / "scripts" / "review_eval_traces.py"
MATRIX_SCRIPT = REPO_ROOT / "scripts" / "view_eval_matrix.py"
TOPIC_SCRIPT = REPO_ROOT / "scripts" / "view_generated_topics.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_review_rows_counts_promoted_and_families() -> None:
    module = _load_module(REVIEW_SCRIPT, "review_eval_traces_script")
    rows = [
        {
            "trace": {"source_tool": "research"},
            "review": {"promote": True, "task_family": "planner"},
        },
        {
            "trace": {"source_tool": "follow_up_research"},
            "review": {"promote": False, "task_family": "synthesis"},
        },
    ]

    summary = module.summarize_review_rows(rows)

    assert summary["rowCount"] == 2
    assert summary["promotedCount"] == 1
    assert summary["pendingCount"] == 1
    assert summary["families"]["planner"] == 1
    assert summary["tools"]["research"] == 1


def test_summarize_matrix_results_counts_scenario_statuses() -> None:
    module = _load_module(MATRIX_SCRIPT, "view_eval_matrix_script")
    payload = {
        "scenarios": [
            {
                "name": "openai-live",
                "exitCode": 0,
                "family": "all",
                "env": {"PAPER_CHASER_AGENTIC_PROVIDER": "openai"},
                "report": {
                    "summary": {"itemCount": 2, "failedItems": 0, "warningItems": 1},
                    "liveSummary": {"failed": 0},
                },
            },
            {
                "name": "nvidia-live",
                "exitCode": 1,
                "family": "all",
                "env": {"PAPER_CHASER_AGENTIC_PROVIDER": "nvidia"},
                "report": {
                    "summary": {"itemCount": 2, "failedItems": 1, "warningItems": 0},
                    "liveSummary": {"failed": 1},
                },
            },
        ]
    }

    summary = module.summarize_matrix_results(payload)

    assert summary["scenarioCount"] == 2
    assert summary["passedScenarios"] == 1
    assert summary["failedScenarios"] == 1
    assert summary["scenarios"][0]["provider"] == "openai"
    assert summary["scenarios"][1]["status"] == "failed"


def test_summarize_matrix_divergences_highlights_item_level_differences() -> None:
    module = _load_module(MATRIX_SCRIPT, "view_eval_matrix_script")
    payload = {
        "scenarios": [
            {
                "name": "openai-live",
                "exitCode": 0,
                "env": {"PAPER_CHASER_AGENTIC_PROVIDER": "openai"},
                "report": {
                    "items": [
                        {
                            "id": "planner_item_1",
                            "family": "planner",
                            "status": "passed",
                            "errors": [],
                            "warnings": [],
                            "liveEvaluation": {"status": "passed", "observedIntent": "discovery"},
                        }
                    ]
                },
            },
            {
                "name": "nvidia-live",
                "exitCode": 0,
                "env": {"PAPER_CHASER_AGENTIC_PROVIDER": "nvidia"},
                "report": {
                    "items": [
                        {
                            "id": "planner_item_1",
                            "family": "planner",
                            "status": "passed",
                            "errors": [],
                            "warnings": [],
                            "liveEvaluation": {"status": "passed", "observedIntent": "known_item"},
                        }
                    ]
                },
            },
        ]
    }

    summary = module.summarize_matrix_divergences(payload)

    assert summary["itemCount"] == 1
    assert summary["divergentItemCount"] == 1
    assert summary["divergences"][0]["itemId"] == "planner_item_1"
    assert "observed intent differs" in summary["divergences"][0]["divergenceReasons"]


def test_summarize_matrix_divergences_detects_trace_lineage_changes() -> None:
    module = _load_module(MATRIX_SCRIPT, "view_eval_matrix_script")
    payload = {
        "scenarios": [
            {
                "name": "provider-a",
                "exitCode": 0,
                "env": {"PAPER_CHASER_AGENTIC_PROVIDER": "openai"},
                "report": {
                    "items": [
                        {
                            "id": "planner_item_1",
                            "family": "planner",
                            "status": "passed",
                            "errors": [],
                            "warnings": [],
                            "liveEvaluation": {
                                "status": "passed",
                                "observedIntent": "discovery",
                                "executedSteps": [
                                    {"tool": "research", "searchSessionId": "ssn_one"},
                                ],
                            },
                        }
                    ]
                },
            },
            {
                "name": "provider-b",
                "exitCode": 0,
                "env": {"PAPER_CHASER_AGENTIC_PROVIDER": "nvidia"},
                "report": {
                    "items": [
                        {
                            "id": "planner_item_1",
                            "family": "planner",
                            "status": "passed",
                            "errors": [],
                            "warnings": [],
                            "liveEvaluation": {
                                "status": "passed",
                                "observedIntent": "discovery",
                                "executedSteps": [
                                    {"tool": "search_papers_smart", "searchSessionId": "ssn_two"},
                                    {"tool": "ask_result_set", "searchSessionId": "ssn_three"},
                                ],
                            },
                        }
                    ]
                },
            },
        ]
    }

    summary = module.summarize_matrix_divergences(payload)

    reasons = summary["divergences"][0]["divergenceReasons"]
    assert "executed tool sequence differs" in reasons
    assert "search session lineage differs" in reasons


def test_summarize_generated_topics_counts_families_and_quality() -> None:
    module = _load_module(TOPIC_SCRIPT, "view_generated_topics_script")
    payload = {
        "summary": {
            "topicCount": 2,
            "averageQualityScore": 50.0,
            "families": {"environmental_remediation": 1, "computer_science": 1},
            "intents": {"review": 1, "discovery": 1},
            "qualityTiers": {"high": 1, "medium": 1},
            "familyCount": 2,
            "intentCount": 2,
        },
        "topics": [
            {
                "query": "PFAS remediation in groundwater",
                "family": "environmental_remediation",
                "intent": "review",
                "qualityScore": 60.0,
                "qualityTier": "high",
            },
            {
                "query": "Graph neural network benchmarks",
                "family": "computer_science",
                "intent": "discovery",
                "qualityScore": 40.0,
                "qualityTier": "medium",
            },
        ],
    }

    summary = module.summarize_generated_topics(payload)

    assert summary["topicCount"] == 2
    assert summary["averageQualityScore"] == 50.0
    assert summary["families"]["environmental_remediation"] == 1
    assert summary["qualityTiers"]["high"] == 1
