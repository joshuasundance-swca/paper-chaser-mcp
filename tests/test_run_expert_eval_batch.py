import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_expert_eval_batch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_expert_eval_batch_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_batch_environment_forces_expert_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    args = argparse.Namespace(
        scenario_file="tests/fixtures/evals/expert-batch.sample.json",
        capture_path="build/eval-curation/captured-events.jsonl",
    )

    monkeypatch.setenv("PAPER_CHASER_TOOL_PROFILE", "guided")
    batch_id, run_id = module._prepare_batch_environment(args)

    assert batch_id.startswith("batch_")
    assert run_id.startswith("run_")
    assert module.os.environ["PAPER_CHASER_TOOL_PROFILE"] == "expert"
    assert module.os.environ["PAPER_CHASER_HIDE_DISABLED_TOOLS"] == "false"
    assert module.os.environ["PAPER_CHASER_EVAL_TRACE_PATH"] == "build/eval-curation/captured-events.jsonl"


def test_loading_batch_runner_does_not_import_server_eagerly() -> None:
    sys.modules.pop("paper_chaser_mcp.server", None)

    _load_module()

    assert "paper_chaser_mcp.server" not in sys.modules


def test_resolve_placeholders_raises_helpful_error_for_missing_field() -> None:
    module = _load_module()

    with pytest.raises(RuntimeError) as exc_info:
        module._resolve_placeholders(
            {"searchSessionId": "$result.smart_search.searchSessionId"},
            {"smart_search": {"resultStatus": "failed"}},
            scenario_name="landscape",
        )

    assert "landscape" in str(exc_info.value)
    assert "smart_search" in str(exc_info.value)
    assert "searchSessionId" in str(exc_info.value)


def test_raise_for_structured_tool_error_includes_fallbacks() -> None:
    module = _load_module()

    with pytest.raises(RuntimeError) as exc_info:
        module._raise_for_structured_tool_error(
            "smart_search",
            "search_papers_smart",
            {
                "error": "FEATURE_NOT_CONFIGURED",
                "message": "search_papers_smart requires the agentic runtime to be enabled.",
                "fallbackTools": ["research", "search_papers"],
            },
        )

    assert "FEATURE_NOT_CONFIGURED" in str(exc_info.value)
    assert "research, search_papers" in str(exc_info.value)
