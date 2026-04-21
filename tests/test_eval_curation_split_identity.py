"""Identity checks pinning the Phase 11 eval_curation subpackage layout.

The Phase 11 refactor splits the monolithic ``paper_chaser_mcp/eval_curation.py``
into a ``paper_chaser_mcp/eval_curation/`` subpackage. The package root
(``__init__.py``) re-exports the full public and underscore-prefixed API so
that importers in ``scripts/`` and ``tests/`` continue to work. These tests
guard the submodule ownership of individual symbols so accidentally moving
a helper between submodules fails here loudly.
"""

from paper_chaser_mcp import eval_curation as _pkg
from paper_chaser_mcp.eval_curation import capture as _capture
from paper_chaser_mcp.eval_curation import promotion as _promotion
from paper_chaser_mcp.eval_curation import review_queue as _review_queue
from paper_chaser_mcp.eval_curation import rubric as _rubric

_CAPTURE_NAMES = (
    "_compact_source",
    "_compact_recommendations",
    "_compact_theme",
    "_compact_graph_node",
    "_first_dict",
    "_compact_provider_outcomes",
    "_list_of_strings",
    "_infer_prompt_family",
    "_build_heuristic_summary",
    "_tags_with_prompt_family",
    "_extract_telemetry",
    "extract_ranking_diagnostics",
    "_maybe_attach_ranking_diagnostics",
    "build_eval_capture_payload",
    "maybe_capture_eval_candidate",
)

_REVIEW_QUEUE_NAMES = (
    "load_captured_eval_events",
    "build_review_queue_rows",
    "write_review_queue",
    "render_review_queue",
)

_PROMOTION_NAMES = (
    "build_batch_summary",
    "build_batch_ledger_rows",
    "write_batch_ledger_csv",
)

_RUBRIC_NAMES = (
    "ENV_SCI_JUDGE_ROLES",
    "EnvSciJudgeRubric",
    "DEFAULT_ENV_SCI_BENCHMARK_PACK",
    "DEFAULT_ENV_SCI_JUDGE_RUBRIC",
    "load_env_sci_benchmark_pack",
    "load_env_sci_judge_rubric_template",
    "build_env_sci_judge_prompt",
)


def test_capture_module_owns_capture_helpers() -> None:
    for name in _CAPTURE_NAMES:
        value = getattr(_capture, name)
        assert getattr(_pkg, name) is value, name


def test_review_queue_module_owns_review_queue_helpers() -> None:
    for name in _REVIEW_QUEUE_NAMES:
        value = getattr(_review_queue, name)
        assert getattr(_pkg, name) is value, name
        assert value.__module__ == "paper_chaser_mcp.eval_curation.review_queue", name


def test_promotion_module_owns_ledger_and_summary_helpers() -> None:
    for name in _PROMOTION_NAMES:
        value = getattr(_promotion, name)
        assert getattr(_pkg, name) is value, name
        assert value.__module__ == "paper_chaser_mcp.eval_curation.promotion", name


def test_rubric_module_owns_env_sci_rubric_block() -> None:
    for name in _RUBRIC_NAMES:
        value = getattr(_rubric, name)
        assert getattr(_pkg, name) is value, name
