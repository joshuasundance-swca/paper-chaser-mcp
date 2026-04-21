"""Identity checks pinning the Phase 11 eval_canary subpackage layout.

The Phase 11 refactor splits the monolithic ``paper_chaser_mcp/eval_canary.py``
into a ``paper_chaser_mcp/eval_canary/`` subpackage. The package root
(``__init__.py``) re-exports the full public and underscore-prefixed API so
that importers continue to work. These tests guard submodule ownership so
accidentally moving a helper fails here loudly.
"""

from paper_chaser_mcp import eval_canary as _pkg
from paper_chaser_mcp.eval_canary import schema as _schema
from paper_chaser_mcp.eval_canary import validators as _validators
from paper_chaser_mcp.eval_canary import workflow as _workflow

_SCHEMA_CONSTANTS = (
    "SCHEMA_VERSION",
    "TASK_FAMILIES",
    "ORIGINS",
    "REVIEW_STATUSES",
    "PLANNER_INTENTS",
    "ANSWER_STATUSES",
    "ABSTENTION_BEHAVIORS",
    "PROVENANCE_SOURCE_TYPES",
    "PROVENANCE_TRUST_STATES",
    "PROVENANCE_ACCESS_STATES",
    "RUNTIME_PROFILES",
)

_SCHEMA_HELPERS = (
    "_validate_structured_model",
    "_jsonl_files",
    "_iter_raw_eval_rows",
    "_result_item",
)

_VALIDATOR_NAMES = (
    "_validate_meta",
    "_validate_input",
    "_validate_expected",
    "validate_eval_item",
    "_planner_provider_candidates",
    "_validate_runtime_live_response",
    "_validate_planner_live_response",
    "_validate_synthesis_live_response",
)

_WORKFLOW_NAMES = (
    "run_eval_canary",
    "_default_live_tool_sequence",
    "run_live_eval_canary",
    "render_canary_report",
)


def test_schema_module_owns_constants() -> None:
    for name in _SCHEMA_CONSTANTS:
        value = getattr(_schema, name)
        assert getattr(_pkg, name) is value, name


def test_schema_module_owns_helpers() -> None:
    for name in _SCHEMA_HELPERS:
        value = getattr(_schema, name)
        assert getattr(_pkg, name) is value, name
        assert value.__module__ == "paper_chaser_mcp.eval_canary.schema", name


def test_validators_module_owns_validator_functions() -> None:
    for name in _VALIDATOR_NAMES:
        value = getattr(_validators, name)
        assert getattr(_pkg, name) is value, name
        assert value.__module__ == "paper_chaser_mcp.eval_canary.validators", name


def test_workflow_module_owns_orchestration_functions() -> None:
    for name in _WORKFLOW_NAMES:
        value = getattr(_workflow, name)
        assert getattr(_pkg, name) is value, name
        assert value.__module__ == "paper_chaser_mcp.eval_canary.workflow", name
