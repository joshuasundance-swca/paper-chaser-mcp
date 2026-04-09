import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_eval_topics.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_eval_topics_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_settings() -> SimpleNamespace:
    return SimpleNamespace(
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        enable_agentic=True,
        agentic_provider="deterministic",
        disable_embeddings=True,
        agentic_openai_timeout_seconds=30.0,
        agentic_index_backend="memory",
        session_ttl_seconds=3600,
        enable_agentic_trace_log=False,
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_http_referer=None,
        openrouter_title=None,
        azure_openai_api_key=None,
        azure_openai_endpoint=None,
        azure_openai_api_version=None,
        azure_openai_planner_deployment=None,
        azure_openai_synthesis_deployment=None,
        anthropic_api_key=None,
        nvidia_api_key=None,
        nvidia_nim_base_url=None,
        google_api_key=None,
        mistral_api_key=None,
        huggingface_api_key=None,
        huggingface_base_url=None,
    )


def test_generate_topics_forwards_openrouter_settings_to_bundle_resolution(tmp_path: Path) -> None:
    module = _load_module()

    class _Plan:
        def __init__(self) -> None:
            self.intent = "discovery"
            self.follow_up_mode = "qa"
            self.provider_plan = ["semantic-scholar"]
            self.candidate_concepts = ["wildfire smoke"]
            self.constraints: dict[str, str] = {}
            self.success_criteria = ["collect relevant evidence"]

    class _Bundle:
        async def asuggest_speculative_expansions(self, **kwargs):
            return []

        async def aplan_search(self, **kwargs):
            return _Plan()

        def selection_metadata(self):
            return {
                "configuredSmartProvider": "openrouter",
                "activeSmartProvider": "openrouter",
                "plannerModel": "arcee-ai/trinity-mini",
            }

        async def aclose(self):
            return None

    captured_kwargs: dict[str, object] = {}

    def _settings() -> SimpleNamespace:
        settings = _stub_settings()
        settings.agentic_provider = "openrouter"
        settings.openrouter_api_key = "sk-or-test"
        settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        settings.openrouter_http_referer = "https://example.test"
        settings.openrouter_title = "Paper Chaser Test"
        settings.planner_model = "arcee-ai/trinity-mini"
        settings.synthesis_model = "arcee-ai/trinity-large-thinking"
        return settings

    def _resolve(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _Bundle()

    original_settings = module._settings_from_env
    original_resolve = module.resolve_provider_bundle
    try:
        module._settings_from_env = lambda _: _settings()
        module.resolve_provider_bundle = _resolve
        payload = asyncio.run(
            module.generate_topics(
                argparse.Namespace(
                    dotenv_path=str(tmp_path / ".env"),
                    context_text=None,
                    taxonomy_preset="balanced-science",
                    taxonomy_file=None,
                    seed_query=["wildfire smoke exposure and respiratory outcomes"],
                    seed_file=None,
                    seed_preset="none",
                    max_variants=0,
                    include_original=True,
                    latency_profile="balanced",
                    merge_inputs=None,
                    min_quality_score=0.0,
                    max_topics=1,
                    ai_prune_mode="off",
                    ai_prune_below_score=35.0,
                    domain_balance_mode="off",
                    domain_balance_max_share=0.4,
                )
            )
        )
    finally:
        module._settings_from_env = original_settings
        module.resolve_provider_bundle = original_resolve

    assert payload["topics"]
    assert captured_kwargs["openrouter_api_key"] == "sk-or-test"
    assert captured_kwargs["openrouter_base_url"] == "https://openrouter.ai/api/v1"
    assert captured_kwargs["openrouter_http_referer"] == "https://example.test"
    assert captured_kwargs["openrouter_title"] == "Paper Chaser Test"


def test_load_seed_queries_combines_inline_and_file_inputs(tmp_path: Path) -> None:
    module = _load_module()
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("PFAS remediation in groundwater\nGraph neural network benchmarks\n", encoding="utf-8")
    args = argparse.Namespace(
        seed_query=["PFAS remediation in groundwater", "soil carbon verification"],
        seed_file=str(seed_file),
    )

    queries = module.load_seed_queries(args)

    assert queries == [
        "PFAS remediation in groundwater",
        "soil carbon verification",
        "Graph neural network benchmarks",
    ]


def test_load_seed_queries_falls_back_to_default_seed_preset() -> None:
    module = _load_module()
    args = argparse.Namespace(
        seed_query=None,
        seed_file=None,
        seed_preset=None,
    )

    queries = module.load_seed_queries(args)

    assert "PFAS remediation in groundwater treatment systems" in queries
    assert "public health evidence on wildfire smoke exposure and respiratory outcomes" in queries


def test_parser_allows_easy_button_without_explicit_output() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(["--easy-button"])

    assert args.easy_button is True
    assert args.output is None


def test_build_scenario_payload_emits_search_and_follow_up_pairs() -> None:
    module = _load_module()
    payload = module.build_scenario_payload(
        [
            {"query": "PFAS remediation in groundwater", "followUpMode": "qa"},
            {"query": "Attention Is All You Need", "followUpMode": "claim_check"},
        ],
        latency_profile="balanced",
        emit_follow_up=True,
    )

    scenarios = payload["scenarios"]
    assert scenarios[0]["tool"] == "search_papers_smart"
    assert scenarios[1]["tool"] == "ask_result_set"
    assert scenarios[1]["arguments"]["answerMode"] == "qa"
    assert scenarios[3]["arguments"]["answerMode"] == "claim_check"


def test_follow_up_question_varies_by_mode() -> None:
    module = _load_module()

    assert "strongly supported" in module._follow_up_question("claim_check")
    assert "comparisons or tradeoffs" in module._follow_up_question("comparison")
    assert "evidence gaps" in module._follow_up_question("qa")


def test_assign_topic_taxonomy_uses_matching_rule() -> None:
    module = _load_module()

    family, tags = module.assign_topic_taxonomy(
        query="PFAS remediation in groundwater",
        intent="discovery",
        candidate_concepts=["groundwater cleanup", "treatment trains"],
        taxonomy_rules=[
            {
                "family": "environmental_remediation",
                "tags": ["water", "pfas"],
                "keywords": ["pfas", "groundwater"],
                "intents": ["discovery", "review"],
            }
        ],
    )

    assert family == "environmental_remediation"
    assert "water" in tags
    assert "discovery" in tags


def test_merge_generated_topic_payloads_dedupes_query_and_prefers_current(tmp_path: Path) -> None:
    module = _load_module()
    current = {
        "seedQueries": ["Current seed"],
        "topics": [{"query": "PFAS remediation in groundwater", "intent": "discovery", "family": "new_family"}],
    }

    prior_path = tmp_path / "merge_prior_topics.json"
    prior_path.write_text(
        (
            '{"seedQueries": ["Prior seed"], '
            '"topics": ['
            '{"query": "PFAS remediation in groundwater", "intent": "review", "family": "old_family"},'
            '{"query": "soil carbon verification", "intent": "review", "family": "carbon_market"}'
            "]}"
        ),
        encoding="utf-8",
    )
    merged = module.merge_generated_topic_payloads(current, [str(prior_path)])

    assert merged["seedQueries"] == ["Prior seed", "Current seed"]
    assert len(merged["topics"]) == 2
    assert merged["topics"][0]["family"] == "new_family"
    assert merged["topics"][1]["query"] == "soil carbon verification"


def test_load_taxonomy_rules_accepts_rules_object(tmp_path: Path) -> None:
    module = _load_module()
    taxonomy_file = tmp_path / "taxonomy.json"
    taxonomy_file.write_text(
        '{"rules": [{"family": "benchmarking", "tags": ["benchmark"], "keywords": ["benchmark"]}]}',
        encoding="utf-8",
    )

    rules = module.load_taxonomy_rules(str(taxonomy_file))

    assert rules[0]["family"] == "benchmarking"


def test_score_topic_candidate_rewards_taxonomy_and_planning_depth() -> None:
    module = _load_module()

    score, tier, signals = module.score_topic_candidate(
        {
            "seedQuery": "PFAS remediation",
            "query": "PFAS remediation in groundwater treatment systems",
            "intent": "review",
            "family": "environmental_remediation",
            "tags": ["review", "water", "pfas", "environmental_remediation"],
            "providerPlan": ["semantic-scholar", "openalex"],
            "candidateConcepts": ["groundwater", "treatment trains"],
            "successCriteria": ["identify strongest papers", "surface evidence gaps"],
        }
    )

    assert score >= 50.0
    assert tier in {"medium", "high"}
    assert "taxonomy_family" in signals
    assert "provider_plan" in signals


def test_rank_topics_filters_and_orders_by_quality_score() -> None:
    module = _load_module()
    ranked = module.rank_topics(
        [
            {
                "seedQuery": "seed",
                "query": "short query",
                "intent": "discovery",
                "family": "literature_discovery",
                "tags": ["discovery"],
            },
            {
                "seedQuery": "PFAS remediation",
                "query": "PFAS remediation in groundwater treatment systems",
                "intent": "review",
                "family": "environmental_remediation",
                "tags": ["review", "water", "pfas", "environmental_remediation"],
                "providerPlan": ["semantic-scholar", "openalex"],
                "candidateConcepts": ["groundwater", "treatment trains"],
                "successCriteria": ["identify strongest papers", "surface evidence gaps"],
            },
        ],
        min_quality_score=20.0,
        max_topics=1,
    )

    assert len(ranked) == 1
    assert ranked[0]["query"] == "PFAS remediation in groundwater treatment systems"
    assert ranked[0]["qualityScore"] >= 20.0


def test_default_taxonomy_file_exists() -> None:
    module = _load_module()

    assert module.DEFAULT_TAXONOMY_FILE.exists()


def test_resolve_taxonomy_file_prefers_explicit_path(tmp_path: Path) -> None:
    module = _load_module()
    explicit = tmp_path / "custom-taxonomy.json"
    explicit.write_text('{"rules": []}', encoding="utf-8")
    args = argparse.Namespace(
        taxonomy_file=str(explicit),
        taxonomy_preset="balanced-science",
    )

    path = module.resolve_taxonomy_file(args)

    assert path == explicit


def test_resolve_taxonomy_file_uses_preset_when_no_override() -> None:
    module = _load_module()
    args = argparse.Namespace(
        taxonomy_file=None,
        taxonomy_preset="environmental-consulting",
    )

    path = module.resolve_taxonomy_file(args)

    assert path == module.TAXONOMY_PRESETS["environmental-consulting"]


def test_write_topics_jsonl_and_csv_emit_reviewable_outputs(tmp_path: Path) -> None:
    module = _load_module()
    topics = [
        {
            "query": "PFAS remediation in groundwater treatment systems",
            "seedQuery": "PFAS remediation",
            "intent": "review",
            "family": "environmental_remediation",
            "tags": ["review", "water"],
            "qualityScore": 62.0,
            "qualityTier": "high",
            "followUpMode": "comparison",
            "providerPlan": ["semantic-scholar", "openalex"],
            "candidateConcepts": ["groundwater", "treatment trains"],
            "successCriteria": ["identify strongest papers"],
            "rationale": "Generated topic candidate.",
        }
    ]
    jsonl_path = tmp_path / "topics.jsonl"
    csv_path = tmp_path / "topics.csv"

    module.write_topics_jsonl(jsonl_path, topics)
    module.write_topics_csv(csv_path, topics)

    jsonl_text = jsonl_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")
    assert '"qualityScore": 62.0' in jsonl_text
    assert "PFAS remediation in groundwater treatment systems" in csv_text
    assert "environmental_remediation" in csv_text


def test_write_topics_markdown_emits_table(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "topics.md"
    module.write_topics_markdown(
        path,
        [
            {
                "query": "PFAS remediation in groundwater treatment systems",
                "family": "environmental_remediation",
                "intent": "review",
                "qualityScore": 62.0,
                "qualityTier": "high",
                "tags": ["review", "water"],
            }
        ],
    )

    markdown = path.read_text(encoding="utf-8")
    assert "| Query | Family | Intent | Score | Tier | Tags |" in markdown
    assert "PFAS remediation in groundwater treatment systems" in markdown


def test_build_viewer_command_uses_same_python_and_output_path() -> None:
    module = _load_module()
    args = argparse.Namespace(
        output="build/eval-workflow/generated-topics.json",
        viewer_host="127.0.0.1",
        viewer_port=8767,
    )

    command = module.build_viewer_command(args)

    assert command[0] == module.sys.executable
    assert command[1].endswith("view_generated_topics.py")
    assert "build/eval-workflow/generated-topics.json" in command


def test_apply_easy_button_defaults_populates_integrated_outputs(tmp_path: Path) -> None:
    module = _load_module()
    args = argparse.Namespace(
        easy_button=True,
        artifact_root=str(tmp_path),
        output=None,
        jsonl_output=None,
        csv_output=None,
        markdown_output=None,
        scenario_output=None,
        seed_preset=None,
        include_original=False,
        emit_follow_up=False,
        min_quality_score=0.0,
        max_topics=None,
        launch_workflow=False,
        launch_viewer=False,
        workflow_matrix_preset=None,
    )

    module.apply_easy_button_defaults(args)

    assert args.output.endswith("generated-topics.json")
    assert args.jsonl_output.endswith("generated-topics.jsonl")
    assert args.csv_output.endswith("generated-topics.csv")
    assert args.markdown_output.endswith("generated-topics.md")
    assert args.scenario_output.endswith("generated-batch.json")
    assert args.seed_preset == module.DEFAULT_SEED_PRESET
    assert args.include_original is True
    assert args.emit_follow_up is True
    assert args.min_quality_score == 30.0
    assert args.max_topics == 25
    assert args.launch_viewer is True


def test_assess_topic_pool_risks_flags_narrow_low_quality_pool() -> None:
    module = _load_module()
    warnings, recommendation = module.assess_topic_pool_risks(
        {
            "topicCount": 6,
            "averageQualityScore": 28.0,
            "familyCount": 1,
            "maxFamilyShare": 1.0,
            "maxIntentShare": 1.0,
            "intents": {"discovery": 6},
            "qualityTiers": {"medium": 2, "low": 4},
        }
    )

    assert recommendation == "human-review-required"
    assert any("domain diversity is low" in warning for warning in warnings)
    assert any("non-discovery intents are underrepresented" in warning for warning in warnings)


def test_build_family_cross_check_detects_disagreement() -> None:
    module = _load_module()

    cross_check = module.build_family_cross_check(
        {
            "query": "wildfire smoke exposure and respiratory outcomes",
            "intent": "review",
            "family": "computer_science",
            "candidateConcepts": ["public health", "respiratory outcomes"],
            "tags": ["health"],
        },
        [
            {
                "family": "health_science",
                "keywords": ["wildfire smoke", "respiratory", "public health"],
                "tags": ["health"],
                "intents": ["review"],
            }
        ],
    )

    assert cross_check["disagreement"] is True
    assert cross_check["topCandidateFamily"] == "health_science"


def test_annotate_family_cross_checks_adds_summary_counts() -> None:
    module = _load_module()
    topics, summary = module.annotate_family_cross_checks(
        [
            {
                "query": "wildfire smoke exposure and respiratory outcomes",
                "intent": "review",
                "family": "computer_science",
                "candidateConcepts": ["public health"],
            }
        ],
        [
            {
                "family": "health_science",
                "keywords": ["wildfire smoke", "respiratory", "public health"],
                "intents": ["review"],
            }
        ],
    )

    assert summary["disagreementCount"] == 1
    assert topics[0]["familyCrossCheck"]["disagreement"] is True


def test_build_workflow_command_uses_generator_outputs() -> None:
    module = _load_module()
    args = argparse.Namespace(
        scenario_output="build/eval-workflow/generated-batch.json",
        artifact_root="build/eval-workflow",
        dotenv_path=".env",
        workflow_review_mode="yolo",
        workflow_matrix_preset=["cross-provider-lower-bound"],
        workflow_launch_matrix_viewer=True,
    )

    command = module.build_workflow_command(args)

    assert command[0] == module.sys.executable
    assert command[1].endswith("run_eval_workflow.py")
    assert "build/eval-workflow/generated-batch.json" in command
    assert "cross-provider-lower-bound" in command
    assert "--launch-matrix-viewer" in command


def test_apply_easy_button_defaults_sets_workflow_preset_when_launching(tmp_path: Path) -> None:
    module = _load_module()
    args = argparse.Namespace(
        easy_button=True,
        artifact_root=str(tmp_path),
        output=None,
        jsonl_output=None,
        csv_output=None,
        markdown_output=None,
        scenario_output=None,
        seed_preset=None,
        include_original=False,
        emit_follow_up=False,
        min_quality_score=0.0,
        max_topics=None,
        ai_prune_mode="off",
        domain_balance_mode="off",
        launch_workflow=True,
        launch_viewer=False,
        workflow_matrix_preset=None,
    )

    module.apply_easy_button_defaults(args)

    assert args.workflow_matrix_preset == ["cross-provider-lower-bound"]
    assert args.ai_prune_mode == "rewrite-or-drop"
    assert args.domain_balance_mode == "round-robin"


def test_rebalance_topics_round_robin_limits_family_dominance() -> None:
    module = _load_module()
    topics = [
        {"query": "a1", "family": "env", "qualityScore": 90},
        {"query": "a2", "family": "env", "qualityScore": 80},
        {"query": "b1", "family": "health", "qualityScore": 70},
        {"query": "c1", "family": "cs", "qualityScore": 60},
        {"query": "a3", "family": "env", "qualityScore": 50},
    ]

    balanced, summary = module.rebalance_topics(
        topics,
        mode="round-robin",
        max_topics=4,
        max_family_share=0.5,
    )

    assert len(balanced) == 4
    assert {topic["family"] for topic in balanced[:3]} == {"env", "health", "cs"}
    assert summary["applied"] is True


def test_ai_assisted_prune_topics_rewrites_weak_topic() -> None:
    module = _load_module()

    class _Candidate:
        def __init__(self, variant: str, rationale: str) -> None:
            self.variant = variant
            self.rationale = rationale

    class _Plan:
        def __init__(self) -> None:
            self.intent = "discovery"
            self.follow_up_mode = "qa"
            self.provider_plan = ["semantic-scholar", "openalex"]
            self.candidate_concepts = ["groundwater", "treatment"]
            self.constraints: dict[str, str] = {}
            self.success_criteria = ["identify strongest papers"]

    class _Bundle:
        async def asuggest_speculative_expansions(self, **kwargs):
            return [_Candidate("PFAS remediation in groundwater treatment systems", "adds treatment specificity")]

        async def aplan_search(self, **kwargs):
            return _Plan()

    pruned, summary = asyncio.run(
        module.ai_assisted_prune_topics(
            [
                {
                    "seedQuery": "PFAS remediation",
                    "query": "PFAS remediation",
                    "intent": "discovery",
                    "family": "literature_discovery",
                    "tags": ["discovery"],
                }
            ],
            mode="rewrite-or-drop",
            prune_below_score=35.0,
            provider_bundle=_Bundle(),
            taxonomy_rules=[],
            context_texts=[],
        )
    )

    assert summary["rewritten"] == 1
    assert summary["audit"][0]["action"] == "rewritten"
    assert pruned[0]["query"] == "PFAS remediation in groundwater treatment systems"
    assert pruned[0]["rewriteSource"] == "PFAS remediation"


def test_should_launch_workflow_respects_review_recommendation_and_force_flag() -> None:
    module = _load_module()
    args = argparse.Namespace(
        launch_workflow=True,
        force_launch_workflow=False,
    )

    assert module.should_launch_workflow(args, {"reviewRecommendation": "ai-assisted-ok"}) is True
    assert module.should_launch_workflow(args, {"reviewRecommendation": "human-review-required"}) is False

    args.force_launch_workflow = True
    assert module.should_launch_workflow(args, {"reviewRecommendation": "human-review-required"}) is True


def test_should_launch_workflow_safe_policy_requires_clean_pool() -> None:
    module = _load_module()
    args = argparse.Namespace(
        launch_workflow=True,
        force_launch_workflow=False,
        workflow_autopilot_policy="safe",
        safe_min_topic_count=3,
        safe_min_family_count=2,
        safe_min_intent_count=1,
        hard_blocker_min_topic_count=3,
        hard_blocker_max_family_share=1.0,
    )

    assert (
        module.should_launch_workflow(
            args,
            {
                "reviewRecommendation": "ai-assisted-ok",
                "riskWarnings": [],
                "summary": {
                    "topicCount": 3,
                    "averageQualityScore": 52.0,
                    "familyCount": 2,
                    "intentCount": 1,
                    "qualityTiers": {"high": 3},
                    "maxFamilyShare": 0.35,
                    "familyDisagreementCount": 0,
                    "rewrittenShare": 0.0,
                },
            },
        )
        is True
    )
    assert (
        module.should_launch_workflow(
            args,
            {
                "reviewRecommendation": "ai-assisted-ok",
                "riskWarnings": ["family cross-check disagrees"],
                "summary": {
                    "topicCount": 3,
                    "averageQualityScore": 52.0,
                    "familyCount": 2,
                    "intentCount": 1,
                    "qualityTiers": {"high": 3},
                    "maxFamilyShare": 0.35,
                    "familyDisagreementCount": 1,
                    "rewrittenShare": 0.0,
                },
            },
        )
        is False
    )


def test_workflow_suppression_reason_explains_safe_policy_failure() -> None:
    module = _load_module()
    args = argparse.Namespace(
        launch_workflow=True,
        force_launch_workflow=False,
        workflow_autopilot_policy="safe",
        safe_min_topic_count=3,
        safe_min_family_count=2,
        safe_min_intent_count=1,
        hard_blocker_min_topic_count=3,
        hard_blocker_max_family_share=1.0,
    )

    reason = module.workflow_suppression_reason(
        args,
        {
            "reviewRecommendation": "ai-assisted-ok",
            "riskWarnings": [],
            "summary": {
                "topicCount": 3,
                "averageQualityScore": 44.0,
                "familyCount": 2,
                "intentCount": 1,
                "qualityTiers": {"high": 3},
                "maxFamilyShare": 0.35,
                "familyDisagreementCount": 0,
                "rewrittenShare": 0.0,
            },
        },
    )

    assert "average quality score" in reason


def test_build_workflow_launch_decision_includes_trace_and_threshold_override() -> None:
    module = _load_module()
    args = argparse.Namespace(
        launch_workflow=True,
        force_launch_workflow=False,
        workflow_autopilot_policy="safe",
        safe_min_average_quality=45.0,
        safe_min_topic_count=3,
        safe_min_family_count=1,
        safe_min_intent_count=1,
        safe_max_family_share=1.0,
        safe_max_risk_warnings=0,
        safe_max_family_disagreement_count=0,
        safe_max_rewritten_share=0.5,
        hard_blocker_min_topic_count=3,
        hard_blocker_max_family_share=1.0,
        hard_blocker_max_family_disagreement_count=1,
        hard_blocker_max_rewritten_share=0.5,
    )

    decision = module.build_workflow_launch_decision(
        args,
        {
            "reviewRecommendation": "ai-assisted-ok",
            "riskWarnings": [],
            "hardBlockers": [],
            "summary": {
                "topicCount": 3,
                "averageQualityScore": 82.0,
                "familyCount": 1,
                "intentCount": 1,
                "qualityTiers": {"high": 3},
                "maxFamilyShare": 1.0,
                "familyDisagreementCount": 0,
                "rewrittenShare": 0.0,
            },
        },
    )

    assert decision["shouldLaunch"] is True
    assert decision["policy"] == "safe"
    assert any(item["check"] == "topic_count" for item in decision["trace"])


def test_identify_hard_blockers_respects_threshold_overrides() -> None:
    module = _load_module()
    blockers = module.identify_hard_blockers(
        {
            "topicCount": 3,
            "qualityTiers": {"high": 3},
            "maxFamilyShare": 1.0,
            "familyDisagreementCount": 0,
            "rewrittenShare": 0.0,
        },
        args=argparse.Namespace(
            hard_blocker_min_topic_count=3,
            hard_blocker_max_family_share=1.0,
            hard_blocker_max_family_disagreement_count=1,
            hard_blocker_max_rewritten_share=0.5,
        ),
    )

    assert blockers == []


def test_generate_topics_matches_snapshot_fixture(tmp_path: Path) -> None:
    module = _load_module()
    snapshot_path = REPO_ROOT / "tests" / "fixtures" / "evals" / "generated-topics.snapshot.json"

    class _Candidate:
        def __init__(self, variant: str, rationale: str) -> None:
            self.variant = variant
            self.rationale = rationale

    class _Plan:
        def __init__(self, intent: str, follow_up_mode: str, concepts: list[str]) -> None:
            self.intent = intent
            self.follow_up_mode = follow_up_mode
            self.provider_plan = ["semantic-scholar", "openalex"]
            self.candidate_concepts = concepts
            self.constraints: dict[str, str] = {}
            self.success_criteria = ["identify strongest papers"]

    class _Bundle:
        async def asuggest_speculative_expansions(self, **kwargs):
            return [
                _Candidate(
                    "public health evidence on wildfire smoke exposure and respiratory outcomes",
                    "adds public health framing",
                )
            ]

        async def aplan_search(self, **kwargs):
            query = str(kwargs["query"])
            concepts = (
                ["wildfire smoke", "public health"]
                if "public health" in query
                else ["wildfire smoke", "respiratory outcomes"]
            )
            return _Plan("review", "comparison", concepts)

        def selection_metadata(self):
            return {
                "configuredSmartProvider": "deterministic",
                "activeSmartProvider": "deterministic",
                "plannerModel": "stub-planner",
            }

        async def aclose(self):
            return None

    original_settings = module._settings_from_env
    original_resolve = module.resolve_provider_bundle
    try:
        module._settings_from_env = lambda _: _stub_settings()
        module.resolve_provider_bundle = lambda *args, **kwargs: _Bundle()
        payload = asyncio.run(
            module.generate_topics(
                argparse.Namespace(
                    dotenv_path=str(tmp_path / ".env"),
                    context_text=None,
                    taxonomy_preset="balanced-science",
                    taxonomy_file=None,
                    seed_query=["wildfire smoke exposure and respiratory outcomes"],
                    seed_file=None,
                    seed_preset="none",
                    max_variants=1,
                    include_original=True,
                    latency_profile="balanced",
                    merge_inputs=None,
                    min_quality_score=0.0,
                    max_topics=None,
                    ai_prune_mode="off",
                    ai_prune_below_score=35.0,
                    domain_balance_mode="off",
                    domain_balance_max_share=0.4,
                )
            )
        )
    finally:
        module._settings_from_env = original_settings
        module.resolve_provider_bundle = original_resolve

    assert payload == json.loads(snapshot_path.read_text(encoding="utf-8"))


def test_generate_topics_handles_provider_failures_with_fallback_snapshot(tmp_path: Path) -> None:
    module = _load_module()
    snapshot_path = REPO_ROOT / "tests" / "fixtures" / "evals" / "generated-topics.failure.snapshot.json"

    class _FailingBundle:
        async def asuggest_speculative_expansions(self, **kwargs):
            raise RuntimeError("expansion failure")

        async def aplan_search(self, **kwargs):
            raise RuntimeError("plan failure")

        def selection_metadata(self):
            return {
                "configuredSmartProvider": "deterministic",
                "activeSmartProvider": "deterministic",
                "plannerModel": "stub-planner",
            }

        async def aclose(self):
            return None

    original_settings = module._settings_from_env
    original_resolve = module.resolve_provider_bundle
    try:
        module._settings_from_env = lambda _: _stub_settings()
        module.resolve_provider_bundle = lambda *args, **kwargs: _FailingBundle()
        payload = asyncio.run(
            module.generate_topics(
                argparse.Namespace(
                    dotenv_path=str(tmp_path / ".env"),
                    context_text=None,
                    taxonomy_preset="balanced-science",
                    taxonomy_file=None,
                    seed_query=["PFAS remediation in groundwater"],
                    seed_file=None,
                    seed_preset="none",
                    max_variants=1,
                    include_original=True,
                    latency_profile="balanced",
                    merge_inputs=None,
                    min_quality_score=0.0,
                    max_topics=None,
                    ai_prune_mode="off",
                    ai_prune_below_score=35.0,
                    domain_balance_mode="off",
                    domain_balance_max_share=0.4,
                )
            )
        )
    finally:
        module._settings_from_env = original_settings
        module.resolve_provider_bundle = original_resolve

    assert payload == json.loads(snapshot_path.read_text(encoding="utf-8"))


def test_generate_topics_adds_single_seed_diversification_candidates(tmp_path: Path) -> None:
    module = _load_module()

    class _Candidate:
        def __init__(self, variant: str, rationale: str) -> None:
            self.variant = variant
            self.rationale = rationale

    class _Plan:
        def __init__(self, intent: str, concepts: list[str]) -> None:
            self.intent = intent
            self.follow_up_mode = "qa"
            self.provider_plan = ["semantic-scholar"]
            self.candidate_concepts = concepts
            self.constraints: dict[str, str] = {}
            self.success_criteria = ["collect relevant evidence"]

    class _Bundle:
        async def asuggest_speculative_expansions(self, **kwargs):
            evidence_text = " ".join(str(item) for item in kwargs.get("evidence_texts") or [])
            if "Diversification angle: review" in evidence_text:
                return [_Candidate("systematic review of wildfire smoke respiratory outcomes", "review angle")]
            if "Diversification angle: regulatory" in evidence_text:
                return [_Candidate("wildfire smoke public health guidance", "regulatory angle")]
            if "Diversification angle: methods" in evidence_text:
                return [_Candidate("wildfire smoke exposure monitoring methods", "methods angle")]
            return []

        async def aplan_search(self, **kwargs):
            query = str(kwargs["query"])
            if "systematic review" in query:
                return _Plan("review", ["systematic review", "respiratory outcomes"])
            if "guidance" in query:
                return _Plan("regulatory", ["public health guidance", "wildfire smoke"])
            if "monitoring" in query:
                return _Plan("discovery", ["monitoring", "exposure methods"])
            return _Plan("discovery", ["wildfire smoke", "respiratory outcomes"])

        def selection_metadata(self):
            return {
                "configuredSmartProvider": "deterministic",
                "activeSmartProvider": "deterministic",
                "plannerModel": "stub-planner",
            }

        async def aclose(self):
            return None

    original_settings = module._settings_from_env
    original_resolve = module.resolve_provider_bundle
    try:
        module._settings_from_env = lambda _: _stub_settings()
        module.resolve_provider_bundle = lambda *args, **kwargs: _Bundle()
        payload = asyncio.run(
            module.generate_topics(
                argparse.Namespace(
                    dotenv_path=str(tmp_path / ".env"),
                    context_text=None,
                    taxonomy_preset="balanced-science",
                    taxonomy_file=None,
                    seed_query=["wildfire smoke exposure and respiratory outcomes"],
                    seed_file=None,
                    seed_preset="none",
                    max_variants=1,
                    include_original=True,
                    single_seed_diversification=True,
                    latency_profile="balanced",
                    merge_inputs=None,
                    min_quality_score=0.0,
                    max_topics=None,
                    ai_prune_mode="off",
                    ai_prune_below_score=35.0,
                    domain_balance_mode="off",
                    domain_balance_max_share=0.4,
                )
            )
        )
    finally:
        module._settings_from_env = original_settings
        module.resolve_provider_bundle = original_resolve

    intents = {topic["intent"] for topic in payload["topics"]}
    assert "review" in intents
    assert "regulatory" in intents
    assert any(topic.get("diversificationAngle") == "review" for topic in payload["topics"])
