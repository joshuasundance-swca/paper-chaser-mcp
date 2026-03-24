import builtins
import sys
import types
from typing import Any

import pytest
from pydantic import BaseModel, Field

from scholar_search_mcp.agentic import providers as providers_module
from scholar_search_mcp.agentic.config import AgenticConfig
from scholar_search_mcp.agentic.models import ExpansionCandidate, PlannerDecision
from scholar_search_mcp.agentic.providers import (
    COMMON_QUERY_WORDS,
    DeterministicProviderBundle,
    OpenAIProviderBundle,
    _cosine_similarity,
    _extract_seed_identifiers,
    _lexical_similarity,
    _normalize_confidence_label,
    _normalized_embedding_text,
    _PlannerResponseSchema,
    _tokenize,
    _top_terms,
    resolve_provider_bundle,
)
from scholar_search_mcp.provider_runtime import (
    ProviderCallResult,
    ProviderDiagnosticsRegistry,
    ProviderOutcomeEnvelope,
)


def _config(
    *,
    provider: str = "openai",
    disable_embeddings: bool = False,
    timeout: float = 30.0,
) -> AgenticConfig:
    return AgenticConfig(
        enabled=True,
        provider=provider,
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        disable_embeddings=disable_embeddings,
        openai_timeout_seconds=timeout,
    )


class _ExpansionItem(BaseModel):
    variant: str
    source: str = "speculative"
    rationale: str = ""


class _ExpansionPayload(BaseModel):
    expansions: list[_ExpansionItem]


class _AnswerPayload(BaseModel):
    answer: str
    unsupportedAsks: list[str] = Field(default_factory=list)
    followUpQuestions: list[str] = Field(default_factory=list)
    confidence: str = "medium"


def _success_result(payload: Any, *, endpoint: str) -> ProviderCallResult:
    return ProviderCallResult(
        payload=payload,
        outcome=ProviderOutcomeEnvelope(
            provider="openai",
            endpoint=endpoint,
            status_bucket="success",
        ),
    )


def test_provider_helpers_and_deterministic_bundle_paths() -> None:
    assert _tokenize("Recent graph-search papers") == [
        "recent",
        "graph",
        "search",
        "papers",
    ]
    assert _normalized_embedding_text("  graph\n search   papers  ") == ("graph search papers")
    top_terms = _top_terms(
        [
            "recent graph retrieval papers",
            "graph planning papers for retrieval agents",
        ],
        limit=4,
    )
    assert "graph" in top_terms
    assert not (set(top_terms) & COMMON_QUERY_WORDS)
    assert _lexical_similarity("graph retrieval", "retrieval graph") > 0.99
    assert _cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0
    assert _cosine_similarity((1.0, 1.0), (1.0, 1.0)) == pytest.approx(1.0)
    assert _normalize_confidence_label("very high") == "high"
    assert _normalize_confidence_label("mixed") == "medium"
    assert _normalize_confidence_label(0.2) == "low"
    assert _extract_seed_identifiers("doi 10.1234/example arxiv:2401.12345 https://example.com/paper 2401.12345") == [
        "10.1234/example",
        "arxiv:2401.12345",
        "https://example.com/paper",
        "2401.12345",
    ]

    bundle = DeterministicProviderBundle(_config(provider="deterministic"))

    known_item = bundle.plan_search(
        query="doi 10.1234/example",
        mode="auto",
        year="2024",
        venue="Nature",
        focus="benchmarks",
    )
    author = bundle.plan_search(query="author Yoshua Bengio", mode="auto")
    citation = bundle.plan_search(query="cited by graph papers", mode="auto")
    review = bundle.plan_search(query="landscape review of agents", mode="auto")

    assert known_item.intent == "known_item"
    assert known_item.constraints == {
        "year": "2024",
        "venue": "Nature",
        "focus": "benchmarks",
    }
    assert known_item.seed_identifiers == ["10.1234/example"]
    assert author.intent == "author"
    assert citation.intent == "citation"
    assert review.intent == "review"
    assert review.follow_up_mode == "claim_check"

    expansions = bundle.suggest_speculative_expansions(
        query="retrieval agents",
        evidence_texts=[
            "retrieval agents use citation graphs",
            "retrieval agents explore biomedical graph evidence",
            "recent work surveys graph retrieval",
        ],
        max_variants=2,
    )
    assert expansions
    assert len(expansions) <= 2
    assert all(isinstance(item, ExpansionCandidate) for item in expansions)
    assert all("retrieval agents" in item.variant for item in expansions)

    assert bundle.label_theme(seed_terms=["graph retrieval", "grounding"], papers=[]) == "Graph Retrieval / Grounding"
    assert bundle.label_theme(seed_terms=[], papers=[{"venue": "Nature"}]) == "Nature cluster"
    assert bundle.label_theme(seed_terms=[], papers=[]) == "General theme"
    assert (
        bundle.label_theme(
            seed_terms=[],
            papers=[
                {"title": "Noise impacts on wildlife"},
                {"title": "Noise effects on wildlife physiology"},
                {"title": "Noise and wildlife population trends"},
            ],
        )
        == "Noise / Wildlife"
    )

    empty_summary = bundle.summarize_theme(title="Agents", papers=[])
    populated_summary = bundle.summarize_theme(
        title="Agents",
        papers=[
            {"title": "A", "venue": "Nature", "year": 2022},
            {"title": "B", "venue": "Science", "year": 2024},
        ],
    )
    assert "no papers were available" in empty_summary
    assert "Nature, Science" in populated_summary
    assert "2022-2024" in populated_summary
    assert "Representative papers include A, B." in populated_summary

    no_evidence = bundle.answer_question(
        question="What is the strongest result?",
        evidence_papers=[],
        answer_mode="qa",
    )
    claim_check = bundle.answer_question(
        question="Is the claim supported?",
        evidence_papers=[
            {"title": "Paper A", "venue": "Nature", "year": 2024},
            {"title": "Paper B", "venue": "Science", "year": 2025},
        ],
        answer_mode="claim_check",
    )
    comparison = bundle.answer_question(
        question="Compare them",
        evidence_papers=[
            {"title": "Paper A", "venue": "Nature", "year": 2024},
            {"paperId": "paper-b"},
        ],
        answer_mode="comparison",
    )
    qa = bundle.answer_question(
        question="Answer directly",
        evidence_papers=[{"title": "Paper A"}, {"title": "Paper B"}],
        answer_mode="qa",
    )
    gap_qa = bundle.answer_question(
        question="What are the main knowledge gaps highlighted across these papers?",
        evidence_papers=[
            {
                "title": "Anthropogenic noise effects on wildlife behavior in birds",
                "abstract": "Acoustic behavior responses in North America.",
            },
            {
                "title": "Noise and wildlife behavior across birds",
                "abstract": "Behavioral communication changes in North America.",
            },
            {
                "title": "Noise response of mammals in Canada",
                "abstract": "Acoustic behavior study near protected areas.",
            },
        ],
        answer_mode="qa",
    )

    assert no_evidence["confidence"] == "low"
    assert claim_check["confidence"] == "medium"
    assert "supported" in claim_check["answer"]
    assert comparison["answer"].startswith("Comparison grounded")
    assert "Takeaway:" in comparison["answer"]
    assert "Paper A" in qa["answer"]
    assert gap_qa["confidence"] == "medium"
    assert "main recurring knowledge gaps" in gap_qa["answer"]
    assert ("behavioral responses are better covered than physiological or demographic consequences") in gap_qa[
        "answer"
    ]
    assert "long-term or chronic exposure evidence is still thin" in gap_qa["answer"]
    assert ("community- and ecosystem-level impacts remain underrepresented") in gap_qa["answer"]
    assert ("interactions with other stressors are rarely studied directly") in gap_qa["answer"]

    assert isinstance(
        resolve_provider_bundle(
            _config(provider="deterministic"),
            openai_api_key=None,
        ),
        DeterministicProviderBundle,
    )
    assert isinstance(
        resolve_provider_bundle(
            _config(provider="openai"),
            openai_api_key="sk-test",
        ),
        OpenAIProviderBundle,
    )


def test_openai_provider_loaders_and_response_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, *, api_key: str, timeout: float, max_retries: int) -> None:
            created["openai"] = (api_key, timeout, max_retries)

    class _FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, timeout: float, max_retries: int) -> None:
            created["async_openai"] = (api_key, timeout, max_retries)

    class _FakeChatModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def _init_chat_model(**kwargs: Any) -> _FakeChatModel:
        return _FakeChatModel(**kwargs)

    class _FakeEmbeddings:
        def __init__(self, *, model: str, api_key: Any, max_retries: int) -> None:
            created["embeddings"] = (model, api_key.get_secret_value(), max_retries)

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    openai_module.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[attr-defined]
    langchain_package = types.ModuleType("langchain")
    chat_models_module = types.ModuleType("langchain.chat_models")
    setattr(chat_models_module, "init_chat_model", _init_chat_model)
    langchain_package.chat_models = chat_models_module  # type: ignore[attr-defined]
    langchain_openai_module = types.ModuleType("langchain_openai")
    langchain_openai_module.OpenAIEmbeddings = _FakeEmbeddings  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "langchain", langchain_package)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", chat_models_module)
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai_module)

    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")

    sync_client = bundle._load_openai_client()
    async_client = bundle._load_async_openai_client()
    planner, synthesizer = bundle._load_models()
    embeddings = bundle._load_embeddings()

    assert created["openai"] == ("sk-test", 30.0, 0)
    assert created["async_openai"] == ("sk-test", 30.0, 0)
    assert created["embeddings"] == ("text-embedding-3-large", "sk-test", 0)
    assert bundle._load_openai_client() is sync_client
    assert bundle._load_async_openai_client() is async_client
    assert bundle._load_models() == (planner, synthesizer)
    assert bundle._load_embeddings() is embeddings

    cached = bundle._cache_embedding(" alpha\nbeta ", [0.1, 0.2])
    assert cached == (0.1, 0.2)
    assert bundle._embedding_cache["alpha beta"] == cached
    assert bundle._responses_input("system", {"query": "agents"}) == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": '{"query": "agents"}'},
    ]

    assert (
        OpenAIProviderBundle._extract_response_text(types.SimpleNamespace(output_text="  direct answer  "))
        == "direct answer"
    )
    assert (
        OpenAIProviderBundle._extract_response_text(
            types.SimpleNamespace(
                output=[types.SimpleNamespace(content=[types.SimpleNamespace(text="  nested answer  ")])]
            )
        )
        == "nested answer"
    )

    planner_schema = OpenAIProviderBundle._extract_response_parsed(
        types.SimpleNamespace(
            output_parsed={
                "intent": "review",
                "constraints": {},
                "seedIdentifiers": [],
                "candidateConcepts": [],
                "providerPlan": [],
                "followUpMode": "qa",
            }
        ),
        _PlannerResponseSchema,
    )
    assert planner_schema.intent == "review"
    assert (
        OpenAIProviderBundle._extract_response_parsed(
            types.SimpleNamespace(
                output=[
                    types.SimpleNamespace(
                        content=[
                            types.SimpleNamespace(
                                parsed={
                                    "intent": "author",
                                    "constraints": {},
                                    "seedIdentifiers": [],
                                    "candidateConcepts": [],
                                    "providerPlan": [],
                                    "followUpMode": "qa",
                                }
                            )
                        ]
                    )
                ]
            ),
            _PlannerResponseSchema,
        ).intent
        == "author"
    )
    assert (
        OpenAIProviderBundle._extract_response_parsed(
            types.SimpleNamespace(
                output=[
                    types.SimpleNamespace(
                        content=[
                            types.SimpleNamespace(
                                text=(
                                    '{"intent":"citation","constraints":{},'
                                    '"seedIdentifiers":[],"candidateConcepts":[],'
                                    '"providerPlan":[],"followUpMode":"qa"}'
                                )
                            )
                        ]
                    )
                ]
            ),
            _PlannerResponseSchema,
        ).intent
        == "citation"
    )
    with pytest.raises(ValueError):
        OpenAIProviderBundle._extract_response_parsed(
            types.SimpleNamespace(output=[]),
            _PlannerResponseSchema,
        )

    assert bundle._embedding_vectors({"data": [{"embedding": [1, 2]}]}) == [[1.0, 2.0]]
    assert bundle._embedding_vectors(types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[3, 4])])) == [
        [3.0, 4.0]
    ]


def test_openai_provider_sync_response_and_embedding_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse_calls: list[dict[str, Any]] = []
    text_calls: list[dict[str, Any]] = []
    embedding_calls: list[dict[str, Any]] = []

    class _Responses:
        def parse(self, **kwargs: Any) -> Any:
            parse_calls.append(kwargs)
            return types.SimpleNamespace(
                output_parsed={
                    "intent": "review",
                    "constraints": {},
                    "seedIdentifiers": [],
                    "candidateConcepts": ["agents"],
                    "providerPlan": ["semantic_scholar"],
                    "followUpMode": "claim_check",
                }
            )

        def create(self, **kwargs: Any) -> Any:
            text_calls.append(kwargs)
            return types.SimpleNamespace(output_text=' "Theme label" ')

    class _EmbeddingsClient:
        def create(self, **kwargs: Any) -> Any:
            embedding_calls.append(kwargs)
            input_value = kwargs["input"]
            if isinstance(input_value, list):
                return {"data": [{"embedding": [1.0, 0.0]} for _ in input_value]}
            return {"data": [{"embedding": [0.8, 0.2]}]}

    def _execute_sync(**kwargs: Any) -> ProviderCallResult:
        return _success_result(kwargs["operation"](), endpoint=kwargs["endpoint"])

    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    bundle._openai_client = types.SimpleNamespace(
        responses=_Responses(),
        embeddings=_EmbeddingsClient(),
    )
    monkeypatch.setattr(providers_module, "execute_provider_call_sync", _execute_sync)

    parsed = bundle._responses_parse(
        endpoint="responses.parse:planner",
        model_name=bundle.planner_model_name,
        response_model=_PlannerResponseSchema,
        system_prompt="plan",
        payload={"query": "agents"},
        previous_response_id="resp_123",
    )
    text = bundle._responses_text(
        endpoint="responses.create:label_theme",
        model_name=bundle.synthesis_model_name,
        system_prompt="label",
        payload={"titles": ["A"]},
        max_output_tokens=40,
        previous_response_id="resp_456",
    )
    query_embedding = bundle.embed_query("retrieval agents")
    text_embeddings = bundle.embed_texts(["retrieval agents", "citation graph"])
    similarity = bundle.similarity("retrieval agents", "retrieval grounding")
    batched = bundle.batched_similarity(
        "retrieval agents",
        ["retrieval grounding", "bird migration"],
    )

    assert parsed is not None
    assert parsed.intent == "review"
    assert text == '"Theme label"'
    assert query_embedding == (0.8, 0.2)
    assert text_embeddings == [(0.8, 0.2), (1.0, 0.0)]
    assert similarity > 0.0
    assert len(batched) == 2
    assert batched[0] > batched[1]
    assert parse_calls[0]["previous_response_id"] == "resp_123"
    assert text_calls[0]["previous_response_id"] == "resp_456"
    assert embedding_calls


def test_openai_provider_sync_high_level_methods_cover_direct_and_model_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")

    def _direct_parse(**kwargs: Any) -> Any:
        endpoint = kwargs["endpoint"]
        if endpoint == "responses.parse:planner":
            return _PlannerResponseSchema(
                intent="review",
                constraints=providers_module._PlannerConstraintsSchema(year="2024"),
                seedIdentifiers=["10.1234/direct"],
                candidateConcepts=["agents"],
                providerPlan=["semantic_scholar"],
                followUpMode="claim_check",
            )
        if endpoint == "responses.parse:expansions":
            return _ExpansionPayload(
                expansions=[
                    _ExpansionItem(variant="retrieval agents citation"),
                    _ExpansionItem(variant="retrieval agents recent"),
                    _ExpansionItem(variant=" "),
                ]
            )
        if endpoint == "responses.parse:answer":
            return _AnswerPayload(
                answer="Grounded answer.",
                confidence="very high",
            )
        return None

    def _direct_text(**kwargs: Any) -> str | None:
        if kwargs["endpoint"] == "responses.create:label_theme":
            return '"Direct label"'
        if kwargs["endpoint"] == "responses.create:summarize_theme":
            return "Direct summary."
        return None

    monkeypatch.setattr(bundle, "_responses_parse", _direct_parse)
    monkeypatch.setattr(bundle, "_responses_text", _direct_text)

    plan = bundle.plan_search(query="agents", mode="auto")
    expansions = bundle.suggest_speculative_expansions(
        query="retrieval agents",
        evidence_texts=["citation graphs for retrieval agents"],
        max_variants=3,
    )
    label = bundle.label_theme(seed_terms=["agents"], papers=[])
    summary = bundle.summarize_theme(title="Agents", papers=[])
    answer = bundle.answer_question(
        question="What is best?",
        evidence_papers=[{"title": "Paper A"}],
        answer_mode="qa",
    )

    assert plan.intent == "review"
    assert plan.seed_identifiers == ["10.1234/direct"]
    assert [item.variant for item in expansions] == ["retrieval agents citation"]
    assert label == "Direct label"
    assert summary == "Direct summary."
    assert answer["confidence"] == "high"

    class _StructuredInvoker:
        def __init__(self, response: Any) -> None:
            self._response = response

        def invoke(self, messages: list[tuple[str, str]]) -> Any:
            assert messages
            return self._response

    class _PlannerModel:
        def __init__(self, response: Any) -> None:
            self._response = response

        def with_structured_output(self, schema: Any, method: str) -> _StructuredInvoker:
            assert schema is not None
            assert method == "function_calling"
            return _StructuredInvoker(self._response)

    class _SynthesizerModel:
        def __init__(self, response: Any) -> None:
            self._response = response

        def with_structured_output(self, schema: Any, method: str) -> _StructuredInvoker:
            assert schema is not None
            assert method == "function_calling"
            return _StructuredInvoker(self._response)

        def invoke(self, messages: list[tuple[str, str]]) -> Any:
            assert messages
            return self._response

    monkeypatch.setattr(bundle, "_responses_parse", lambda **kwargs: None)
    monkeypatch.setattr(bundle, "_responses_text", lambda **kwargs: None)
    monkeypatch.setattr(
        bundle,
        "_load_models",
        lambda: (
            _PlannerModel(
                _PlannerResponseSchema(
                    intent="author",
                    constraints=providers_module._PlannerConstraintsSchema(),
                    seedIdentifiers=[],
                    candidateConcepts=["scholar"],
                    providerPlan=["openalex"],
                    followUpMode="qa",
                )
            ),
            _SynthesizerModel(
                _AnswerPayload(
                    answer="Model answer.",
                    confidence="mixed",
                )
            ),
        ),
    )

    fallback_plan = bundle.plan_search(query="author Ada Lovelace", mode="auto")
    fallback_expansions = bundle.suggest_speculative_expansions(
        query="retrieval agents",
        evidence_texts=["biomedical graph retrieval"],
        max_variants=1,
    )
    fallback_label = bundle.label_theme(seed_terms=["graph retrieval"], papers=[])
    fallback_summary = bundle.summarize_theme(
        title="Agents",
        papers=[{"title": "A", "year": 2024, "venue": "Nature"}],
    )
    fallback_answer = bundle.answer_question(
        question="Summarize",
        evidence_papers=[{"title": "Paper A"}],
        answer_mode="qa",
    )

    assert fallback_plan.intent == "author"
    assert fallback_expansions
    assert fallback_label == "Graph Retrieval"
    assert "These papers share overlapping terms" in fallback_summary
    assert fallback_answer["confidence"] == "medium"


def test_deterministic_provider_gap_answers_and_theme_labels_are_more_specific() -> None:
    bundle = DeterministicProviderBundle(_config(provider="deterministic"))

    label = bundle.label_theme(
        seed_terms=["noise", "effects"],
        papers=[
            {"title": "Anthropogenic noise effects on wildlife"},
            {"title": "Noise impacts on birds"},
        ],
    )
    answer = bundle.answer_question(
        question="What are the main knowledge gaps highlighted across these papers?",
        evidence_papers=[
            {
                "title": "Anthropogenic noise effects on wildlife",
                "abstract": (
                    "Behavioral responses in birds dominate the evidence base, "
                    "while long-term and community consequences remain unclear."
                ),
            },
            {
                "title": "Noise exposure in mammals",
                "abstract": (
                    "Acute playback experiments emphasize behaviour more than "
                    "physiology, demography, or multistressor contexts."
                ),
            },
        ],
        answer_mode="qa",
    )

    assert "Effects" not in label
    assert "Noise" in label
    assert "long-term" in answer["answer"] or "community" in answer["answer"]


def test_openai_provider_sync_embedding_fallbacks() -> None:
    class _EmbeddingFallback:
        def embed_query(self, text: str) -> list[float]:
            if "broken" in text:
                raise RuntimeError("broken query embedding")
            return [0.2, 0.8]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError(f"broken batch for {texts!r}")

    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    bundle._embeddings = _EmbeddingFallback()
    bundle._openai_client = None

    assert bundle.embed_query("healthy query") == (0.2, 0.8)
    assert bundle.embed_query("broken query") is None
    assert bundle.embed_texts(["alpha", "beta"]) == [(0.2, 0.8), (0.2, 0.8)]


@pytest.mark.asyncio
async def test_openai_provider_async_high_level_methods_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = OpenAIProviderBundle(
        _config(),
        api_key="sk-test",
        provider_registry=ProviderDiagnosticsRegistry(),
    )

    async def _direct_parse(**kwargs: Any) -> Any:
        endpoint = kwargs["endpoint"]
        if endpoint == "responses.parse:planner":
            return _PlannerResponseSchema(
                intent="citation",
                constraints=providers_module._PlannerConstraintsSchema(),
                seedIdentifiers=["seed-1"],
                candidateConcepts=["graphs"],
                providerPlan=["semantic_scholar"],
                followUpMode="qa",
            )
        if endpoint == "responses.parse:expansions":
            return _ExpansionPayload(
                expansions=[
                    _ExpansionItem(variant="retrieval agents graphs"),
                    _ExpansionItem(variant="retrieval agents recent"),
                ]
            )
        if endpoint == "responses.parse:answer":
            return _AnswerPayload(
                answer="Async answer.",
                confidence="uncertain",
            )
        return None

    async def _direct_text(**kwargs: Any) -> str | None:
        if kwargs["endpoint"] == "responses.create:label_theme":
            return '"Async label"'
        if kwargs["endpoint"] == "responses.create:summarize_theme":
            return "Async summary."
        return None

    monkeypatch.setattr(bundle, "_aresponses_parse", _direct_parse)
    monkeypatch.setattr(bundle, "_aresponses_text", _direct_text)

    plan = await bundle.aplan_search(query="cited by agents", mode="auto")
    expansions = await bundle.asuggest_speculative_expansions(
        query="retrieval agents",
        evidence_texts=["graphs for retrieval agents"],
        max_variants=2,
    )
    label = await bundle.alabel_theme(seed_terms=["graphs"], papers=[])
    summary = await bundle.asummarize_theme(title="Agents", papers=[])
    answer = await bundle.aanswer_question(
        question="Answer",
        evidence_papers=[{"title": "Paper A"}],
        answer_mode="qa",
    )

    assert plan.intent == "citation"
    assert [item.variant for item in expansions] == ["retrieval agents graphs"]
    assert label == "Async label"
    assert summary == "Async summary."
    assert answer["confidence"] == "low"

    async def _raise_parse(**kwargs: Any) -> Any:
        raise RuntimeError(kwargs["endpoint"])

    async def _raise_text(**kwargs: Any) -> str | None:
        raise RuntimeError(kwargs["endpoint"])

    monkeypatch.setattr(bundle, "_aresponses_parse", _raise_parse)
    monkeypatch.setattr(bundle, "_aresponses_text", _raise_text)

    fallback_plan = await bundle.aplan_search(query="author Ada Lovelace", mode="auto")
    fallback_expansions = await bundle.asuggest_speculative_expansions(
        query="retrieval agents",
        evidence_texts=["citation graphs for retrieval agents"],
        max_variants=2,
    )
    fallback_label = await bundle.alabel_theme(
        seed_terms=["graph retrieval"],
        papers=[],
    )
    fallback_summary = await bundle.asummarize_theme(
        title="Agents",
        papers=[{"title": "A", "year": 2024, "venue": "Nature"}],
    )
    fallback_answer = await bundle.aanswer_question(
        question="Answer",
        evidence_papers=[],
        answer_mode="qa",
    )

    assert isinstance(fallback_plan, PlannerDecision)
    assert fallback_expansions
    assert fallback_label == "Graph Retrieval"
    assert "These papers share overlapping terms" in fallback_summary
    assert fallback_answer["confidence"] == "low"

    closed: list[str] = []

    class _AsyncClosable:
        async def aclose(self) -> None:
            closed.append("async")

    class _SyncClosable:
        def close(self) -> None:
            closed.append("sync")

    bundle._async_openai_client = _AsyncClosable()
    bundle._openai_client = _SyncClosable()
    await bundle.aclose()
    assert closed == ["async", "sync"]


def test_openai_provider_loader_failures_and_sync_wrapper_none_paths(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    no_key_bundle = OpenAIProviderBundle(_config(), api_key=None)
    assert no_key_bundle._load_openai_client() is None
    assert no_key_bundle._load_async_openai_client() is None
    assert no_key_bundle._load_models() == (None, None)
    assert no_key_bundle._load_embeddings() is None

    disabled_bundle = OpenAIProviderBundle(
        _config(disable_embeddings=True),
        api_key="sk-test",
    )
    assert disabled_bundle._load_embeddings() is None

    real_import = builtins.__import__

    def _raising_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name in {"openai", "langchain.chat_models", "langchain_openai"}:
            raise ImportError(f"blocked import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    caplog.set_level("INFO", logger="scholar-search-mcp")

    import_error_bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    assert import_error_bundle._load_openai_client() is None
    assert import_error_bundle._load_async_openai_client() is None
    assert import_error_bundle._load_models() == (None, None)
    assert import_error_bundle._load_embeddings() is None
    assert any("falling back" in record.getMessage() for record in caplog.records)

    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    bundle._openai_client = object()
    assert (
        bundle._responses_parse(
            endpoint="responses.parse:planner",
            model_name=bundle.planner_model_name,
            response_model=_PlannerResponseSchema,
            system_prompt="plan",
            payload={"query": "agents"},
        )
        is None
    )
    assert (
        bundle._responses_text(
            endpoint="responses.create:label_theme",
            model_name=bundle.synthesis_model_name,
            system_prompt="label",
            payload={"query": "agents"},
        )
        is None
    )

    bundle._openai_client = types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **kwargs: object()))
    assert (
        bundle._responses_parse(
            endpoint="responses.parse:planner",
            model_name=bundle.planner_model_name,
            response_model=_PlannerResponseSchema,
            system_prompt="plan",
            payload={"query": "agents"},
        )
        is None
    )


def test_openai_provider_sync_wrapper_error_and_embedding_edge_paths(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    bundle._openai_client = types.SimpleNamespace(
        responses=types.SimpleNamespace(
            parse=lambda **kwargs: object(),
            create=lambda **kwargs: object(),
        )
    )

    def _payload_none(**kwargs: Any) -> ProviderCallResult:
        return ProviderCallResult(
            payload=None,
            outcome=ProviderOutcomeEnvelope(
                provider="openai",
                endpoint=kwargs["endpoint"],
                status_bucket="provider_error",
                error="RuntimeError: boom",
            ),
        )

    def _invalid_payload(**kwargs: Any) -> ProviderCallResult:
        return _success_result(
            types.SimpleNamespace(
                output=[types.SimpleNamespace(content=[types.SimpleNamespace(text="not valid json")])]
            ),
            endpoint=kwargs["endpoint"],
        )

    caplog.set_level("WARNING", logger="scholar-search-mcp")
    monkeypatch.setattr(providers_module, "execute_provider_call_sync", _payload_none)
    assert (
        bundle._responses_parse(
            endpoint="responses.parse:planner",
            model_name=bundle.planner_model_name,
            response_model=_PlannerResponseSchema,
            system_prompt="plan",
            payload={"query": "agents"},
        )
        is None
    )
    assert (
        bundle._responses_text(
            endpoint="responses.create:label_theme",
            model_name=bundle.synthesis_model_name,
            system_prompt="label",
            payload={"query": "agents"},
        )
        is None
    )

    monkeypatch.setattr(providers_module, "execute_provider_call_sync", _invalid_payload)
    assert (
        bundle._responses_parse(
            endpoint="responses.parse:planner",
            model_name=bundle.planner_model_name,
            response_model=_PlannerResponseSchema,
            system_prompt="plan",
            payload={"query": "agents"},
        )
        is None
    )
    assert any("OpenAI Responses parse failed" in record.getMessage() for record in caplog.records)

    disabled_bundle = OpenAIProviderBundle(
        _config(disable_embeddings=True),
        api_key="sk-test",
    )
    lexical_score = _lexical_similarity("retrieval agents", "retrieval graphs")
    assert disabled_bundle.embed_query("retrieval agents") is None
    assert disabled_bundle.embed_texts(["retrieval agents"]) == [None]
    assert disabled_bundle.similarity(
        "retrieval agents",
        "retrieval graphs",
    ) == pytest.approx(lexical_score)
    assert disabled_bundle.batched_similarity(
        "retrieval agents",
        ["retrieval graphs"],
    ) == [pytest.approx(lexical_score)]

    no_key_bundle = OpenAIProviderBundle(_config(), api_key=None)
    assert no_key_bundle.embed_query("   ") is None
    assert no_key_bundle.embed_query("retrieval agents") is None
    assert no_key_bundle.embed_texts(["retrieval agents"]) == [None]

    class _DocumentEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            del text
            return [0.1, 0.9]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.9] for _ in texts]

    embeddings_bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    embeddings_bundle._openai_client = None
    embeddings_bundle._embeddings = _DocumentEmbeddings()
    assert embeddings_bundle.embed_texts(["alpha", "beta"]) == [
        (0.1, 0.9),
        (0.1, 0.9),
    ]

    partial_bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    partial_bundle.embed_query = (  # type: ignore[assignment,method-assign]
        lambda text: (1.0, 0.0) if text == "query" else (0.0, 1.0)
    )
    partial_bundle.embed_texts = (  # type: ignore[assignment,method-assign]
        lambda texts: [None, (1.0, 0.0)]
    )
    scores = partial_bundle.batched_similarity(
        "query",
        ["query overlap", "query overlap"],
    )
    assert scores[0] < scores[1]


@pytest.mark.asyncio
async def test_openai_provider_async_wrapper_and_embedding_success_paths(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    bundle._log_embedding_batch_start(
        request_id=None,
        total_texts=1,
        uncached_texts=1,
    )
    bundle._log_embedding_batch_failure(
        request_id=None,
        total_texts=1,
        uncached_texts=1,
        status_bucket="provider_error",
        reason="boom",
    )

    bundle._async_openai_client = object()
    assert (
        await bundle._aresponses_parse(
            endpoint="responses.parse:planner",
            model_name=bundle.planner_model_name,
            response_model=_PlannerResponseSchema,
            system_prompt="plan",
            payload={"query": "agents"},
        )
        is None
    )
    assert (
        await bundle._aresponses_text(
            endpoint="responses.create:label_theme",
            model_name=bundle.synthesis_model_name,
            system_prompt="label",
            payload={"query": "agents"},
        )
        is None
    )

    class _AsyncResponses:
        async def create(self, **kwargs: Any) -> Any:
            del kwargs
            return types.SimpleNamespace(output_text="async")

    bundle._async_openai_client = types.SimpleNamespace(responses=_AsyncResponses())
    assert (
        await bundle._aresponses_parse(
            endpoint="responses.parse:planner",
            model_name=bundle.planner_model_name,
            response_model=_PlannerResponseSchema,
            system_prompt="plan",
            payload={"query": "agents"},
        )
        is None
    )

    async def _payload_none(**kwargs: Any) -> ProviderCallResult:
        return ProviderCallResult(
            payload=None,
            outcome=ProviderOutcomeEnvelope(
                provider="openai",
                endpoint=kwargs["endpoint"],
                status_bucket="provider_error",
                error="RuntimeError: async boom",
            ),
        )

    async def _success_async(**kwargs: Any) -> ProviderCallResult:
        return _success_result(await kwargs["operation"](), endpoint=kwargs["endpoint"])

    async def _invalid_async(**kwargs: Any) -> ProviderCallResult:
        return _success_result(
            types.SimpleNamespace(
                output=[types.SimpleNamespace(content=[types.SimpleNamespace(text="not valid json")])]
            ),
            endpoint=kwargs["endpoint"],
        )

    class _AsyncFullResponses:
        async def parse(self, **kwargs: Any) -> Any:
            del kwargs
            return types.SimpleNamespace(output=[])

        async def create(self, **kwargs: Any) -> Any:
            del kwargs
            return types.SimpleNamespace(output_text="async text")

    caplog.set_level("WARNING", logger="scholar-search-mcp")
    bundle._async_openai_client = types.SimpleNamespace(responses=_AsyncFullResponses())
    monkeypatch.setattr(providers_module, "execute_provider_call", _payload_none)
    assert (
        await bundle._aresponses_parse(
            endpoint="responses.parse:planner",
            model_name=bundle.planner_model_name,
            response_model=_PlannerResponseSchema,
            system_prompt="plan",
            payload={"query": "agents"},
        )
        is None
    )
    assert (
        await bundle._aresponses_text(
            endpoint="responses.create:label_theme",
            model_name=bundle.synthesis_model_name,
            system_prompt="label",
            payload={"query": "agents"},
        )
        is None
    )

    monkeypatch.setattr(providers_module, "execute_provider_call", _invalid_async)
    assert (
        await bundle._aresponses_parse(
            endpoint="responses.parse:planner",
            model_name=bundle.planner_model_name,
            response_model=_PlannerResponseSchema,
            system_prompt="plan",
            payload={"query": "agents"},
        )
        is None
    )
    assert any("OpenAI Responses parse failed" in record.getMessage() for record in caplog.records)

    class _EmbeddingsClient:
        async def create(self, *, model: str, input: Any) -> dict[str, Any]:
            del model
            if isinstance(input, list):
                return {"data": [{"embedding": [1.0, 0.0] if "retrieval" in text else [0.0, 1.0]} for text in input]}
            return {"data": [{"embedding": [1.0, 0.0]}]}

    async_bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    async_bundle._async_openai_client = types.SimpleNamespace(embeddings=_EmbeddingsClient())
    monkeypatch.setattr(providers_module, "execute_provider_call", _success_async)
    query_embedding = await async_bundle.aembed_query(
        "retrieval agents",
        request_id="async-emb",
    )
    text_embeddings = await async_bundle.aembed_texts(
        ["retrieval agents", "bird migration"],
        request_id="async-emb",
    )
    scores = await async_bundle.abatched_similarity(
        "retrieval agents",
        ["retrieval agents", "bird migration"],
        request_id="async-emb",
    )
    disabled_bundle = OpenAIProviderBundle(
        _config(disable_embeddings=True),
        api_key="sk-test",
    )

    assert query_embedding == (1.0, 0.0)
    assert text_embeddings == [(1.0, 0.0), (0.0, 1.0)]
    assert scores[0] > scores[1]
    assert await disabled_bundle.aembed_texts(["alpha"]) == [None]
