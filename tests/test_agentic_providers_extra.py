import builtins
import sys
import types
from typing import Any

import pytest
from pydantic import BaseModel, Field

from paper_chaser_mcp.agentic import providers as providers_module
from paper_chaser_mcp.agentic.config import AgenticConfig
from paper_chaser_mcp.agentic.models import ExpansionCandidate, PlannerDecision
from paper_chaser_mcp.agentic.providers import (
    COMMON_QUERY_WORDS,
    AnthropicProviderBundle,
    AzureOpenAIProviderBundle,
    DeterministicProviderBundle,
    GoogleProviderBundle,
    OpenAIProviderBundle,
    _coerce_langchain_structured_response,
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
from paper_chaser_mcp.provider_runtime import (
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
    assert "scholarapi" in known_item.provider_plan
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
            # Return expansion-compatible payload when the schema has an
            # ``expansions`` field so the LangChain fallback path inside
            # suggest_speculative_expansions is exercised correctly.
            if any(f == "expansions" for f in getattr(schema, "model_fields", {})):
                return _StructuredInvoker(
                    _ExpansionPayload(expansions=[_ExpansionItem(variant="langchain fallback expansion")])
                )
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
    assert fallback_expansions[0].variant == "langchain fallback expansion"
    assert fallback_label == "Graph Retrieval"
    assert "groups 1 papers" in fallback_summary
    assert fallback_answer["confidence"] == "medium"


def test_openai_provider_sync_label_normalizes_prefixed_markdown_text() -> None:
    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")

    bundle._responses_text = (  # type: ignore[method-assign]
        lambda **kwargs: '## Theme: "Graph Retrieval"'
    )

    label = bundle.label_theme(seed_terms=["graph retrieval"], papers=[])

    assert label == "Graph Retrieval"


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
    assert "groups 1 papers" in fallback_summary
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


@pytest.mark.asyncio
async def test_openai_provider_async_label_normalizes_prefixed_markdown_text() -> None:
    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")

    async def _text_json(**kwargs: Any) -> str:
        del kwargs
        return '## Theme: "Async Graph Retrieval"'

    bundle._aresponses_text = _text_json  # type: ignore[method-assign]

    label = await bundle.alabel_theme(seed_terms=["graph retrieval"], papers=[])

    assert label == "Async Graph Retrieval"


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
    caplog.set_level("INFO", logger="paper-chaser-mcp")

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

    caplog.set_level("WARNING", logger="paper-chaser-mcp")
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

    caplog.set_level("WARNING", logger="paper-chaser-mcp")
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


def test_resolve_provider_bundle_routes_additional_provider_types() -> None:
    azure_bundle = resolve_provider_bundle(
        _config(provider="azure-openai"),
        openai_api_key=None,
        azure_openai_api_key="azure-key",
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_api_version="2024-10-21",
    )
    anthropic_bundle = resolve_provider_bundle(
        _config(provider="anthropic"),
        openai_api_key=None,
        anthropic_api_key="sk-ant-test",
    )
    google_bundle = resolve_provider_bundle(
        _config(provider="google"),
        openai_api_key=None,
        google_api_key="google-key",
    )

    assert isinstance(azure_bundle, AzureOpenAIProviderBundle)
    assert isinstance(anthropic_bundle, AnthropicProviderBundle)
    assert isinstance(google_bundle, GoogleProviderBundle)


def test_azure_openai_provider_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}

    class _FakeAzureOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            created["sync"] = kwargs

    class _FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            created["async"] = kwargs

    class _FakeAzureChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            created.setdefault("chat", []).append(kwargs)

    openai_module = types.ModuleType("openai")
    openai_module.AzureOpenAI = _FakeAzureOpenAI  # type: ignore[attr-defined]
    openai_module.AsyncAzureOpenAI = _FakeAsyncAzureOpenAI  # type: ignore[attr-defined]
    langchain_openai_module = types.ModuleType("langchain_openai")
    langchain_openai_module.AzureChatOpenAI = _FakeAzureChatOpenAI  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai_module)

    bundle = AzureOpenAIProviderBundle(
        _config(provider="azure-openai"),
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-21",
    )

    assert bundle._load_openai_client() is not None
    assert bundle._load_async_openai_client() is not None
    planner, synthesizer = bundle._load_models()

    assert planner is not None
    assert synthesizer is not None
    assert created["sync"]["api_key"] == "azure-key"
    assert created["sync"]["azure_endpoint"] == "https://example.openai.azure.com/"
    assert created["sync"]["api_version"] == "2024-10-21"
    assert created["chat"][0]["azure_deployment"] == bundle.planner_model_name
    assert created["chat"][1]["azure_deployment"] == bundle.synthesis_model_name


def test_azure_openai_provider_uses_deployment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}

    class _FakeAzureChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            created.setdefault("chat", []).append(kwargs)

    langchain_openai_module = types.ModuleType("langchain_openai")
    langchain_openai_module.AzureChatOpenAI = _FakeAzureChatOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai_module)

    bundle = AzureOpenAIProviderBundle(
        _config(provider="azure-openai"),
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-21",
        azure_planner_deployment="planner-deployment",
        azure_synthesis_deployment="synthesis-deployment",
    )

    bundle._load_models()

    assert bundle.planner_model_name == "planner-deployment"
    assert bundle.synthesis_model_name == "synthesis-deployment"
    assert created["chat"][0]["azure_deployment"] == "planner-deployment"
    assert created["chat"][1]["azure_deployment"] == "synthesis-deployment"


def test_azure_openai_provider_loads_azure_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}

    class _FakeAzureOpenAIEmbeddings:
        def __init__(self, **kwargs: Any) -> None:
            created.update(kwargs)

    langchain_openai_module = types.ModuleType("langchain_openai")
    langchain_openai_module.AzureOpenAIEmbeddings = _FakeAzureOpenAIEmbeddings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai_module)

    bundle = AzureOpenAIProviderBundle(
        _config(provider="azure-openai", disable_embeddings=False),
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-21",
    )

    embeddings = bundle._load_embeddings()

    assert embeddings is not None
    assert created["azure_endpoint"] == "https://example.openai.azure.com/"
    assert created["api_version"] == "2024-10-21"
    assert created["azure_deployment"] == bundle.embedding_model_name
    assert created["model"] == bundle.embedding_model_name


@pytest.mark.parametrize(
    ("bundle", "expected"),
    [
        (OpenAIProviderBundle(_config(disable_embeddings=False), api_key="sk-test"), True),
        (
            AzureOpenAIProviderBundle(
                _config(provider="azure-openai", disable_embeddings=False),
                api_key="azure-key",
                azure_endpoint="https://example.openai.azure.com/",
                api_version="2024-10-21",
            ),
            True,
        ),
        (
            AnthropicProviderBundle(
                _config(provider="anthropic", disable_embeddings=False),
                api_key="sk-ant-test",
            ),
            False,
        ),
        (
            GoogleProviderBundle(
                _config(provider="google", disable_embeddings=False),
                api_key="google-key",
            ),
            False,
        ),
    ],
)
def test_provider_embedding_support_flags(bundle: Any, expected: bool) -> None:
    assert bundle.supports_embeddings() is expected


@pytest.mark.parametrize(
    "bundle",
    [
        AzureOpenAIProviderBundle(
            _config(provider="azure-openai"),
            api_key=None,
            azure_endpoint=None,
            api_version=None,
        ),
        AnthropicProviderBundle(_config(provider="anthropic"), api_key=None),
        GoogleProviderBundle(_config(provider="google"), api_key=None),
    ],
)
def test_additional_provider_selection_metadata_reports_deterministic_fallback(bundle: Any) -> None:
    plan = bundle.plan_search(query="author Ada Lovelace", mode="auto")
    metadata = bundle.selection_metadata()

    assert isinstance(plan, PlannerDecision)
    assert metadata["configuredSmartProvider"] in {"azure-openai", "anthropic", "google"}
    assert metadata["activeSmartProvider"] == "deterministic"
    assert metadata["plannerModelSource"] == "deterministic"
    assert metadata["synthesisModelSource"] == "deterministic"
    assert metadata["plannerModel"].endswith(":deterministic-planner")
    assert metadata["synthesisModel"].endswith(":deterministic-synthesizer")


def test_openai_provider_reloads_models_when_cache_is_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []

    def _fake_init_chat_model(**kwargs: Any) -> dict[str, str]:
        created.append(kwargs["model"])
        return {"model": kwargs["model"]}

    langchain_module = types.ModuleType("langchain")
    chat_models_module = types.ModuleType("langchain.chat_models")
    chat_models_module.init_chat_model = _fake_init_chat_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain", langchain_module)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", chat_models_module)

    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    bundle._planner = object()
    bundle._synthesizer = None

    planner, synthesizer = bundle._load_models()

    assert created == [bundle.planner_model_name, bundle.synthesis_model_name]
    assert planner == {"model": bundle.planner_model_name}
    assert synthesizer == {"model": bundle.synthesis_model_name}


def test_azure_openai_provider_reloads_models_when_cache_is_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, Any] = {}

    class _FakeAzureChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            created.setdefault("chat", []).append(kwargs)

    langchain_openai_module = types.ModuleType("langchain_openai")
    langchain_openai_module.AzureChatOpenAI = _FakeAzureChatOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai_module)

    bundle = AzureOpenAIProviderBundle(
        _config(provider="azure-openai"),
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-21",
    )
    bundle._planner = object()
    bundle._synthesizer = None

    planner, synthesizer = bundle._load_models()

    assert planner is not None
    assert synthesizer is not None
    assert len(created["chat"]) == 2


def test_langchain_provider_reloads_models_when_cache_is_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []
    bundle = AnthropicProviderBundle(_config(provider="anthropic"), api_key="sk-ant-test")

    def _fake_create_chat_model(model_name: str) -> dict[str, str]:
        created.append(model_name)
        return {"model": model_name}

    monkeypatch.setattr(bundle, "_create_chat_model", _fake_create_chat_model)
    bundle._planner = object()
    bundle._synthesizer = None

    planner, synthesizer = bundle._load_models()

    assert created == [bundle.planner_model_name, bundle.synthesis_model_name]
    assert planner == {"model": bundle.planner_model_name}
    assert synthesizer == {"model": bundle.synthesis_model_name}


def test_azure_openai_sync_methods_skip_langchain_chat_fallback_when_responses_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = AzureOpenAIProviderBundle(
        _config(provider="azure-openai"),
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-21",
        azure_planner_deployment="planner-deployment",
        azure_synthesis_deployment="synthesis-deployment",
    )

    class _ExplodingModel:
        def invoke(self, messages: list[tuple[str, str]]) -> Any:
            raise AssertionError("LangChain chat fallback should not be invoked for Azure")

        def with_structured_output(self, schema: Any, method: str | None = None) -> Any:
            del schema, method
            raise AssertionError("Structured fallback should not be invoked for Azure")

    monkeypatch.setattr(bundle, "_responses_parse", lambda **kwargs: None)
    monkeypatch.setattr(bundle, "_responses_text", lambda **kwargs: None)
    monkeypatch.setattr(bundle, "_load_models", lambda: (_ExplodingModel(), _ExplodingModel()))

    plan = bundle.plan_search(query="author Ada Lovelace", mode="auto")
    expansions = bundle.suggest_speculative_expansions(
        query="retrieval agents",
        evidence_texts=["citation graphs for retrieval agents"],
        max_variants=2,
    )
    label = bundle.label_theme(seed_terms=["graph retrieval"], papers=[])
    summary = bundle.summarize_theme(title="Graph Retrieval", papers=[])
    answer = bundle.answer_question(
        question="What matters?",
        evidence_papers=[{"title": "Paper A"}],
        answer_mode="qa",
    )

    assert isinstance(plan, PlannerDecision)
    assert isinstance(expansions, list)
    assert label == "Graph Retrieval"
    assert summary
    assert isinstance(answer, dict)


def test_azure_openai_answer_uses_native_text_fallback_before_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = AzureOpenAIProviderBundle(
        _config(provider="azure-openai"),
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2024-10-21",
        azure_planner_deployment="planner-deployment",
        azure_synthesis_deployment="synthesis-deployment",
    )

    class _ExplodingModel:
        def invoke(self, messages: list[tuple[str, str]]) -> Any:
            raise AssertionError("LangChain chat fallback should not be invoked for Azure")

        def with_structured_output(self, schema: Any, method: str | None = None) -> Any:
            del schema, method
            raise AssertionError("Structured fallback should not be invoked for Azure")

    monkeypatch.setattr(bundle, "_responses_parse", lambda **kwargs: None)
    recovered_answer = (
        '{"answer":"Recovered Azure answer","unsupportedAsks":[],"followUpQuestions":[],"confidence":"high"}'
    )
    monkeypatch.setattr(
        bundle,
        "_responses_text",
        lambda **kwargs: recovered_answer,
    )
    monkeypatch.setattr(bundle, "_load_models", lambda: (_ExplodingModel(), _ExplodingModel()))

    answer = bundle.answer_question(
        question="What matters?",
        evidence_papers=[{"title": "Paper A"}],
        answer_mode="qa",
    )

    assert answer["answer"] == "Recovered Azure answer"
    assert answer["confidence"] == "high"


@pytest.mark.asyncio
async def test_openai_async_answer_uses_native_text_fallback_before_deterministic() -> None:
    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")

    async def _parse_none(**kwargs: Any) -> None:
        del kwargs
        return None

    async def _text_json(**kwargs: Any) -> str:
        del kwargs
        return '{"answer":"Recovered async answer","unsupportedAsks":[],"followUpQuestions":[],"confidence":"medium"}'

    bundle._aresponses_parse = _parse_none  # type: ignore[method-assign]
    bundle._aresponses_text = _text_json  # type: ignore[method-assign]

    answer = await bundle.aanswer_question(
        question="What matters?",
        evidence_papers=[{"title": "Paper A"}],
        answer_mode="qa",
    )

    assert answer["answer"] == "Recovered async answer"
    assert answer["confidence"] == "medium"


class _StructuredInvoker:
    def __init__(self, model: "_SequenceModel") -> None:
        self._model = model

    def invoke(self, messages: list[tuple[str, str]]) -> Any:
        assert messages
        return self._model.structured_responses.pop(0)

    async def ainvoke(self, messages: list[tuple[str, str]]) -> Any:
        assert messages
        return self._model.structured_responses.pop(0)


class _SequenceModel:
    def __init__(self, *, structured_responses: list[Any], text_responses: list[str]) -> None:
        self.structured_responses = list(structured_responses)
        self.text_responses = list(text_responses)
        self.methods: list[str | None] = []

    def with_structured_output(self, schema: Any, method: str | None = None) -> _StructuredInvoker:
        assert schema is not None
        self.methods.append(method)
        return _StructuredInvoker(self)

    def invoke(self, messages: list[tuple[str, str]]) -> Any:
        assert messages
        return types.SimpleNamespace(content=self.text_responses.pop(0))

    async def ainvoke(self, messages: list[tuple[str, str]]) -> Any:
        assert messages
        return types.SimpleNamespace(content=self.text_responses.pop(0))


def test_coerce_langchain_structured_response_extracts_json_from_markdown() -> None:
    response = types.SimpleNamespace(
        content='```json\n{"answer":"Grounded","unsupportedAsks":[],"followUpQuestions":[],"confidence":"high"}\n```'
    )

    parsed = _coerce_langchain_structured_response(response, _AnswerPayload)

    assert parsed.answer == "Grounded"
    assert parsed.confidence == "high"


@pytest.mark.parametrize(
    ("bundle", "expected_method"),
    [
        (AnthropicProviderBundle(_config(provider="anthropic"), api_key="sk-ant-test"), None),
        (GoogleProviderBundle(_config(provider="google"), api_key="google-key"), "json_schema"),
    ],
)
def test_langchain_provider_bundles_sync_high_level_methods(
    monkeypatch: pytest.MonkeyPatch,
    bundle: Any,
    expected_method: str | None,
) -> None:
    planner = _SequenceModel(
        structured_responses=[
            _PlannerResponseSchema(
                intent="review",
                constraints=providers_module._PlannerConstraintsSchema(),
                seedIdentifiers=[],
                candidateConcepts=["agents"],
                providerPlan=["semantic_scholar"],
                followUpMode="qa",
            ),
            _ExpansionPayload(expansions=[_ExpansionItem(variant="retrieval agents citation")]),
        ],
        text_responses=[],
    )
    synthesizer = _SequenceModel(
        structured_responses=[_AnswerPayload(answer="Grounded answer.", confidence="medium")],
        text_responses=['"Theme"', "Summary text."],
    )
    monkeypatch.setattr(bundle, "_load_models", lambda: (planner, synthesizer))

    plan = bundle.plan_search(query="agents", mode="auto")
    expansions = bundle.suggest_speculative_expansions(
        query="retrieval agents",
        evidence_texts=["citation graphs for retrieval agents"],
        max_variants=2,
    )
    label = bundle.label_theme(seed_terms=["agents"], papers=[])
    summary = bundle.summarize_theme(title="Agents", papers=[])
    answer = bundle.answer_question(
        question="What matters?",
        evidence_papers=[{"title": "Paper A"}],
        answer_mode="qa",
    )

    assert plan.intent == "review"
    assert [item.variant for item in expansions] == ["retrieval agents citation"]
    assert label == "Theme"
    assert summary == "Summary text."
    assert answer["answer"] == "Grounded answer."
    assert planner.methods == [expected_method, expected_method]
    assert synthesizer.methods == [expected_method]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bundle", "expected_method"),
    [
        (AnthropicProviderBundle(_config(provider="anthropic"), api_key="sk-ant-test"), None),
        (GoogleProviderBundle(_config(provider="google"), api_key="google-key"), "json_schema"),
    ],
)
async def test_langchain_provider_bundles_async_high_level_methods(
    monkeypatch: pytest.MonkeyPatch,
    bundle: Any,
    expected_method: str | None,
) -> None:
    planner = _SequenceModel(
        structured_responses=[
            _PlannerResponseSchema(
                intent="citation",
                constraints=providers_module._PlannerConstraintsSchema(),
                seedIdentifiers=["seed-1"],
                candidateConcepts=["graphs"],
                providerPlan=["semantic_scholar"],
                followUpMode="qa",
            ),
            _ExpansionPayload(expansions=[_ExpansionItem(variant="retrieval agents graphs")]),
        ],
        text_responses=[],
    )
    synthesizer = _SequenceModel(
        structured_responses=[_AnswerPayload(answer="Async answer.", confidence="medium")],
        text_responses=['"Async label"', "Async summary."],
    )
    monkeypatch.setattr(bundle, "_load_models", lambda: (planner, synthesizer))

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
    assert answer["answer"] == "Async answer."
    assert planner.methods == [expected_method, expected_method]
    assert synthesizer.methods == [expected_method]


@pytest.mark.parametrize(
    "bundle",
    [
        AnthropicProviderBundle(_config(provider="anthropic"), api_key=None),
        GoogleProviderBundle(_config(provider="google"), api_key=None),
    ],
)
def test_langchain_provider_bundles_fallback_without_credentials(bundle: Any) -> None:
    plan = bundle.plan_search(query="author Ada Lovelace", mode="auto")
    label = bundle.label_theme(seed_terms=["graph retrieval"], papers=[])

    assert isinstance(plan, PlannerDecision)
    assert label == "Graph Retrieval"


@pytest.mark.parametrize(
    "bundle",
    [
        AnthropicProviderBundle(_config(provider="anthropic"), api_key="sk-ant-test"),
        GoogleProviderBundle(_config(provider="google"), api_key="google-key"),
    ],
)
def test_langchain_provider_label_normalization_and_text_json_fallback(
    monkeypatch: pytest.MonkeyPatch,
    bundle: Any,
) -> None:
    planner = _SequenceModel(
        structured_responses=[],
        text_responses=[],
    )
    synthesizer = _SequenceModel(
        structured_responses=[],
        text_responses=[
            "# Retrieval-Augmented AI Systems\n\nThis theme covers retrieval-backed agents.",
            '{"answer":"Recovered answer","unsupportedAsks":[],"followUpQuestions":[],"confidence":"high"}',
        ],
    )
    monkeypatch.setattr(bundle, "_load_models", lambda: (planner, synthesizer))
    monkeypatch.setattr(bundle, "_structured_sync", lambda **kwargs: None)

    label = bundle.label_theme(seed_terms=["retrieval"], papers=[])
    answer = bundle.answer_question(
        question="What matters?",
        evidence_papers=[{"title": "Paper A"}],
        answer_mode="qa",
    )

    assert label == "Retrieval-Augmented AI Systems"
    assert answer["answer"] == "Recovered answer"
    assert answer["confidence"] == "high"
