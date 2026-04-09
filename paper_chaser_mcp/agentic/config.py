"""Configuration and thresholds for additive smart research workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..settings import AppSettings

LatencyProfile = Literal["fast", "balanced", "deep"]
ModelSelectionSource = Literal["configured", "azure_deployment", "provider_default", "deterministic"]

_DEFAULT_OPENAI_PLANNER_MODEL = "gpt-5.4-mini"
_DEFAULT_OPENAI_SYNTHESIS_MODEL = "gpt-5.4"
_PROVIDER_DEFAULT_MODELS: dict[str, tuple[str, str]] = {
    "anthropic": ("claude-haiku-4-5", "claude-sonnet-4-6"),
    "nvidia": ("nvidia/nemotron-3-nano-30b-a3b", "nvidia/nemotron-3-super-120b-a12b"),
    "google": ("gemini-2.5-flash", "gemini-2.5-pro"),
    "mistral": ("mistral-medium-latest", "mistral-large-latest"),
    "huggingface": ("moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.5"),
}


@dataclass(frozen=True)
class LatencyProfileSettings:
    """Derived execution settings for one smart-tool latency profile."""

    name: LatencyProfile
    search_config: "AgenticConfig"
    use_deterministic_bundle: bool
    allow_serpapi_on_input: bool
    allow_serpapi_on_expansions: bool
    enable_speculative_expansions: bool
    enable_deep_recommendations: bool
    use_embedding_rerank: bool


@dataclass(frozen=True)
class AgenticConfig:
    """Runtime configuration for the additive smart-tool layer."""

    enabled: bool
    provider: str
    planner_model: str
    synthesis_model: str
    embedding_model: str
    index_backend: str
    session_ttl_seconds: int
    enable_trace_log: bool
    disable_embeddings: bool = True
    openai_timeout_seconds: float = 30.0
    planner_model_source: ModelSelectionSource = "configured"
    synthesis_model_source: ModelSelectionSource = "configured"
    max_grounded_variants: int = 3
    max_speculative_variants: int = 3
    max_total_variants: int = 6
    max_initial_hypotheses: int = 3
    candidate_pool_size: int = 80
    speculative_accept_min_novel_papers: int = 2
    speculative_top_pool_cutoff: int = 40
    drift_similarity_threshold: float = 0.16
    landscape_min_themes: int = 3
    landscape_max_themes: int = 5

    def for_latency_profile(self, profile: LatencyProfile) -> "AgenticConfig":
        """Return a copy tuned for fast, balanced, or deep execution."""

        if profile == "fast":
            return replace(
                self,
                max_grounded_variants=min(self.max_grounded_variants, 1),
                max_speculative_variants=0,
                max_total_variants=min(self.max_total_variants, 2),
                max_initial_hypotheses=min(self.max_initial_hypotheses, 2),
                candidate_pool_size=min(self.candidate_pool_size, 40),
                speculative_top_pool_cutoff=min(self.speculative_top_pool_cutoff, 20),
            )
        if profile == "deep":
            return replace(
                self,
                max_grounded_variants=min(self.max_grounded_variants + 1, 4),
                max_speculative_variants=min(self.max_speculative_variants + 1, 4),
                max_total_variants=min(self.max_total_variants + 2, 8),
                max_initial_hypotheses=min(self.max_initial_hypotheses + 1, 4),
                candidate_pool_size=min(self.candidate_pool_size + 20, 120),
                speculative_top_pool_cutoff=min(
                    self.speculative_top_pool_cutoff + 20,
                    80,
                ),
            )
        return replace(
            self,
            max_grounded_variants=min(self.max_grounded_variants, 2),
            max_speculative_variants=min(self.max_speculative_variants, 2),
            max_total_variants=min(self.max_total_variants, 4),
            max_initial_hypotheses=min(self.max_initial_hypotheses, 3),
            candidate_pool_size=min(self.candidate_pool_size, 50),
            speculative_top_pool_cutoff=min(self.speculative_top_pool_cutoff, 30),
        )

    def latency_profile_settings(
        self,
        profile: LatencyProfile,
    ) -> LatencyProfileSettings:
        """Return derived toggles for the requested latency profile."""

        tuned = self.for_latency_profile(profile)
        if profile == "fast":
            return LatencyProfileSettings(
                name=profile,
                search_config=tuned,
                use_deterministic_bundle=True,
                allow_serpapi_on_input=False,
                allow_serpapi_on_expansions=False,
                enable_speculative_expansions=False,
                enable_deep_recommendations=False,
                use_embedding_rerank=False,
            )
        if profile == "deep":
            return LatencyProfileSettings(
                name=profile,
                search_config=tuned,
                use_deterministic_bundle=False,
                allow_serpapi_on_input=True,
                allow_serpapi_on_expansions=True,
                enable_speculative_expansions=True,
                enable_deep_recommendations=True,
                use_embedding_rerank=not self.disable_embeddings,
            )
        return LatencyProfileSettings(
            name=profile,
            search_config=tuned,
            use_deterministic_bundle=False,
            allow_serpapi_on_input=True,
            allow_serpapi_on_expansions=False,
            enable_speculative_expansions=True,
            enable_deep_recommendations=False,
            use_embedding_rerank=False,
        )

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "AgenticConfig":
        planner_model = settings.planner_model
        synthesis_model = settings.synthesis_model
        planner_model_source: ModelSelectionSource = "configured"
        synthesis_model_source: ModelSelectionSource = "configured"

        if settings.agentic_provider == "azure-openai":
            if settings.azure_openai_planner_deployment:
                planner_model = settings.azure_openai_planner_deployment
                planner_model_source = "azure_deployment"
            if settings.azure_openai_synthesis_deployment:
                synthesis_model = settings.azure_openai_synthesis_deployment
                synthesis_model_source = "azure_deployment"
        elif (
            settings.agentic_provider in _PROVIDER_DEFAULT_MODELS
            and planner_model == _DEFAULT_OPENAI_PLANNER_MODEL
            and synthesis_model == _DEFAULT_OPENAI_SYNTHESIS_MODEL
        ):
            planner_model, synthesis_model = _PROVIDER_DEFAULT_MODELS[settings.agentic_provider]
            planner_model_source = "provider_default"
            synthesis_model_source = "provider_default"

        return cls(
            enabled=settings.enable_agentic,
            provider=settings.agentic_provider,
            planner_model=planner_model,
            synthesis_model=synthesis_model,
            embedding_model=settings.embedding_model,
            disable_embeddings=settings.disable_embeddings,
            openai_timeout_seconds=settings.agentic_openai_timeout_seconds,
            planner_model_source=planner_model_source,
            synthesis_model_source=synthesis_model_source,
            index_backend=settings.agentic_index_backend,
            session_ttl_seconds=settings.session_ttl_seconds,
            enable_trace_log=settings.enable_agentic_trace_log,
        )
