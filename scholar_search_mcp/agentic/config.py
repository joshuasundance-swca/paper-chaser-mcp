"""Configuration and thresholds for additive smart research workflows."""

from __future__ import annotations

from dataclasses import dataclass

from ..settings import AppSettings


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
    max_grounded_variants: int = 3
    max_speculative_variants: int = 3
    max_total_variants: int = 6
    candidate_pool_size: int = 80
    speculative_accept_min_novel_papers: int = 2
    speculative_top_pool_cutoff: int = 40
    drift_similarity_threshold: float = 0.16
    landscape_min_themes: int = 3
    landscape_max_themes: int = 5

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "AgenticConfig":
        return cls(
            enabled=settings.enable_agentic,
            provider=settings.agentic_provider,
            planner_model=settings.planner_model,
            synthesis_model=settings.synthesis_model,
            embedding_model=settings.embedding_model,
            index_backend=settings.agentic_index_backend,
            session_ttl_seconds=settings.session_ttl_seconds,
            enable_trace_log=settings.enable_agentic_trace_log,
        )
