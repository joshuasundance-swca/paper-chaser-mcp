# Paper-Chaser UX Remediation Checklist

Source: live UX testing (16 guided tool calls via Copilot CLI MCP) plus 7 targeted
code investigations against `main`. Each item cites the file:line where the
problem lives and where the fix should go, with tests to add/update.

Guided tools are the public contract, so remediations target `research`,
`follow_up_research`, `resolve_reference`, `inspect_source`, and
`get_runtime_status` behavior first; deeper fixes land in the agentic/ranking
stack that feeds them.

## UX test summary (baseline)

| Area | Grade | Symptom |
| --- | --- | --- |
| Hallucination resistance | B | `answerStatus=answered` promoted on weak/deterministic synthesis |
| Full-text reality | D+ | `fullTextObserved` / `full_text_verified` set from URL discovery, not body text |
| Payload efficiency | C- | 7/16 follow-ups spilled to temp files (29-35KB); legacy fields always serialized |
| Schema clarity | D | `coverageSummary`, `trustSummary`, `sourceAlias`, `topicalRelevance` contradict each other across tools |
| Regulatory routing | C+ | Works, but `inspect_source` vs `research` disagree on "on_topic" for the same FR doc |
| Reference resolution | C- | Watson/Crick natural-language citation → `no_match` while DOI works |

## Priority buckets

- **P0 — Correctness / trust bugs.** Fix the lies (answered-on-weak, full-text-implies-body-text, schema self-contradictions). These directly mislead agents.
- **P1 — Contract reshaping.** Payload slimdown (compact follow-ups, legacy opt-in) and `topRecommendation` for comparative follow-ups. Ship behind flags but flip the defaults.
- **P2 — Ranking / recall.** Conceptual-query foundational boost, citation repair for natural-language classics. Quality improvements, not correctness bugs.

---

## P0-1 · Stop promoting `answered` / `grounded` on weak or deterministic synthesis

**Symptom:** `ask_result_set` and `follow_up_research` emit `answerStatus=answered`, `answerability=grounded` even when (a) provider is the deterministic boilerplate bundle, (b) selected evidence is `weak_match`/`off_topic`, or (c) synthesis mode is metadata-only.

**Root causes:**
- `paper_chaser_mcp/agentic/graphs.py:1765-1766` — promotes `answerability` to `grounded` whenever `answer_status == "answered"` and any `selectedEvidenceIds` exist, regardless of provider or topical relevance.
- `paper_chaser_mcp/dispatch.py:4855` — `_answer_follow_up_from_session_state` unconditionally emits `answerStatus="answered"` for metadata / relevance-triage modes without gating on session evidence quality.
- `paper_chaser_mcp/agentic/provider_base.py:818-833` — `DeterministicProviderBundle.answer_question` returns `answerability="grounded"` for the default Q&A path regardless of evidence count or confidence.
- `paper_chaser_mcp/agentic/graphs.py:1557-1566` — `selectedEvidenceIds` validation only requires non-empty identifiers; no filter against `topical_relevance` or `verification_status`.

**Fix:**
- Gate the `graphs.py:1765-1766` promotion on: `provider_used != "deterministic"` AND ≥1 source with `topicalRelevance="on_topic"` AND `verificationStatus ∈ {verified_primary_source, verified_metadata}` AND `confidence ∈ {high, medium}`. Otherwise downgrade to `limited`.
- In `_answer_follow_up_from_session_state` (`dispatch.py:4628-4907`), require ≥1 on-topic verified source in the saved session before emitting `answerStatus="answered"`; else return `insufficient_evidence`.
- In `DeterministicProviderBundle.answer_question`, require ≥2 on-topic papers and medium+ computed confidence before returning `answerability="grounded"`; clear `selectedEvidenceIds` when confidence is low.
- Filter `selectedEvidenceIds` through a `strong_evidence_ids` set (`graphs.py:1557-1566`) so synthesis can't encode weak/off-topic IDs into grounded answers.
- Thread `synthesisMode` and `evidenceQualityProfile` into `classify_answerability` (`dispatch.py:4064-4072` + `paper_chaser_mcp/guided_semantic.py`) so `weak_authoritative_only` / `off_topic` profiles downgrade synthesis answers.

**Tests (new, `tests/test_ask_result_set_llm_mode_gate.py`, `tests/test_dispatch.py`, `tests/test_provider_base.py`):**
- `test_ask_result_set_does_not_promote_grounded_on_deterministic_boilerplate`
- `test_follow_up_research_session_introspection_does_not_emit_answered_on_weak_pool`
- `test_deterministic_bundle_default_qa_gates_answerability_on_evidence_count`

---

## P0-2 · Split URL-found from body-text-available

**Symptom:** `fullTextObserved` / `accessStatus="full_text_verified"` are set purely from URL discovery or the presence of a `markdown`/`contentSource` field; nothing actually fetched/embedded body text for QA. Downstream `grounded` answerability gates on this lie.

**Root causes:**
- `paper_chaser_mcp/agentic/graphs.py:4051-4073` — sets `accessStatus="full_text_verified"` and `fullTextRetrieved=True` when `markdown or contentSource` exists, regardless of whether text was ingested.
- `paper_chaser_mcp/agentic/graphs.py:5097-5109` — `_paper_text()` (used by synthesis) only concatenates title/abstract/venue/year/authors; never reads full body.
- `paper_chaser_mcp/dispatch.py:1032-1043` — `_assign_verification_status` maps bare `full_text_url_found` to `verified_primary_source` for regulatory types.
- `paper_chaser_mcp/models/common.py:150-158, 498-509` — `AccessStatus` enum conflates URL/body-text semantics in a single field.

**Fix:**
- Introduce three distinct signals on `Paper` / `StructuredSourceRecord`:
  - `fullTextUrlFound` — URL discovered only.
  - `bodyTextEmbedded` — body text fetched and indexed (true for GovInfo inline markdown).
  - `qaReadableText` — body text fetched for *this* synthesis call.
- Update `AccessStatus` to add `url_verified`, `body_text_embedded`, `qa_readable_text`; deprecate (don't yet remove) `full_text_verified`.
- Flip `graphs.py:4051-4073` to set `accessStatus=body_text_embedded` and `bodyTextEmbedded=True` only when inline markdown/content is present; URL-only stays `url_verified`.
- In `graphs.py:1500-1525, 1765-1796`, require at least one selected evidence item with `qa_readable_text=True` before `answerability="grounded"`; otherwise downgrade to `limited` and flip `answer_status` to `insufficient_evidence`.
- In `dispatch.py:1032-1043`, split `verified_primary_source` (needs `bodyTextEmbedded`) from `url_verified` (URL only).

**Tests (`tests/test_schema_invariants.py`, `tests/test_ask_result_set_llm_mode_gate.py`, `tests/test_dispatch.py`):**
- `test_full_text_url_found_does_not_imply_body_text_embedded`
- `test_govinfo_inline_markdown_sets_body_text_embedded`
- `test_ask_result_set_not_grounded_without_qa_readable_text`
- `test_ask_result_set_grounded_with_qa_readable_text`
- Update `tests/test_trust_ux_deepen.py:31-51` and `tests/test_provider_runtime_fixes.py:70-77` to assert the new split instead of `full_text_verified`.

---

## P0-3 · Resolve schema self-contradictions

**Symptoms observed in live testing:**
1. `coverageSummary.providersZeroResults` lists providers that did return evidence.
2. `trustSummary.verifiedPrimarySourceCount` disagrees with per-source `isPrimarySource` flags.
3. Duplicate `sourceAlias` (e.g., two `source-1`) across merged buckets.
4. Same FR document labeled `on_topic` by `research` and `weak_match` by `inspect_source`.
5. `authoritativeButWeak` bucket semantics are ambiguous and under-documented.

**Root causes and fixes:**

| # | File:line (current) | Fix |
| --- | --- | --- |
| 1 | `paper_chaser_mcp/agentic/graphs.py:3042-3046` — `zero_results` never reconciled with later `succeeded`/`structured_sources`. | Recompute `zero_results = [p for p in attempted if p not in succeeded and p not in failed]` immediately before building `CoverageSummary`. |
| 2 | `paper_chaser_mcp/dispatch.py:2795-2797` — counts `verificationStatus` but is named `verifiedPrimarySourceCount`. | Count sources where `isPrimarySource=True` AND `verificationStatus ∈ {verified_primary_source, verified_metadata}`. Add a separate `fullTextVerifiedPrimarySourceCount` for the full-text-only signal. |
| 3 | `paper_chaser_mcp/dispatch.py:1130, 1198` — per-call indices produce `source-1` in both papers and FR doc builders; `paper_chaser_mcp/agentic/workspace.py:528-558` only assigns missing aliases. | Stop pre-assigning `sourceAlias` in the per-bucket helpers. Enhance `_attach_source_aliases` to dedup (`seen_aliases` set; reissue `src_{i}` on collisions) and write the canonical map back to `sessionSourceAliases`. |
| 4 | `paper_chaser_mcp/dispatch.py:1190-1206` — FR docs hardcode `topicalRelevance="on_topic"`; smart path in `graphs.py:955-983` uses `_paper_topical_relevance`. | Extract a canonical `compute_topical_relevance(query, source)` and call it from both `_guided_sources_from_fr_documents` and smart retrieval. FR docs should earn `on_topic` from facet/term matching, not by fiat. |
| 5 | `paper_chaser_mcp/dispatch.py:2772-2786, 2820-2839` — `authoritativeButWeak` semantics only inferable from code. | Keep current predicate (authoritative family AND `weak_match`/`off_topic`) but add an inline docstring + tool-spec description making clear this is a **missed-escalation** bucket, and emit a one-line prose note when populated. |

**Tests (`tests/test_schema_invariants.py`, `tests/test_dispatch.py`):**
- `test_zero_results_excludes_succeeded_providers_post_fallback`
- `test_primary_source_flag_reflected_in_verified_count`
- `test_source_alias_collision_detection`
- `test_source_alias_globally_unique_across_buckets`
- `test_topical_relevance_consistent_across_research_and_inspect_source`
- `test_fr_document_relevance_computed_not_assumed`
- `test_authoritative_but_weak_semantics_clear`

---

## P1-1 · Payload slimdown for follow-ups

**Symptom:** 7/16 guided calls in the UX run had to be spilled to temp files; follow-ups commonly land at 29-35KB. Legacy `unverifiedLeads`/`verifiedFindings`, full `StructuredSourceRecord`s, and verbose `coverageSummary`/`trustSummary`/`providerDiagnostics` are repeated on every turn.

**Root causes:**
- `paper_chaser_mcp/dispatch.py:1230-1267` — `_guided_findings_from_sources` and `_guided_unverified_leads_from_sources` always run.
- `paper_chaser_mcp/dispatch.py:3927-3975, 3978-4013` — compact mode only engages for `insufficient_evidence`; every other follow-up ships full payload.
- `paper_chaser_mcp/models/tools.py` (response models) — `StructuredSourceRecord` serializes ~60 fields including `classificationRationale`, `relevanceReason`, `classification_source`.
- `paper_chaser_mcp/agentic/models.py` response classes (AskResultSetResponse, SmartSearchResponse, LandscapeResponse) list both `sources`/`structuredSources` and legacy `verifiedFindings`/`unverifiedLeads`.

**Fix (ship sequentially, flip defaults after each step):**
1. Strip null/empty fields and default-omit `classificationRationale`/`relevanceReason`/`classification_source` from `StructuredSourceRecord` on follow-ups.
2. Default `sourcesSuppressed=true` on `follow_up_research` responses (except `research_mode=debug`). When suppressed, return only `selectedEvidenceIds`, `selectedLeadIds`, and a `sourceDeltaSummary` (new/removed counts); full source records remain retrievable via `inspect_source`.
3. Add explicit `response_mode: "compact" | "standard" | "debug"` on `follow_up_research` (and optionally `research`). Map:
   - `compact` → sourcesSuppressed=true, legacy fields off, minimal diagnostics (default for follow-ups).
   - `standard` → modern sources on, legacy fields off (default for `research`).
   - `debug` → everything, for `inspect_source`-style calls.
4. Gate `verifiedFindings` / `unverifiedLeads` / full `trustSummary` behind `includeLegacyFields=true` (default false).
5. In compact mode, collapse `CoverageSummary` to `providersAttempted` only and drop `FailureSummary` unless `answer_status == "error"`.

**Target paths:**
- `paper_chaser_mcp/dispatch.py:1230-1267, 2787, 3927-3975, 3978-4013, 4005-4013, 4182`
- `paper_chaser_mcp/agentic/models.py` (AskResultSetResponse, SmartSearchResponse, LandscapeResponse)
- `paper_chaser_mcp/models/tools.py` (tool schema — add `response_mode`, `includeLegacyFields`)

**Tests (`tests/test_payload_efficiency.py`, `tests/test_dispatch.py`):**
- `test_follow_up_compact_by_default`
- `test_legacy_fields_opt_in`
- `test_structured_sources_ids_only_in_follow_up`
- `test_diagnostic_fields_stripped_in_compact`
- Extend `test_follow_up_insufficient_evidence_payload_is_compact` and `test_research_abstention_payload_is_compact` to the new `response_mode` enum.

---

## P1-2 · `topRecommendation` for comparative / selection follow-ups

**Symptom:** Comparative questions ("which should I start with?", "most recent?", "most authoritative?") get a generic evidence dump; no explicit "pick this one and why" field. Classification already detects comparative intent; the response surface does not use it.

**Root causes:**
- `paper_chaser_mcp/dispatch.py:4560-4616` — intent classification routes comparative markers but selection-ranking markers (e.g. "best starting point", "beginner-friendly", "most accessible") aren't covered.
- `paper_chaser_mcp/dispatch.py:4628-4750` + `paper_chaser_mcp/agentic/answer_modes.py:36-237` — `SYNTHESIS_MODES` / `ANSWER_MODES` lack a `selection` mode; synthesis falls through to LLM with no ranking axis.
- `paper_chaser_mcp/agentic/graphs.py:1433-1934` — no comparative-axis scoring; `selectedEvidenceIds` populated without justification.
- `paper_chaser_mcp/agentic/models.py:567-609` (AskResultSetResponse) has no `topRecommendation` field.

**Fix:**
- Extend `ANSWER_MODES` / `SYNTHESIS_MODES` (`answer_modes.py:36, 49`) with `"selection"`.
- Extend `_classify_question_mode_keyword` (`answer_modes.py:206-237`) with markers: `which is best`, `best starting point`, `most suitable`, `beginner-friendly`, `most recent`, `most authoritative`.
- New module `paper_chaser_mcp/agentic/selection_scoring.py`:
  - `infer_comparative_axis(question) -> Literal["beginner_friendly","recency","authority","coverage","relevance"]`
  - `score_papers_for_comparative_axis(papers, question, axis, provider_bundle) -> dict[str, float]` with deterministic fallbacks (citation count for authority, year for recency, inverse-citation + survey-title for beginner, abstract/title keyword coverage for coverage).
- Branch in `ask_result_set` (`graphs.py:1745-1860`): when `resolved_question_mode in {"comparison","selection"}`, compute axis scores, pick top, build a `topRecommendation` payload with `sourceId`, `title`, `recommendationReason`, `comparativeAxis`, `axisScore`, and up to 2 `alternativeRecommendations` on other axes.
- Add `top_recommendation: dict | None = Field(default=None, alias="topRecommendation")` to `AskResultSetResponse` (`models.py:567`) and populate before return at `graphs.py:1884-1914`.

**Tests (new `tests/test_follow_up_comparative_selection.py` + extend `tests/test_follow_up_synthesis_integrity.py`):**
- Intent detection for beginner/recency/authority/coverage markers.
- `infer_comparative_axis` covers each axis + default to `relevance`.
- Deterministic axis scoring with/without LLM provider.
- `ask_result_set` populates `topRecommendation` for selection mode, suppresses for non-selection modes.
- `topRecommendation` is `None` when session evidence is too weak to score.

---

## P2-1 · Conceptual-query foundational / survey boost

**Symptom:** "What is RAG?" returns recent optimization papers instead of foundational surveys or the Lewis et al. 2020 canonical paper.

**Root causes:**
- `paper_chaser_mcp/agentic/planner.py:671-688, 770-805` — broad-concept detection flips `query_specificity=low` but `initial_retrieval_hypotheses` never adds a "survey" / "overview" variant.
- `paper_chaser_mcp/agentic/ranking.py:254-258, 302-310` — dampens but never inverts the recency/citation prior for low-specificity queries; no canonical boost for `old + highly-cited`.
- `paper_chaser_mcp/agentic/ranking.py:287-295` — no keyword boost for `survey|review|overview|introduction|tutorial|primer|guide` titles.
- `paper_chaser_mcp/agentic/ranking.py:339-350` — bridge bonus rewards variant diversity but not cross-hypothesis agreement on a single paper.

**Fix:**
- Add `_is_definitional_query(query)` in `planner.py` (after line 437) covering `what is|are|does`, `introduction to`, `overview of`, `explain`, `define`, `guide to`.
- When true, inject two extra variants in `initial_retrieval_hypotheses` (`planner.py:770-805`): `"{core_term} survey"` and `"{core_term} overview introduction"`.
- In `ranking.py:302-312`, add a `canonical_bonus` for `citation_count >= 100 and age >= 5` (linear to +0.10) and `citation_count >= 50 and age >= 8` (flat +0.08).
- In `ranking.py:287-295`, add `+0.08` (broad_query_mode) or `+0.06` (low specificity) when title contains `survey|review|overview|introduction|tutorial|primer|guide`.
- In `ranking.py:339-350`, add consensus bonus: `+0.05` for `variantSources` diversity ≥ 2, `+0.03` when variants ≥ 3.

**Tests (`tests/test_smart_tools.py`):**
- `test_classify_query_detects_definitional_what_is_pattern`
- `test_rerank_candidates_promotes_survey_papers_for_what_is_rag_query`
- `test_rerank_candidates_canonical_lewis_2020_rag_paper_ranks_high`
- `test_rerank_candidates_dampens_year_recency_bonus_for_low_specificity` (offset by canonical bonus)

---

## P2-2 · Citation repair for natural-language classics

**Symptom:** `resolve_reference("Watson JD, Crick FH. Molecular structure of nucleic acids. Nature 1953.")` returns `no_match`; DOI input works. This is a confluence failure (initials lost, sparse queries missing venue, year penalty too lenient, author weight too low).

**Root causes (`paper_chaser_mcp/citation_repair.py`):**
- `:1437-1460` `_extract_author_surnames` — initials like `JD`, `FH` don't cross-match full author names in candidate papers.
- `:1541-1559` `_sparse_search_queries` — journal/venue hint isn't prioritized; the `author + year` query at line 1554 lacks venue.
- `:1156-1159` year-mismatch penalty peaks around −0.20 for 2-5 year deltas; easily absorbed by a weak title match.
- `:1151` author overlap capped at +0.10 vs title similarity × 0.35.
- No famous-citation registry: iconic DOIs like `10.1038/171964b0` aren't short-circuited when only title/author/year is supplied.
- Note on repo memory: the recorded "matchConfidence bonus 0.25" claim is off; the actual upstream confidence bonus maxes at +0.15 (`:1166-1175`), and year penalty can reach −0.26 when delta > 5, not 0.06.

**Fix (priority order):**
1. Add a small hand-curated `FAMOUS_CITATIONS: dict[tuple[str, ...], str]` registry checked before sparse-metadata fan-out (`resolve_citation` in `citation_repair.py:362-505`, around line 457). Short-circuits iconic papers through the identifier path (instant 0.95+ score).
2. In `_sparse_search_queries` (`:1541-1559`), when `parsed.author_surnames and parsed.year and parsed.venue_hints`, insert `"{authors} {year} {venue}"` at query position 2, and pre-pend `"{venue}"` as the first query.
3. In `_author_overlap` (`:1610-1618`), match short (`<=2` char) parsed surnames against initials extracted from candidate author names (`"James D Watson"` → initials `{J,D,W}` matches `JD`).
4. In `_rank_candidate` (`:1156-1159`), raise year-mismatch penalty: `delta 2-3 → -delta * 0.06`, `delta 4+ → -0.20 - (delta - 4) * 0.02`, additional `-0.08` for `delta > 5`.
5. In `_rank_candidate` (`:1151`), bump author-overlap scaling from `min(overlap,2) * 0.05` to `min(overlap,3) * 0.08`.
6. Loosen title-candidate skip (`:1496-1504`): allow ≤4-word title fragments when `venue_hints` present or `year` matches.

**Tests (`tests/test_citation_repair.py`):**
- `test_parse_citation_extracts_author_initials_watson_crick`
- `test_sparse_search_queries_prioritizes_journal_hint`
- `test_rank_candidate_penalizes_year_mismatch_watson_crick_variant`
- `test_rank_candidate_boosts_multi_author_match`
- `test_resolve_reference_watson_crick_status_not_no_match`
- `test_resolve_citation_famous_paper_registry_watson_crick` (once registry lands)

---

## Suggested implementation order

1. **P0-3 item 3 (sourceAlias dedup)** and **P0-3 item 1 (providersZeroResults)** — pure computation fixes, smallest blast radius.
2. **P0-1 (answer-status semantics)** — contract-shaping; unlocks trust for everything downstream.
3. **P0-2 (full-text vs body-text)** — requires schema additions but is gating P0-1's `grounded` story.
4. **P0-3 items 2, 4, 5** — semantic alignment, smaller now that the schema is settled.
5. **P1-1 (payload slimdown)** — ride alongside P0-1/P0-2 since both touch response assembly.
6. **P1-2 (topRecommendation)** — additive field, self-contained.
7. **P2-1 (conceptual ranking)** and **P2-2 (citation repair)** — recall/ranking quality improvements.

## Validation gate (run before any PR)

```powershell
pip install -e ".[all]"
python -m pip check
pre-commit run --all-files
python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=85
python -m mypy --config-file pyproject.toml
python -m ruff check .
python -m bandit -c pyproject.toml -r paper_chaser_mcp
python -m build
python -m pip_audit . --progress-spinner off
```

Each remediation area should also update the durable surfaces the custom
instructions call out (`README.md`, `paper_chaser_mcp/server.py`
`SERVER_INSTRUCTIONS` / `AGENT_WORKFLOW_GUIDE` / `plan_paper_chaser_search`,
`docs/golden-paths.md`, `docs/guided-reset-migration-note.md`,
`docs/agent-handoff.md`) when the user-visible contract changes.
