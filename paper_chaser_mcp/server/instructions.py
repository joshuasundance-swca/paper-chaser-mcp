"""Static instruction strings surfaced by the Paper Chaser MCP server."""

SERVER_INSTRUCTIONS = """
Decision tree for tool selection:

1. DEFAULT GUIDED RESEARCH → research
     (default low-context entry point for discovery, literature review,
     citation repair, known-item recovery, and regulatory or species-history
     questions; guided research uses a server-owned quality-first policy and now
     returns executionProvenance plus evidence-first sections such as
     evidence/leads/routingSummary/coverageSummary/evidenceGaps, with
     verifiedFindings/sources/unverifiedLeads kept as compatibility views.
     Underspecified citation-like fragments can stop early with
     needs_disambiguation + clarification instead of speculative retrieval)
2. GUIDED FOLLOW-UP → follow_up_research
    (one grounded follow-up over a saved searchSessionId; returns answerStatus,
    nextActions, executionProvenance, and sessionResolution when session reuse is
    ambiguous)
3. GUIDED REFERENCE NORMALIZATION → resolve_reference
     (normalize DOI/arXiv/URL/citation/reference inputs before broader discovery;
     do not treat bestMatch as citation-ready unless status=resolved)
4. GUIDED SOURCE AUDIT → inspect_source
    (per-source provenance, trust, and direct-read recommendations; ambiguity now
    returns structured sessionResolution/sourceResolution instead of raw errors)
5. RUNTIME / PROFILE SANITY CHECK → get_runtime_status
     (surfaces effective profile, smart-provider state, guidedPolicy, guided
     latency defaults, and runtime warnings; disabled vs suppressed/degraded/
     quota-limited providers are split intentionally)
6. EXPERT CONCEPT-LEVEL DISCOVERY / REVIEW → search_papers_smart
    (returns searchSessionId, strategyMetadata, resultStatus, answerability,
    routingSummary, evidence, leads, evidenceGaps, structuredSources,
    coverageSummary, plus resourceUris and agentHints; legacy trust fields stay
    available as compatibility views. Use latencyProfile=deep for highest-quality
    expert work, balanced when lower latency matters, and fast only for smoke
    tests or debugging)
7. QUICK RAW DISCOVERY → search_papers
    (brokered, single page, returns brokerMetadata plus agentHints/resourceUris)
8. EXHAUSTIVE / MULTI-PAGE → search_papers_bulk
   (cursor-paginated, up to 1 000 returned/call; read retrievalNote because
   default bulk ordering is not relevance-ranked)
9. EXPERT CITATION REPAIR / ALMOST-RIGHT REFERENCES → resolve_citation
10. EXPERT KNOWN ITEM (messy title) → search_papers_match
   (takes only a query string — the title text — not separate author/year/venue
   fields; use resolve_citation for multi-field bibliographic references)
11. EXPERT KNOWN ITEM (DOI / arXiv / URL) → get_paper_details
12. PAPER ENRICHMENT / OA CHECK → get_paper_metadata_crossref,
   get_paper_open_access_unpaywall, or enrich_paper after you already have a
   concrete paper, DOI, or DOI-bearing identifier
13. EXPERT GROUNDED FOLLOW-UP → ask_result_set or map_research_landscape using searchSessionId
14. CITATION EXPANSION → get_paper_citations (cited-by) or get_paper_references (refs)
15. AUTHOR PIVOT → search_authors → get_author_info → get_author_papers
16. PHRASE / QUOTE RECOVERY → search_snippets (last resort)
17. OPENALEX-SPECIFIC PATHS → use the *_openalex tools when you explicitly need
   OpenAlex-native DOI/ID lookup, OpenAlex cursor paging, author pivots, or
   source/institution/topic pivots via search_entities_openalex and
   search_papers_openalex_by_entity
18. SCHOLARAPI FULL-TEXT PATHS → use search_papers_scholarapi,
    list_papers_scholarapi, get_paper_text_scholarapi,
    get_paper_texts_scholarapi, or get_paper_pdf_scholarapi when the workflow
    explicitly needs ScholarAPI-ranked discovery, indexed-at monitoring,
    accessible full text, or binary PDF retrieval
19. SERPAPI RECOVERY PATHS → use search_papers_serpapi_cited_by,
   search_papers_serpapi_versions, get_author_profile_serpapi,
   get_author_articles_serpapi, or get_serpapi_account_status only when
   PAPER_CHASER_ENABLE_SERPAPI=true and the workflow justifies paid recall recovery
20. ECOS SPECIES DOSSIERS → search_species_ecos → get_species_profile_ecos →
   list_species_documents_ecos → get_document_text_ecos for species pages,
   regulatory documents, and recovery PDFs from the U.S. Fish and Wildlife
   Service ECOS system
21. REGULATORY PRIMARY SOURCES → search_federal_register for discovery,
    get_federal_register_document for one notice or rule, and get_cfr_text for
    authoritative CFR part/section text. NOTE: Biological opinions, Section 7
    consultation records, and incidental take permits live in ECOS, not the
    Federal Register — use the ECOS species dossier chain for those.
22. PROVIDER HEALTH / DEBUGGING → get_provider_diagnostics

After search_papers: read brokerMetadata.nextStepHint for the recommended next move.
After search_papers_smart: reuse searchSessionId for ask_result_set,
map_research_landscape, or expand_research_graph, and inspect resultStatus,
answerability, routingSummary, evidence, leads, structuredSources,
coverageSummary, failureSummary,
acceptedExpansions, rejectedExpansions, speculativeExpansions, providersUsed,
driftWarnings, latencyProfile, providerBudgetApplied, and providerOutcomes. Set
includeEnrichment=true only when you want Crossref, Unpaywall, and OpenAlex metadata on the
final smart-ranked hits; enrichment is post-ranking only and never changes
retrieval or provider ordering. When ScholarAPI is enabled, smart retrieval may
also include it explicitly, and providerBudget.maxScholarApiCalls can cap that
paid path.
When the query is clearly regulatory or species-history oriented, search_papers_smart can also route into
ECOS/Federal Register/CFR retrieval first and return a regulatoryTimeline instead of paper-centric ranking.
Primary read tools now also return agentHints, clarification, resourceUris, and,
when they produce reusable result sets, searchSessionId.
For known-item flows, includeEnrichment=true on search_papers_match,
get_paper_details, or resolve_citation adds Crossref, Unpaywall, and OpenAlex metadata only
after the base paper resolution succeeds.
For Semantic Scholar expansion tools, prefer paper.recommendedExpansionId when
present. If paper.expansionIdStatus is not_portable, do not retry with brokered
paperId/sourceId/canonicalId values; resolve the paper through DOI or a
Semantic Scholar-native lookup first.
If search_papers_match returns no match, or if the user has a broken
bibliography line, partial reference, or almost-right citation, prefer
resolve_citation before guessing. A no-match can still mean the item is a
dissertation, software release, report, or other output outside the indexed
paper surface. If the citation is clearly regulatory (for example a Federal
Register or CFR reference), switch to search_federal_register,
get_federal_register_document, or get_cfr_text instead of forcing a paper
lookup.
For common-name author lookup, add affiliation, coauthor, venue, or topic clues
before expanding into get_author_info/get_author_papers.
To steer the broker: use preferredProvider (try-first) or providerOrder (full override).
Provider names: semantic_scholar, arxiv, core, scholarapi, serpapi / serpapi_google_scholar.
Provider-specific search inputs: search_papers_core, search_papers_serpapi, and
search_papers_arxiv only accept query/limit/year; search_papers_semantic_scholar
supports the wider Semantic Scholar filter set; search_papers_scholarapi and
list_papers_scholarapi expose ScholarAPI-specific cursor/date/full-text filters.
OpenAlex is available through explicit *_openalex tools instead of the broker because
its citation, author, and pagination semantics differ from Semantic Scholar.
Continuation rule: search_papers_bulk is the closest continuation path only for
Semantic Scholar-style retrieval; from CORE, arXiv, ScholarAPI, or SerpApi results it is a
Semantic Scholar pivot rather than another page from the same provider.
Even on Semantic Scholar paths, default bulk ordering is NOT relevance-ranked;
it is not 'page 2' of search_papers. Read retrievalNote in each bulk response,
or pass sort='citationCount:desc' for citation-ranked bulk traversal.
For small targeted pages, prefer search_papers or search_papers_semantic_scholar;
Semantic Scholar's bulk endpoint may ignore small limits internally.
For agentic UX review loops, run a small smoke baseline first, then widen into
OpenAlex, snippet recovery, paper-to-author pivots, or a feature-specific probe
only when the workflow goal calls for broader coverage. Capture any defects as
reproduction-ready issues that can guide code changes and documentation updates.

Pagination rule: treat pagination.nextCursor as opaque — pass it back exactly as
returned, do not derive, edit, or fabricate it, and do not reuse it across a
different tool or query flow.
For repo-local eval bootstrap or workflow QA work, prefer the checked-in
scripts (`scripts/generate_eval_topics.py`, `scripts/run_eval_autopilot.py`,
and `scripts/run_eval_workflow.py`) so runs stay reproducible and emit the
expected bundle artifacts.
""".strip()

GUIDED_SERVER_INSTRUCTIONS = """
Default guided workflow:

1. RESEARCH -> research
    Use this for topic discovery, literature review, known-item recovery, citation repair,
    and regulatory or species-history requests when you want one trust-graded answer.
     The server applies a server-owned quality-first policy for this guided path.
    Vague citation/reference fragments can stop here with needs_disambiguation +
    clarification instead of speculative retrieval.
2. FOLLOW UP -> follow_up_research
   Reuse searchSessionId from research to ask one grounded question. The tool abstains
   when the saved evidence is too weak or off-topic.
   Responses are compact by default (sources collapsed to selectedEvidenceIds/
  selectedLeadIds; diagnostics and legacy verifiedFindings/unverifiedLeads omitted).
  Compact responses can also expose suppressedSourceSummaries with minimal
  {sourceId, title, topicalRelevance, reason} records for excluded weak/off-topic items.
   Pass responseMode="standard" for full source records, responseMode="debug" for
   full diagnostics, or includeLegacyFields=true to restore legacy compatibility views.
   Grounded answers only land when synthesis is backed by at least one on-topic,
   verified source with qa-readable text and a non-deterministic provider; otherwise
   the tool returns answerStatus=insufficient_evidence or abstained.
   Comparative / selection follow-ups (e.g. "which should I start with?", "most
   recent?") include a topRecommendation payload with sourceId, recommendationReason,
   and comparativeAxis when the saved evidence can be scored.
3. RESOLVE ONE REFERENCE -> resolve_reference
    Use this for citations, DOI strings, arXiv IDs, URLs, title fragments, and regulatory references.
    Treat bestMatch as citation-ready only when status=resolved.
4. INSPECT ONE SOURCE -> inspect_source
   Pass searchSessionId plus evidenceId, sourceAlias, or sourceId from research to inspect provenance, trust state,
   and direct-read next steps.

Use get_runtime_status when behavior looks different across environments and you need the active runtime truth.
Read disabledProviderSet separately from suppressedProviderSet/degradedProviderSet/quotaLimitedProviderSet.
The guided surface is intentionally opinionated: it prefers trust-graded evidence, explicit abstention, and direct next
actions over raw provider control.
""".strip()

AGENT_WORKFLOW_GUIDE = """
# Paper Chaser agent workflow guide

## Default guided path

- Start with `research` for topic discovery, literature review, known-item recovery,
  citation repair, and regulatory or species-history requests.
- Treat guided `research` as server-managed quality-first behavior rather than a
    place to choose fast/balanced/deep execution modes.
- If guided `research` returns `needs_disambiguation` with clarification for an
  underspecified citation-like fragment, tighten the anchor or switch to
  `resolve_reference` instead of forcing retrieval.
- Save the returned `searchSessionId`. It is the anchor for `follow_up_research`
  and `inspect_source`.
- Use `follow_up_research` for one grounded question over the saved evidence.
  It is supposed to abstain when the evidence is weak, off-topic, or incomplete.
  Responses are compact by default. Request `responseMode="standard"` when you
  need full source records, `responseMode="debug"` for full diagnostics, or
  `includeLegacyFields=true` to restore `verifiedFindings`/`unverifiedLeads`.
  Compact abstention paths can also expose `suppressedSourceSummaries` with
  minimal `{sourceId, title, topicalRelevance, reason}` records for excluded
  weak/off-topic items.
  Grounded answers require an on-topic, verified source with qa-readable text
  and a non-deterministic synthesis provider; otherwise expect abstention.
  Comparative / selection asks ("which should I start with?", "most recent?",
  "most authoritative?") surface a `topRecommendation` with the chosen source,
  a one-line reason, and the inferred `comparativeAxis`.
- Use `resolve_reference` when the user already has a citation, DOI, arXiv ID,
  URL, title fragment, or regulatory reference and wants the safest next anchor.
  Treat `bestMatch` as final only when `status=resolved`; otherwise use
  `multiple_candidates` / `needs_disambiguation` as a prompt to disambiguate.
- Use `inspect_source` with `searchSessionId` plus `evidenceId`, `sourceAlias`,
  or `sourceId` to inspect
  provenance, trust state, access status, and direct-read next steps.
  Access now distinguishes `fullTextUrlFound` (URL discovered), `bodyTextEmbedded`
  (body text indexed into the session), and `qaReadableText` (body text actually
  used for the current synthesis call) so agents can tell URL discovery apart
  from true full-text reads.
- Use `get_runtime_status` when behavior differs across environments and you need
  the active runtime truth without digging through low-level diagnostics. Read
  `disabledProviderSet`, `suppressedProviderSet`, `degradedProviderSet`, and
  `quotaLimitedProviderSet` as distinct top-level summaries of the per-provider
  rows rather than one collapsed availability bucket; `disabledProviderSet`
  means configured-off, while the other three describe runtime-only state.

## Guided output contract

- `research` returns `resultStatus`, `answerability`, `summary`, `routingSummary`,
  `coverageSummary`, `evidence`, `leads`, `evidenceGaps`, `timeline`,
  `nextActions`, and `clarification`.
- `resultStatus` is one of `succeeded`, `partial`, `needs_disambiguation`,
  `abstained`, or `failed`.
- Treat `evidence` as the canonical grounded support set for answers and
  inspection. Treat `leads` as auditable but not-yet-grounded context.
- Legacy `verifiedFindings`, `sources`, and `unverifiedLeads` may still be
  present for compatibility, but they should be derived views rather than the
  primary trust contract.
- If the tool abstains or asks for clarification, do not smooth that over with
  your own synthesis. Ask a narrower question or inspect the returned sources.

## Expert/operator-only fallback

- Use the expert surface only when you truly need raw provider control,
  pagination semantics, or provider-native payloads.
- For expert smart tools, deep is the default quality-first mode; balanced is a
    lower-latency alternative, and fast is reserved for smoke tests.
- Expert discovery tools include `search_papers`, `search_papers_bulk`,
  `search_papers_smart`, `map_research_landscape`, and `expand_research_graph`.
- Expert primary-source tools include `search_federal_register`,
  `get_federal_register_document`, `get_cfr_text`, `search_species_ecos`,
  `get_species_profile_ecos`, and `list_species_documents_ecos`.
- Expert runtime debugging lives under `get_provider_diagnostics`.

## Repo-local eval bootstrap

- When the task is repo-local eval generation or workflow QA rather than an
    end-user research answer, prefer the checked-in scripts over improvised tool
    loops: `scripts/generate_eval_topics.py`, `scripts/run_eval_autopilot.py`,
    and `scripts/run_eval_workflow.py`.
- Use the checked-in autopilot profiles for narrow one-seed experiments instead
    of editing thresholds ad hoc. The exploratory profiles are meant to make
    small-run behavior explicit and reproducible.
- Prefer single-seed diversification when you need broader one-seed coverage;
    it asks the planner for review, regulatory, and methods-oriented variants
    rather than relying only on looser workflow gating.

## Safety habits

- Prefer guided tools unless a concrete expert-only need is present.
- Expect guided summaries to lead with a short recommendation first, then use
    evidence, leads, and provenance for the audit trail.
- Reuse `searchSessionId` instead of rephrasing the same question into multiple
  raw tools.
- Treat `pagination.nextCursor` as opaque whenever you are on an expert paginated
  path.
- Capture defects with the exact tool call, what the user expected, what the
  tool returned, and whether the fix belongs in code, docs, or both.
""".strip()

__all__ = [
    "SERVER_INSTRUCTIONS",
    "GUIDED_SERVER_INSTRUCTIONS",
    "AGENT_WORKFLOW_GUIDE",
]
