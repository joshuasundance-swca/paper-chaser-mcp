# Scholar Search Golden Paths

This document captures the primary planning assumptions for the MCP surface so
agents can choose the next tool with low context and high reliability.

These same discovery, pagination, lookup, citation, author-pivot, and smart
result-set paths are exercised by the agentic smoke test in
`.github/workflows/test-scholar-search.md`.

## Primary Personas

- **Quick discovery user**: wants a strong first page fast to decide whether to
  dig deeper.
- **Concept-level researcher**: starts from a topic or question rather than a
  known paper and wants a reusable result set with grounded follow-up paths.
- **Researcher doing exhaustive retrieval**: wants many papers across pages to
  build a reading list or dataset.
- **Citation chaser**: starts from one paper and expands outward through cited-by
  and reference relationships.
- **Known-item lookup user**: has a messy title, DOI, arXiv ID, URL, or partial
  clue and wants the right paper quickly.
- **Author-centric user**: wants to understand an author's work, recent output,
  or collaborators.

## Golden Paths

### 1. Smart concept discovery and grounded follow-up

1. Start with `search_papers_smart` when the task is concept-level discovery,
   literature review, or a grounded follow-up workflow.
2. Inspect `strategyMetadata`, especially `acceptedExpansions`,
   `rejectedExpansions`, `speculativeExpansions`, and `providersUsed`.
3. Save the returned `searchSessionId`.
4. Use `ask_result_set` for grounded QA, `map_research_landscape` for themes,
   and `expand_research_graph` for compact citation/reference/author expansion.
5. If `mode="known_item"` does not confirm one exact anchor, the smart workflow
  now returns a broader candidate set with warnings instead of a dead-end
  error; verify title, year, and venue before treating that fallback set as
  canonical.

**Example request**: "Map the main research themes in retrieval-augmented generation for coding agents"

**Tool sequence**:
```text
search_papers_smart(query="retrieval-augmented generation for coding agents", limit=10)
→ inspect strategyMetadata, agentHints, resourceUris, searchSessionId
→ ask_result_set(searchSessionId="...", question="What does this result set say about evaluation tradeoffs?")
→ map_research_landscape(searchSessionId="...", maxThemes=4)
→ expand_research_graph(seedSearchSessionId="...", direction="citations")
```

**Success signals**

- The first smart result set is useful even when the starting query is concept-level.
- `strategyMetadata` explains what the server tried instead of acting like a black box.
- `searchSessionId` is enough to continue without rerunning the whole discovery step.
- Grounded follow-up answers cite papers from the saved result set instead of guessing.
- Known-item smart search degrades into a reviewable candidate set rather than a hard dead end when exact recovery is weak.

### 2. Quick literature discovery

1. Start with `search_papers`.
2. Read `brokerMetadata.nextStepHint` for the recommended next move.
3. Inspect `brokerMetadata.providerUsed` and `brokerMetadata.attemptedProviders`.
4. Decide whether to broaden, narrow, paginate with `search_papers_bulk`, or
   pivot into citations/authors.

**Example request**: "Find recent papers on large language model alignment"

**Tool sequence**:
```
search_papers(query="large language model alignment", limit=10)
→ read brokerMetadata.nextStepHint
→ if exhaustive collection needed: search_papers_bulk(query="...", limit=100)
  NOTE: search_papers_bulk uses exhaustive corpus traversal (NOT relevance-ranked).
  It is not 'page 2' of search_papers — results may appear in a different order.
  For citation-ranked bulk retrieval pass sort='citationCount:desc'.
→ for interesting paper X: get_paper_citations(paper_id=X.paperId)
```

**Success signals**

- `brokerMetadata.nextStepHint` is readable and tells the agent what to do next.
- The first response clearly explains where the results came from.
- The agent can explain whether to broaden, narrow, paginate, or pivot.
- `brokerMetadata.resultQuality` is `"strong"` for relevant Semantic Scholar
  results, `"low_relevance"` when distinctive query tokens are absent from all
  results (treat with caution), `"lexical"` for CORE/arXiv keyword matches, or
  `"unknown"` for SerpApi / no-result paths.

### 3. Exhaustive retrieval

1. Start with `search_papers_bulk` when the request asks for all results, the
   first N results across pages, or a dataset-like collection.
2. Note that the **default ordering is NOT relevance-ranked** — it is exhaustive
   corpus traversal with an internal ordering. Read `retrievalNote` in the
   response for the active ordering contract. For citation-ranked traversal pass
   `sort='citationCount:desc'`.
3. Treat `pagination.nextCursor` as opaque and pass it back as `cursor`
   unchanged.
4. Continue until `pagination.hasMore` is false.
5. For small targeted pages, prefer `search_papers` or
   `search_papers_semantic_scholar`; the upstream bulk endpoint may ignore small
   `limit` values internally, and this server only truncates the returned data.

**Example request**: "Get me all papers on RLHF published since 2020"

**Tool sequence**:
```
search_papers_bulk(query="reinforcement learning from human feedback", year="2020-", limit=100)
→ while pagination.hasMore:
    search_papers_bulk(query="...", year="2020-", cursor=pagination.nextCursor, limit=100)
```

**Success signals**

- Cursor handling stays explicit and safe.
- The agent does not misuse single-page tools for bulk collection.
- The agent does not misuse bulk search for small targeted pages.
- Pagination state is explained in user-facing language.

### 4. Citation chasing

1. Resolve or confirm the starting paper with `search_papers_match` or
   `get_paper_details` if needed.
2. Use `get_paper_citations` for cited-by expansion.
3. Use `get_paper_references` for backward references.
4. Use `get_paper_authors` if the next step is an author or collaborator pivot.

If the starting paper came from brokered non-Semantic-Scholar results, prefer
`paper.recommendedExpansionId` when it is present. If
`paper.expansionIdStatus` is `not_portable`, do not reuse brokered `paperId`,
`sourceId`, or `canonicalId` directly in Semantic Scholar expansion tools;
resolve the paper through DOI or a Semantic Scholar-native lookup first.

**Example request**: "What papers cite 'Attention Is All You Need'?"

**Tool sequence**:
```
search_papers_match(query="Attention Is All You Need")
→ get_paper_citations(paper_id=result.paperId, limit=100)
→ while pagination.hasMore:
    get_paper_citations(paper_id=result.paperId, cursor=pagination.nextCursor)
```

**Success signals**

- The agent explains the direction clearly: cited by vs references.
- Returned metadata is rich enough to rank or filter the expansion set.
- The workflow naturally chains from a discovered or known paper.

### 5. Known-item lookup

1. Use `resolve_citation` for broken bibliography lines, incomplete references,
   quote fragments, and almost-right citations.
2. Use `search_papers_match` for messy or partial titles.
3. Use `get_paper_details` for DOI, arXiv ID, URL, or canonical IDs.
4. Fall back to `search_papers` only when the user is really asking a topical
   question rather than an item lookup.
5. Treat a structured no-match from `search_papers_match` as a signal that the
   item may be punctuation-variant metadata, a dissertation, software release,
   report, or other output outside the indexed paper surface. Verify externally
   when needed.
6. When `resolve_citation` sees a report-style or non-paper-looking reference,
  prefer abstention plus alternatives over forcing a weak canonical paper
  match.
7. When `resolve_citation` sees a clearly regulatory reference such as a
  Federal Register or CFR citation, switch to `search_federal_register`,
  `get_federal_register_document`, or `get_cfr_text` instead of continuing to
  force scholarly-paper recovery.

**Example requests and tool choices**:
- "Find the paper called something like 'Scaling Laws for Neural Language Models'"
  → `search_papers_match(query="Scaling Laws for Neural Language Models")`
- "Resolve this partial citation: 'Rockstrom et al planetary boundaries 2009 Nature 461 472'"
  → `resolve_citation(citation="Rockstrom et al planetary boundaries 2009 Nature 461 472")`
- "Get details for arxiv:1706.03762"
  → `get_paper_details(paper_id="arXiv:1706.03762")`
- "Look up DOI 10.48550/arXiv.2005.14165"
  → `get_paper_details(paper_id="DOI:10.48550/arXiv.2005.14165")`

**Success signals**

- Disambiguation stays minimal.
- Citation repair returns alternatives and uncertainty instead of forcing a bad match.
- Regulatory citations are redirected into the Federal Register / CFR workflow instead of being treated as papers.
- Exact-title recovery can escalate beyond Semantic Scholar when the paper is
  real but missing from Semantic Scholar title-match results.
- DOI, URL, and identifier-based lookups feel low friction.
- Agents do not spend multiple topic-search turns on a known-item request.

### 6. Author-centric research

1. Start with `search_authors`.
2. Use `get_author_info` to confirm identity and profile metadata.
3. Use `get_author_papers` to expand into the author's work.
4. Use `get_paper_authors` when the user starts from a paper instead of a name.
5. When the starting paper came from brokered non-Semantic-Scholar results,
   prefer `paper.recommendedExpansionId` for the `paper_id` you pass into
   Semantic Scholar expansion tools. If `paper.expansionIdStatus` is
   `not_portable`, do not reuse brokered `paperId`/`sourceId`/`canonicalId`
   values directly; resolve the paper through DOI or a Semantic Scholar-native
   lookup first.
6. For common names, add affiliation, coauthor, venue, or topic clues in the
   initial `search_authors` query, then use profile metadata to confirm the
   right person before reading papers.

**Example request**: "What has Yoshua Bengio published recently?"

**Tool sequence**:
```
search_authors(query="Yoshua Bengio", limit=5)
→ get_author_info(author_id=top_match.authorId)
→ get_author_papers(author_id=top_match.authorId, publicationDateOrYear="2022:", limit=50)
```

**Success signals**

- Author search feels first-class instead of secondary.
- Agents can pivot cleanly between paper and author workflows.
- Agents use Semantic Scholar-compatible paper/author identifiers when pivoting
  out of brokered results.
- Recent work and collaborator exploration are easy next steps.

## Secondary Workflows

- **Regulatory follow-through from ECOS**: when an ECOS dossier or flattened
  ECOS document inventory exposes `frCitation` or a GovInfo FR link, use
  `get_federal_register_document` to anchor the Federal Register notice or
  rule, then use `get_cfr_text` for the affected CFR part or section. Use
  `search_federal_register` when the ECOS clue is still too broad and you need
  agency/date/type narrowing before retrieval. Direct document numbers remain
  the strongest anchor, but FR citation strings now retry broader Federal
  Register discovery before giving up.
- **Quote or snippet validation**: use `search_snippets` only when title or
  keyword search is weak, or when `resolve_citation` suggests a quote fragment
  is the strongest remaining clue.
- **Smart result-set refinement**: after `search_papers_smart`, prefer
  `ask_result_set`, `map_research_landscape`, and `expand_research_graph`
  before starting an entirely new search when the follow-up is still grounded
  in the same corpus.
- **Agent UX feedback loop**: the checked-in agentic workflow at
  `.github/workflows/test-scholar-search.md` now supports `smoke`,
  `comprehensive`, and `feature_probe` review modes. The workflow uses the
  GitHub Copilot CLI engine and can be pointed at GPT-5.4 or another model via
  `GH_AW_MODEL_AGENT_COPILOT`. It explicitly evaluates agent UX in every step:
  intuitiveness, unnecessary round trips, missing features, confusing
  contracts, and dead-end responses. Each run produces a structured UX friction
  summary before filing any issue. Start with the smoke baseline, then add
  deeper probes or a feature-specific focus prompt when you need broader UX
  feedback. Issues created by the verifier carry `agentic` and
  `needs-copilot` labels with stable body markers. The auto-assignment
  workflow listens to both issue events and completed verifier runs so
  workflow-created issues still get assigned to GitHub Copilot without
  duplicates.
  After 3 failed fix attempts the issue escalates to `needs-human`.
- **Explicit OpenAlex workflows**: use `search_papers_openalex` for one
  OpenAlex page, `search_papers_openalex_bulk` for OpenAlex cursor traversal,
  `get_paper_details_openalex` for OpenAlex W-id/DOI lookup, and the OpenAlex
  citation/author tools when the task explicitly needs OpenAlex-native DOI/ID,
  author, or paging semantics instead of the default broker.
- **Citation export**: use `get_paper_citation_formats` after SerpApi discovery
  when a result exposes `paper.scholarResultId` (a first-class field, `None`
  for non-SerpApi results). Pass `result_id=paper.scholarResultId`, not
  `paper.sourceId`.
- **Budget-aware searching**: steer `search_papers` with `preferredProvider` or
  `providerOrder` before falling back to provider-specific tools.

## Recent UX Fixes

These rough edges from the prior UX review are now fixed and should be treated
as part of the intended agent contract.

### 1. `search_papers_match` now resolves to one paper payload

- The Semantic Scholar match path now normalizes wrapped responses to a single
  paper-shaped object.
- Agents should no longer need to defensively inspect `data[0]` for the best
  match.
- Exact-match 400/404 misses now fall back to fuzzy Semantic Scholar title
  search, and exhausted match attempts return a structured no-match payload
  instead of surfacing a raw provider 404.

### 2. Known-item recovery now prefers abstention and cross-provider exact-title confirmation

- `resolve_citation` now prefers abstention plus alternatives when report-like,
  non-paper-looking, or otherwise weakly grounded inputs do not justify a
  canonical paper match.
- `search_papers_match` now tries strict OpenAlex exact-title recovery, then
  strict Crossref exact-title recovery, before giving up on a real paper that
  Semantic Scholar did not recover by title.
- Agents should treat `matchProvider` and `matchStrategy` as useful provenance
  when exact-title recovery succeeds outside the Semantic Scholar path.

### 3. CORE fallback follows redirects cleanly

- When the broker reaches CORE, predictable redirects are followed instead of
  recording an avoidable failed provider attempt.
- Treat new redirect-driven CORE failures in `brokerMetadata.attemptedProviders`
  as a regression if they reappear.

### 4. `nextStepHint` now distinguishes continuation from pivots

- Semantic Scholar results describe `search_papers_bulk` as the closest
  continuation path only when that preserves the research intent reasonably
  well.
- CORE, arXiv, and SerpApi results now describe `search_papers_bulk` as a
  Semantic Scholar pivot rather than another page from the same provider.
- Venue-filtered Semantic Scholar searches explicitly warn that bulk retrieval
  broadens the query semantics.

### 5. Provider-specific schemas now match provider capability

- `search_papers_core`, `search_papers_serpapi`, and `search_papers_arxiv`
   expose only `query`, `limit`, and `year`.
- `search_papers_openalex` also keeps its single-page contract explicit with
  only `query`, `limit`, and `year`, while `search_papers_openalex_bulk`
  exposes OpenAlex cursor pagination as a separate tool.
- `search_papers_semantic_scholar` continues to expose the wider Semantic
   Scholar-compatible filter set.

### 6. Author and paper pivots now call out Semantic Scholar ID boundaries

- `search_authors` normalizes plain-text exact-name punctuation before calling
  Semantic Scholar.
- Common-name author workflows now explicitly recommend affiliation, coauthor,
  venue, and topic clues before profile confirmation.
- Author-profile tools now describe the supported author fields explicitly.
- Semantic Scholar paper-expansion tools now tell agents to prefer
  `paper.recommendedExpansionId` when present, and to treat
  `paper.expansionIdStatus='not_portable'` as a signal that brokered
  `paperId`/`sourceId`/`canonicalId` values still need a DOI or
  Semantic-Scholar-native lookup before expansion.

### 7. Snippet-search provider failures now degrade cleanly

- `search_snippets` now returns an empty degraded payload with retry guidance
  when Semantic Scholar rejects a phrase query or returns a transient 5xx.
- Agents should prefer `search_papers_match` or `search_papers` before falling
  back to snippet search for quote recovery.

### 8. Nonsense / low-relevance queries now surface an explicit weak-match signal

- `search_papers` with Semantic Scholar now checks whether any distinctive
  query tokens (length ≥ 6, not a common academic term) appear in the returned
  result titles and abstracts.
- When a distinctive token is absent from every result, `brokerMetadata.resultQuality`
  is set to `"low_relevance"` instead of `"strong"` and `nextStepHint` includes
  an explicit warning that the results are likely a weak or irrelevant match.
- Agents must treat `resultQuality="low_relevance"` as a signal to stop the
  discovery workflow and instead rephrase the query, broaden it, or try a
  different provider via `providerOrder`.
- This catches common cases like gibberish tokens (e.g. `asdkfjhasdkjfh`) where
  Semantic Scholar returns papers that only match the generic words in the query.
- `resultQuality="strong"` still means Semantic Scholar's semantic ranking was
  used and all inspected distinctive tokens appear in at least one result.

### 9. Title-like broker misses now suppress obvious false positives

- For title-like known-item queries, brokered `search_papers` now suppresses
  obviously low-relevance Semantic Scholar hits instead of surfacing a bad
  paper-shaped record.
- Agents should treat this empty result plus `nextStepHint` as a prompt to
  switch to `search_papers_match`, refine the title, or adjust `providerOrder`
  rather than continuing with the brokered discovery set.

### 10. Smart follow-up outputs are stricter about grounding and relevance

- `ask_result_set(answerMode="comparison")` now prefers a grounded structured
  comparison over a flat list of titles and venues.
- `map_research_landscape` sanitizes weak theme labels before returning them.
- `expand_research_graph` keeps saved-session intent in frontier scoring and
  filters weak off-topic candidates before they become next-hop anchors.

### 8. `get_author_papers` open-ended `publicationDateOrYear` filter normalized

- The `publicationDateOrYear` parameter for `get_author_papers` uses `:` as the
  range separator, not `-` (which is the `year` parameter style).
- The documented open-ended form `"2022-"` is now automatically normalized to the
  correct API form `"2022:"` by the client, so both forms work.
- The 400 error message now points agents at the filter rather than the author ID
  when `publicationDateOrYear` is set, eliminating the dead-end "Use a Semantic
  Scholar authorId" message when the ID is actually valid.
- The golden path example and tool description have been updated to use the
  canonical colon form (`"2022:"`).

### 9. `get_author_papers` `fields` accepts paper fields, not author fields

- `get_author_papers` returns paper records, so its `fields` parameter accepts
  paper field names such as `title`, `year`, `authors`, `citationCount`, `abstract`,
  `url`, `publicationDate`, etc.
- Passing author-profile fields (e.g. `hIndex`, `affiliations`) to
  `get_author_papers` raises a validation error pointing at the supported paper
  fields.
- Omitting `fields` returns the default paper field set.
- The validated paper fields are: `paperId`, `title`, `abstract`, `year`,
  `authors`, `citationCount`, `referenceCount`, `influentialCitationCount`,
  `venue`, `publicationTypes`, `publicationDate`, `url`, `externalIds`,
  `fieldsOfStudy`, `s2FieldsOfStudy`, `isOpenAccess`, `openAccessPdf`.

### 10. `search_papers_bulk` default ordering is now explicitly disclosed

- `search_papers_bulk` uses exhaustive corpus traversal with an internal
  ordering that is **not relevance-ranked** — it is a semantic pivot from
  `search_papers`, not "page 2".
- Every `search_papers_bulk` response now includes a `retrievalNote` field
  that describes the active ordering contract so agents don't need extra round
  trips to infer it.
- `brokerMetadata.nextStepHint` from `search_papers` (Semantic Scholar path)
  now explicitly says the bulk ordering is NOT relevance-ranked and suggests
  `sort='citationCount:desc'` as an alternative.
- The `search_papers_bulk` tool description also calls out this ordering
  difference prominently.
- Agents should read `retrievalNote` before deciding whether bulk retrieval
  serves their intent; for relevance-ranked small pages, stick with
  `search_papers` or `search_papers_semantic_scholar`.

## Future Work

- Consider per-request or per-session provider preferences for budget-aware use.
- Decide whether retry-recovered provider behavior should remain internal or be
  surfaced to agents as part of the broker story.
- Consider whether the agentic workflow should eventually split into multiple
  checked-in workflows once the new manual mode/focus inputs settle and more
  specialized UX probes become routine.
- Evaluate whether OpenAlex institution/source entity pivots (`search_entities_openalex`,
  `search_papers_openalex_by_entity`) should appear more prominently in a dedicated
  golden path — they are first-class tools but currently listed only under secondary
  workflows.
- Consider adding a dedicated ECOS-to-Federal-Register walkthrough golden path
  now that the full species-dossier-to-regulatory-source chain
  (ECOS → frCitation → GovInfo/FR) is stable and documented.
