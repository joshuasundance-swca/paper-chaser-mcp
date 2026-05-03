"""Published tool descriptions and shared prose fragments."""

OPAQUE_CURSOR_CONTRACT = (
    "Treat pagination.nextCursor as an opaque server-issued token: pass it back "
    "exactly as returned, do not derive, edit, or fabricate it, and do not reuse "
    "it across a different tool or different query flow so pagination state "
    "integrity stays bound to the stream that produced it."
)

TOOL_DESCRIPTIONS = {
    "research": (
        "Default guided entry point for low-context research requests. Use this for topic discovery, "
        "known-item recovery, citation repair, and regulatory or species-history questions when you want "
        "one trust-graded response instead of choosing among raw tools. The server picks the safest retrieval "
        "path, applies a server-owned quality-first policy for this guided flow, returns resultStatus, "
        "answerability, routingSummary, coverageSummary, evidence, leads, and source records, "
        "and abstains or marks partial results "
        "when evidence is weak, off-topic, or incomplete. "
        "Compact abstention paths may surface suppressedSourceSummaries with minimal {sourceId, title, "
        "topicalRelevance, reason} records for excluded weak/off-topic items. "
        "Additive trust and grounding signals include confidenceSignals.evidenceQualityProfile, "
        "confidenceSignals.synthesisMode, confidenceSignals.evidenceProfileDetail, "
        "confidenceSignals.synthesisPath, confidenceSignals.trustRevisionNarrative, and the "
        "trustSummary.authoritativeButWeak missed-escalation bucket for primary-source records "
        "that are authoritative but not topically responsive (surface to a human or escalate "
        "to disambiguation rather than treat as grounded evidence). searchStrategy may expose regulatoryIntent "
        "(current_cfr_text, rulemaking_history, species_dossier, guidance_lookup, "
        "hybrid_regulatory_plus_literature), intentFamily, subjectCard for species/regulatory "
        "grounding, and subjectChainGaps describing missing subject-chain evidence."
    ),
    "follow_up_research": (
        "Grounded follow-up over a prior guided research result. Use searchSessionId from research and ask one "
        "specific question. This tool answers only when the saved evidence is strong enough, selects exact "
        "evidence ids when it can ground the answer, and otherwise returns an explicit abstention or "
        "insufficient-evidence response plus next actions. Compact responses may also surface "
        "suppressedSourceSummaries with minimal {sourceId, title, topicalRelevance, reason} records for "
        "excluded weak/off-topic items. "
        "Responses carry the same additive trust signals as research: "
        "confidenceSignals.evidenceQualityProfile, confidenceSignals.synthesisMode, "
        "confidenceSignals.evidenceProfileDetail, confidenceSignals.synthesisPath, "
        "confidenceSignals.trustRevisionNarrative, and the trustSummary.authoritativeButWeak "
        "missed-escalation bucket (authoritative primary-source records that are not topically responsive); "
        "searchStrategy fields (regulatoryIntent, intentFamily, subjectCard, subjectChainGaps) are "
        "inherited from the saved session."
    ),
    "resolve_reference": (
        "Resolve one reference-like input into the safest next anchor. Accepts citations, DOI strings, DOI URLs, "
        "arXiv IDs, title fragments, and regulatory references. Returns the best match when one is trustworthy, "
        "alternatives when ambiguity remains, and direct next actions "
        "when the input should pivot into a primary-source path."
    ),
    "inspect_source": (
        "Inspect one source from a prior guided research result. Pass the searchSessionId and an evidence id, "
        "source alias, or source id from the research response to get provenance, trust state, source-access details, "
        "and the best direct-read follow-through. "
        "Responses surface whyClassifiedAsWeakMatch (a one-sentence rationale explaining why an "
        "authoritative source was treated as a weak or off-topic match) and "
        "directReadRecommendationDetails entries shaped as {trustLevel, whyRecommended, cautions} "
        "so agents can prioritize direct reads by quality instead of position."
    ),
    "get_runtime_status": (
        "Guided runtime and provider-status summary. Use this to confirm the active tool profile, transport, "
        "effective smart provider, enabled coverage, and explicit "
        "disabled/suppressed/degraded/quota-limited provider sets, "
        "and high-level warnings without reading low-level diagnostics."
    ),
    "search_papers": (
        "Primary entry point for quick literature discovery: start here when "
        "quick topic exploration needs one strong first page fast. Best-effort "
        "paper search tries Semantic Scholar → arXiv → CORE → "
        "SerpApi Google Scholar in order by default. CORE is disabled by "
        "default until its reliability gate is re-earned, and SerpApi stays "
        "guarded behind explicit enablement because it is paid. "
        "Use preferredProvider to try one provider first, or providerOrder to "
        "override the broker chain for this call when source constraints matter. "
        "Provider names accepted by those arguments are core, semantic_scholar, "
        "arxiv, scholarapi, and either serpapi or serpapi_google_scholar. "
        "Optional filters: year, venue, publicationDateOrYear, "
        "fieldsOfStudy, publicationTypes, openAccessPdf, minCitationCount. "
        "Providers that cannot honor a requested advanced filter are skipped "
        "instead of silently widening the query; explicit ScholarAPI routing can "
        "also honor openAccessPdf through its PDF-availability filter. "
        "Returns a single page of results (no pagination). For large paginated "
        "retrieval use search_papers_bulk. "
        "brokerMetadata.providerUsed identifies which provider supplied the results, "
        "and brokerMetadata.attemptedProviders explains skips, failures, and fallbacks."
    ),
    "search_papers_core": (
        "Search papers using CORE only. Returns a single page of results with the "
        "same normalized response shape as search_papers, but does not fall back "
        "to other providers. Supported inputs are query, limit, and year."
    ),
    "search_papers_semantic_scholar": (
        "Search papers using Semantic Scholar only. Returns a single page of "
        "results with the same normalized response shape as search_papers, but "
        "does not fall back to other providers. This is the provider-specific "
        "tool that honors fields, year, venue, publicationDateOrYear, "
        "fieldsOfStudy, publicationTypes, openAccessPdf, and minCitationCount."
    ),
    "search_papers_serpapi": (
        "Search papers using SerpApi Google Scholar only. Requires "
        "PAPER_CHASER_ENABLE_SERPAPI=true and SERPAPI_API_KEY. Returns a "
        "single page of results with the same normalized response shape as "
        "search_papers, but does not fall back to other providers. Supported "
        "inputs are query, limit, and year."
    ),
    "search_papers_scholarapi": (
        "Search papers using ScholarAPI only. Requires "
        "PAPER_CHASER_ENABLE_SCHOLARAPI=true and SCHOLARAPI_API_KEY. Returns a "
        "single explicit ScholarAPI relevance-ranked page with normalized paper "
        "results, full-text/PDF availability flags, and an opaque cursor for "
        "continuing the same ScholarAPI query. Supported inputs are query, "
        "limit, cursor, indexed/published date bounds, has_text, and has_pdf. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "search_papers_arxiv": (
        "Search papers using arXiv only. Returns a single page of results with "
        "the same normalized response shape as search_papers, but does not fall "
        "back to other providers. Supported inputs are query, limit, and year."
    ),
    "search_papers_openalex": (
        "Search papers using OpenAlex only. Returns a single explicit OpenAlex "
        "page with the same normalized top-level response shape as search_papers, "
        "but does not fall back to other providers. Supported inputs are query, "
        "limit, and year. OpenAlex does not expose a Semantic Scholar-style "
        "best-match endpoint here; use get_paper_details_openalex for DOI or "
        "OpenAlex ID lookup and search_papers_openalex_bulk for cursor-paginated "
        "OpenAlex traversal."
    ),
    "search_papers_openalex_bulk": (
        "Paginated OpenAlex paper search for explicit OpenAlex retrieval flows. "
        "Uses OpenAlex cursor pagination, returns up to 200 papers per call, and "
        "treats pagination.nextCursor as an opaque server-issued token. Supported "
        "inputs are query, limit, year, and cursor. Use this when you explicitly "
        "want OpenAlex-native paging rather than Semantic Scholar bulk search. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "list_papers_scholarapi": (
        "Ingestion-oriented ScholarAPI listing for monitoring or exhaustive date-window scans. "
        "Requires PAPER_CHASER_ENABLE_SCHOLARAPI=true and SCHOLARAPI_API_KEY. "
        "This uses ScholarAPI /list semantics, sorted by indexed_at rather than relevance, "
        "and is the right continuation path for ScholarAPI monitoring workflows rather than "
        "search_papers_bulk. Supported inputs are optional query, limit, cursor, indexed/published "
        "date bounds, has_text, and has_pdf. Treat any query as a narrowing filter on an indexed-at "
        "stream rather than a ranked topical search; use search_papers_scholarapi for discovery. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "search_papers_bulk": (
        "Primary exhaustive retrieval tool for serious research, datasets, or "
        "multi-page collection. Paginated bulk paper search (Semantic Scholar) "
        "with advanced boolean query syntax. Supports sorting and returns up to "
        "1,000 papers per call, but the upstream bulk endpoint may ignore small "
        "limit values internally; this server truncates the returned data to the "
        "requested limit. Prefer search_papers or search_papers_semantic_scholar "
        "for small targeted pages. "
        "IMPORTANT ORDERING: The default bulk ordering is NOT relevance-ranked — "
        "it is exhaustive corpus traversal with an internal ordering. This is NOT "
        "'page 2' of search_papers; the ranking semantics differ and results may "
        "look unrelated to the search_papers discovery page. For citation-ranked "
        "bulk retrieval pass sort='citationCount:desc'. Every response includes a "
        "retrievalNote field that describes the active ordering contract. "
        "IMPORTANT PROVIDER: This tool always uses Semantic Scholar, regardless "
        "of which provider search_papers used. If your previous search_papers "
        "call returned results from CORE, arXiv, or SerpApi "
        "(brokerMetadata.providerUsed is not 'semantic_scholar', or "
        "brokerMetadata.bulkSearchIsProviderPivot is true), calling this tool is "
        "a provider pivot to Semantic Scholar — not a continuation from the "
        "original provider. "
        "Example first call: {query: 'transformers', limit: 100}. "
        "Use cursor=pagination.nextCursor from the response to fetch the next "
        "page; hasMore signals when more results exist. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "search_papers_match": (
        "Known-item lookup for messy or partial titles. Takes a single query "
        "string (the paper title or partial title text) and finds the single "
        "paper whose title best matches. This tool does NOT accept separate "
        "author, year, or venue fields; use resolve_citation when you have a "
        "multi-field bibliographic reference to repair. If the upstream "
        "exact-match endpoint misses a punctuation-heavy title, the server "
        "falls back to a fuzzy Semantic Scholar title search instead of "
        "surfacing a raw 404, then tries strict OpenAlex and Crossref "
        "exact-title recovery before returning a structured no-match payload. "
        "Optional includeEnrichment=true adds Crossref, Unpaywall, and OpenAlex "
        "metadata only to the final matched paper, never to the candidate-selection path. "
        "A no-match payload can still mean the item is a dissertation, software "
        "release, report, or other output outside the indexed paper surface."
    ),
    "resolve_citation": (
        "First-class citation repair workflow for incomplete, malformed, or "
        "almost-right references. Starts with DOI/arXiv/URL extraction, then "
        "tries title-style recovery, quote/snippet recovery, and sparse "
        "metadata search to return the most likely canonical paper plus "
        "alternatives, confidence, disagreements, and the fastest next step "
        "for disambiguation. Start here for broken bibliography lines before "
        "trying title match or broad search. Report-style or non-paper-looking "
        "inputs prefer abstention plus alternatives over forcing a weak "
        "canonical paper match. If the citation looks regulatory, such as a "
        "Federal Register or CFR reference, use search_federal_register, "
        "get_federal_register_document, or get_cfr_text instead of treating it "
        "as a paper. Optional includeEnrichment=true enriches only the final "
        "bestMatch.paper after resolution."
    ),
    "paper_autocomplete": ("Return paper title completions for a partial query string."),
    "paper_autocomplete_openalex": (
        "Return lightweight OpenAlex work autocomplete matches for a partial "
        "paper title. This is useful when an agent wants OpenAlex-native title "
        "hints before pivoting into explicit OpenAlex work lookup or entity flows."
    ),
    "get_paper_details": (
        "Known-item lookup when you already have an identifier. Get paper "
        "details from a DOI, ArXiv ID, Semantic Scholar ID, or URL. "
        "Optional includeEnrichment=true adds post-resolution Crossref, "
        "Unpaywall, and OpenAlex metadata to the final paper without changing "
        "the lookup path."
    ),
    "get_paper_text_scholarapi": (
        "Fetch one full plain-text document from ScholarAPI by ScholarAPI paper id. "
        "Requires PAPER_CHASER_ENABLE_SCHOLARAPI=true and SCHOLARAPI_API_KEY. "
        "Accepts either a raw ScholarAPI id or a ScholarAPI:<id> value returned by search. "
        "Use this when the workflow explicitly needs accessible full text rather than only metadata."
    ),
    "get_paper_texts_scholarapi": (
        "Fetch full plain-text content for up to 100 ScholarAPI paper ids in one call. "
        "Requires PAPER_CHASER_ENABLE_SCHOLARAPI=true and SCHOLARAPI_API_KEY. "
        "Each paper id may be raw or namespaced as ScholarAPI:<id>. "
        "Preserves request order and keeps null placeholders for items whose text is unavailable."
    ),
    "get_paper_pdf_scholarapi": (
        "Fetch one PDF from ScholarAPI by ScholarAPI paper id. Requires "
        "PAPER_CHASER_ENABLE_SCHOLARAPI=true and SCHOLARAPI_API_KEY. Returns "
        "structured PDF metadata plus base64-encoded binary content. Accepts either "
        "a raw ScholarAPI id or a ScholarAPI:<id> value returned by search."
    ),
    "get_paper_metadata_crossref": (
        "Explicit Crossref paper enrichment. Use this after you already have a "
        "paper, DOI, or DOI-bearing identifier and want Crossref's DOI, "
        "publisher, venue/container, publication date, type, and citation-count "
        "metadata. Accepts paper_id or doi, and Crossref alone also allows a "
        "query fallback when no DOI can be resolved."
    ),
    "get_paper_open_access_unpaywall": (
        "Explicit Unpaywall open-access enrichment. Use this after you already "
        "have a DOI or DOI-bearing identifier and want OA status, best OA URL, "
        "PDF URL, license, and DOAJ status. Requires UNPAYWALL_EMAIL because "
        "the upstream API is DOI-based and expects a contact email."
    ),
    "enrich_paper": (
        "Combined Crossref, Unpaywall, and OpenAlex enrichment for one known "
        "paper, DOI, or DOI-bearing identifier. Runs DOI resolution first, then "
        "fetches additive metadata from the enabled enrichment providers and returns "
        "one merged enrichments object plus per-provider results. This is additive "
        "metadata only; it does not re-rank or re-resolve the base paper."
    ),
    "get_paper_details_openalex": (
        "Known-item lookup using OpenAlex semantics. Get one OpenAlex work by "
        "OpenAlex W-id, OpenAlex work URL, or DOI. This path reconstructs a "
        "plaintext abstract from OpenAlex's abstract_inverted_index when possible."
    ),
    "get_paper_citations": (
        "Citation chasing outward: get papers that cite this paper (cited by). "
        "Prefer paper.recommendedExpansionId from brokered search results. If "
        "paper.expansionIdStatus is not_portable, do not reuse brokered "
        "paperId/sourceId/canonicalId values directly; resolve the paper "
        "through DOI or a Semantic Scholar-native lookup first. "
        "Pass cursor=pagination.nextCursor to continue; hasMore signals more pages. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "get_paper_references": (
        "Citation chasing backward: get the references this paper cites. "
        "Prefer paper.recommendedExpansionId from brokered search results. If "
        "paper.expansionIdStatus is not_portable, do not reuse brokered "
        "paperId/sourceId/canonicalId values directly; resolve the paper "
        "through DOI or a Semantic Scholar-native lookup first. "
        "Pass cursor=pagination.nextCursor to continue; hasMore signals more pages. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "get_paper_citations_openalex": (
        "OpenAlex cited-by expansion. Uses the work's cited_by_api_url under the "
        "hood and keeps OpenAlex cursor pagination opaque and server-issued. Pass "
        "an OpenAlex W-id, OpenAlex work URL, or DOI as paper_id. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "get_paper_references_openalex": (
        "OpenAlex backward-reference expansion. Hydrates referenced_works in "
        "batched OpenAlex ID lookups instead of one request per reference, then "
        "returns an opaque server-issued cursor for the next slice. Pass an "
        "OpenAlex W-id, OpenAlex work URL, or DOI as paper_id. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "get_paper_authors": (
        "Get authors of a paper so you can pivot from a paper into an author or "
        "collaboration workflow. Prefer paper.recommendedExpansionId from "
        "brokered search results. If paper.expansionIdStatus is not_portable, "
        "do not reuse brokered paperId/sourceId/canonicalId values directly; "
        "resolve the paper through DOI or a Semantic Scholar-native lookup "
        "first. "
        "Pass cursor=pagination.nextCursor to continue; hasMore signals more pages. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "get_author_info": (
        "Get author details for an author-centric workflow after search_authors "
        "or get_paper_authors. author_id must be a Semantic Scholar authorId from "
        "those tools."
    ),
    "get_author_info_openalex": (
        "Get one OpenAlex author profile by OpenAlex A-id or OpenAlex author URL "
        "for an explicit OpenAlex author workflow."
    ),
    "get_author_papers": (
        "Author-centric workflow step: get papers by an author, including recent "
        "or filtered work. author_id must be a Semantic Scholar authorId from "
        "search_authors or get_paper_authors. "
        "The optional fields parameter selects paper fields (e.g. title, year, "
        "authors, citationCount); omit to get the default paper field set. "
        "Pass cursor=pagination.nextCursor to continue; hasMore signals more pages. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "search_authors": (
        "Primary author-search entry point. Search for authors by name before "
        "expanding to get_author_info or get_author_papers. Plain-text only: the "
        "server normalizes exact-name punctuation such as initials before calling "
        "Semantic Scholar. For common names, add affiliation, coauthor, venue, or "
        "topic clues, then confirm identity with get_author_info/get_author_papers. "
        "Pass cursor=pagination.nextCursor to continue; hasMore signals more pages. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "search_authors_openalex": (
        "Search OpenAlex authors by name for an explicit OpenAlex author workflow. "
        "This is the first step before get_author_info_openalex or "
        "get_author_papers_openalex; for common names, confirm the right person "
        "with affiliation or profile metadata before expanding papers. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "search_entities_openalex": (
        "Search OpenAlex sources, institutions, or topics explicitly. Use this "
        "when venue, affiliation, or topic disambiguation matters and you want "
        "an OpenAlex-native pivot target before expanding to papers. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "search_papers_openalex_by_entity": (
        "Return OpenAlex works filtered by one explicit source, institution, or "
        "topic entity ID. This is the follow-on tool after search_entities_openalex "
        "when you want an OpenAlex-native venue, affiliation, or topic pivot. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "get_author_papers_openalex": (
        "Get papers for one OpenAlex author by OpenAlex A-id or author URL, with "
        "optional year filtering and OpenAlex cursor pagination. This keeps the "
        "guide's two-step OpenAlex author flow explicit: search_authors_openalex "
        "or get_author_info_openalex first, then expand with this tool. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "batch_get_authors": "Get details for multiple authors at once (up to 1,000).",
    "search_snippets": (
        "Special-purpose recovery tool for quote-like or phrase-based retrieval "
        "when title or keyword search is weak. Returns snippet text plus paper "
        "metadata and relevance score. If Semantic Scholar rejects the phrase "
        "query or is temporarily unavailable, the server degrades to an empty "
        "result with retry guidance instead of surfacing a raw provider 4xx/5xx."
    ),
    "get_paper_recommendations": "Get similar paper recommendations for a paper.",
    "get_paper_recommendations_post": ("Get paper recommendations from positive and negative seed sets."),
    "batch_get_papers": "Get details for multiple papers (up to 500).",
    "get_paper_citation_formats": (
        "Citation export step after discovery: get MLA, APA, BibTeX, and other "
        "formats for a Google Scholar paper. Requires "
        "PAPER_CHASER_ENABLE_SERPAPI=true and SERPAPI_API_KEY (paid SerpApi "
        "service, results cached 1 hour). "
        "Pass result_id=paper.scholarResultId (NOT paper.sourceId) from a "
        "serpapi_google_scholar search_papers result. "
        "paper.scholarResultId is the raw Scholar result_id; paper.sourceId may "
        "be a cluster_id or cites_id, which this tool cannot use. "
        "If paper.scholarResultId is absent, the paper cannot be used with this "
        "tool. "
        "Returns text citation strings and structured export links (BibTeX, "
        "EndNote, RefMan, RefWorks). "
        "Not paginated — single response per paper."
    ),
    "search_papers_serpapi_cited_by": (
        "Explicit SerpApi cited-by expansion for one Google Scholar cites_id. "
        "Use this only when PAPER_CHASER_ENABLE_SERPAPI=true and the workflow "
        "really needs Scholar's cited-by surface, recall recovery, or search-within-"
        "citing-articles behavior. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "search_papers_serpapi_versions": (
        "Explicit SerpApi all-versions expansion for one Google Scholar cluster_id. "
        "Use this to inspect alternate copies or clustered variants of a result "
        "without turning SerpApi into the default broad search path. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "get_author_profile_serpapi": (
        "Fetch one Google Scholar author profile through SerpApi, including top-"
        "level citation summary, interests, and co-authors. This is a paid "
        "provider-specific workflow and should be used intentionally."
    ),
    "get_author_articles_serpapi": (
        "Return normalized Google Scholar author articles through SerpApi. Use "
        "this after get_author_profile_serpapi when an explicit Scholar author "
        "publication list is required. "
        f"{OPAQUE_CURSOR_CONTRACT}"
    ),
    "get_serpapi_account_status": (
        "Return SerpApi account and quota metadata, including remaining search "
        "budget and hourly throughput guidance. Returns a sanitized public "
        "summary only; raw upstream credentials and secret-like account fields "
        "are never exposed. Use this before expensive or deep Scholar recovery "
        "workflows."
    ),
    "get_provider_diagnostics": (
        "Return shared provider-health diagnostics for Semantic Scholar, OpenAlex, "
        "CORE, arXiv, SerpApi, Crossref, Unpaywall, OpenAI, and ECOS. Includes "
        "suppression state, recent rate limits and failures, normalized "
        "outcome envelopes, and a runtimeSummary that exposes effective transport, "
        "enabled/disabled providers, broker order, embeddings state, and warnings "
        "so transport and provider issues are visible without reading raw logs."
    ),
    "search_species_ecos": (
        "Search the U.S. Fish and Wildlife Service ECOS species catalog using "
        "structured Pull Reports filters instead of page scraping. "
        "matchMode=auto tries exact common/scientific-name matches first, then "
        "falls back to prefix matching. Returns compact species hits with ECOS "
        "species ids and profile URLs. Best path for regulatory wildlife work: "
        "search_species_ecos -> get_species_profile_ecos -> list_species_documents_ecos "
        "-> get_document_text_ecos."
    ),
    "get_species_profile_ecos": (
        "Fetch one ECOS species dossier by species id or species URL using the "
        "structured JSON that powers the public species page. Returns the core "
        "species record, per-entity listings, life history, range summary, "
        "grouped documents, and conservation-plan links."
    ),
    "list_species_documents_ecos": (
        "Flatten one ECOS species dossier into a sorted document inventory. "
        "Supports recovery plans, five-year reviews, biological opinions, "
        "federal-register documents, other recovery documents, and "
        "conservation-plan links."
    ),
    "get_document_text_ecos": (
        "Fetch one ECOS or ECOS-linked document URL, follow redirects, detect "
        "content type, and convert PDF/HTML/text content to Markdown for "
        "downstream analysis. Returns extractionStatus plus warnings when a "
        "document is too large, unsupported, nearly empty, conversion-timed "
        "out, or failed to fetch. Prefer the direct PDF or document URL returned "
        "by list_species_documents_ecos when available; landing pages and "
        "intermediate ECOS links are more brittle. Do not use this as the primary "
        "path for GovInfo Federal Register issue links such as /link/fr/... or full "
        "daily FR package PDFs; prefer get_federal_register_document for Register items."
    ),
    "search_federal_register": (
        "Search FederalRegister.gov for notices, rules, proposed rules, and "
        "presidential documents. Supports agency, document-type, publication-date, "
        "document-number, and CFR-aware narrowing. Use this for discovery, "
        "especially when you have a topic, FR citation, or CFR clue but not a "
        "stable document number yet. If you already have an exact document number, "
        "prefer get_federal_register_document for direct retrieval. Multi-filter "
        "queries are supported, but the best low-round-trip path is usually query + "
        "agency/date narrowing first, then exact document retrieval. "
        "NOTE: Biological opinions, Section 7 consultation records, and incidental "
        "take permits are NOT published in the Federal Register; they live in the "
        "USFWS ECOS/TAILS system. For agency consultation history, use "
        "search_species_ecos -> list_species_documents_ecos instead of this tool."
    ),
    "get_federal_register_document": (
        "Resolve one Federal Register document from a document number, FR citation, "
        "or GovInfo FR link. Prefers authoritative GovInfo granule content when it "
        "can resolve package/granule ids cleanly, and falls back to FederalRegister.gov "
        "HTML when GovInfo is unavailable. Best direct-retrieval path once you know "
        "the document number. Historical FR citations can be less reliable than "
        "modern document numbers, so use search_federal_register first when an exact "
        "citation-only lookup is ambiguous."
    ),
    "get_cfr_text": (
        "Resolve CFR part or section text through GovInfo using title/part/section "
        "citation semantics, with an explicit volume-resolution step and a preference "
        "for XML over HTML or PDF. Use this when you need authoritative regulatory text, "
        "not broad document discovery."
    ),
    "search_papers_smart": (
        "Agent-oriented concept and literature-review search. Starts from a broad "
        "concept, known item, author clue, or citation seed; runs grounded query "
        "expansion, multi-provider retrieval, deduplication, reranking, and stores "
        "the result set under searchSessionId for follow-up QA, landscape mapping, "
        "and graph expansion. latencyProfile defaults to deep for the highest-quality "
        "smart retrieval; use balanced when lower latency matters, and reserve fast "
        "for smoke tests or debugging. providerBudget lets advanced clients cap total, "
        "per-provider, or paid usage for one smart search, and includeEnrichment "
        "adds Crossref + Unpaywall metadata only to the final returned hits after "
        "ranking is complete. Returns compact smart hits, strategyMetadata, resultStatus, "
        "answerability, routingSummary, evidence, leads, evidenceGaps, structuredSources, "
        "coverage/failure summaries, agentHints, resourceUris, and a concrete next-step "
        "recommendation. Legacy trust fields remain available as compatibility views. Best "
        "entry point for concept discovery, but treat it as a lead generator on "
        "sparse cross-domain queries and inspect strategyMetadata, driftWarnings, "
        "and the returned evidence before trusting borderline results. In known-item "
        "mode, if exact paper recovery is weak, the tool now falls back to a broader "
        "candidate set instead of ending in a dead-end configuration error. For clearly "
        "regulatory or species-history queries, auto mode can route into ECOS, Federal "
        "Register, and CFR retrieval first and return a regulatoryTimeline instead of "
        "paper-centric ranking."
    ),
    "ask_result_set": (
        "Grounded follow-up over a saved searchSessionId. Answer a question using "
        "only the papers in that result set, returning evidence for every claim. "
        "answerMode supports qa, claim_check, and comparison, and latencyProfile "
        "defaults to deep for the best grounded answer quality. Use balanced when "
        "lower latency matters and fast only for smoke tests. "
        "comparison mode prefers a grounded structured comparison over flat "
        "title-and-venue enumeration and, when possible, separates directly aligned "
        "papers from broader analog or context papers. Best used after search_papers_smart when "
        "the question stays inside the same corpus. Treat the answer as grounded "
        "synthesis, not as a replacement for reading the cited evidence snippets."
    ),
    "map_research_landscape": (
        "Cluster a saved searchSessionId into 3-5 themes, representative papers, "
        "gaps, disagreements, and suggested next searches. Use this when an "
        "agent needs a literature-review map rather than another flat result page. "
        "latencyProfile defaults to deep for the strongest theme labeling and "
        "summarization. Use balanced when lower latency matters and fast only for "
        "smoke tests. "
        "and weak theme labels are sanitized before response emission. Theme summaries "
        "now prioritize representative paper anchors and dominant cluster terms over "
        "token-only labels. Use this "
        "to orient on a corpus, then verify important themes against the returned "
        "representative papers rather than trusting labels alone."
    ),
    "expand_research_graph": (
        "Expand a saved search session or explicit paper seeds into a compact "
        "citation, reference, or author graph. Provide either seedSearchSessionId "
        "(from a prior search_papers_smart call) or seedPaperIds (a list of "
        "Semantic Scholar paper IDs). At least one seed source is required. "
        "direction controls the expansion axis: 'citations' (cited-by), "
        "'references' (backward), or 'authors'. Returns nodes, edges, a ranked "
        "frontier, agentHints, and resourceUris for continued exploration. "
        "latencyProfile defaults to deep for the strongest graph scoring. Use balanced "
        "when lower latency matters and fast only for smoke tests. It does not change the explicit "
        "seed, direction, or hop inputs. Saved-session expansion keeps the "
        "original search intent in frontier scoring and filters weak off-topic "
        "next-hop candidates more aggressively. Best follow-on after search_papers_smart "
        "or a known portable seed paper. If a seed is brokered and expansionIdStatus "
        "is not_portable, resolve it through DOI or a Semantic Scholar-native lookup "
        "before retrying graph expansion."
    ),
}

__all__ = ["OPAQUE_CURSOR_CONTRACT", "TOOL_DESCRIPTIONS"]
