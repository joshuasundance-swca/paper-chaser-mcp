"""Published tool descriptions and shared prose fragments."""

OPAQUE_CURSOR_CONTRACT = (
    "Treat pagination.nextCursor as an opaque server-issued token: pass it back "
    "exactly as returned, do not derive, edit, or fabricate it, and do not reuse "
    "it across a different tool or different query flow so pagination state "
    "integrity stays bound to the stream that produced it."
)

TOOL_DESCRIPTIONS = {
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
        "arxiv, and either serpapi or serpapi_google_scholar. "
        "Optional filters: year, venue, publicationDateOrYear, "
        "fieldsOfStudy, publicationTypes, openAccessPdf, minCitationCount. "
        "Semantic Scholar-only filters still cause CORE and SerpApi to be "
        "skipped even if they appear in providerOrder. "
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
        "SCHOLAR_SEARCH_ENABLE_SERPAPI=true and SERPAPI_API_KEY. Returns a "
        "single page of results with the same normalized response shape as "
        "search_papers, but does not fall back to other providers. Supported "
        "inputs are query, limit, and year."
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
        "Known-item lookup for messy or partial titles. Find the single paper "
        "whose title best matches the query string. If the upstream exact-match "
        "endpoint misses a punctuation-heavy title, the server falls back to a "
        "fuzzy Semantic Scholar title search instead of surfacing a raw 404. "
        "Optional includeEnrichment=true adds Crossref and Unpaywall metadata "
        "only to the final matched paper, never to the candidate-selection path. "
        "A no-match payload can still mean the item is a dissertation, software "
        "release, report, or other output outside the indexed paper surface."
    ),
    "resolve_citation": (
        "First-class citation repair workflow for incomplete, malformed, or "
        "almost-right references. Starts with DOI/arXiv/URL extraction, then "
        "tries title-style recovery, quote/snippet recovery, and sparse "
        "metadata search to return the most likely canonical paper plus "
        "alternatives, confidence, disagreements, and the fastest next step "
        "for disambiguation. Optional includeEnrichment=true enriches only the "
        "final bestMatch.paper after resolution."
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
        "Optional includeEnrichment=true adds post-resolution Crossref and "
        "Unpaywall metadata to the final paper without changing the lookup path."
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
        "Combined Crossref + Unpaywall enrichment for one known paper, DOI, or "
        "DOI-bearing identifier. Runs DOI resolution first, then Crossref, then "
        "Unpaywall, and returns one merged enrichments object plus per-provider "
        "results. This is additive metadata only; it does not re-rank or "
        "re-resolve the base paper."
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
        "SCHOLAR_SEARCH_ENABLE_SERPAPI=true and SERPAPI_API_KEY (paid SerpApi "
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
        "Use this only when SCHOLAR_SEARCH_ENABLE_SERPAPI=true and the workflow "
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
        "budget and hourly throughput guidance. Use this before expensive or "
        "deep Scholar recovery workflows."
    ),
    "get_provider_diagnostics": (
        "Return shared provider-health diagnostics for Semantic Scholar, OpenAlex, "
        "CORE, arXiv, SerpApi, Crossref, Unpaywall, OpenAI, and ECOS. Includes "
        "suppression state, recent rate limits and failures, and normalized "
        "outcome envelopes so transport and provider issues are visible without "
        "reading raw logs."
    ),
    "search_species_ecos": (
        "Search the U.S. Fish and Wildlife Service ECOS species catalog using "
        "structured Pull Reports filters instead of page scraping. "
        "matchMode=auto tries exact common/scientific-name matches first, then "
        "falls back to prefix matching. Returns compact species hits with ECOS "
        "species ids and profile URLs."
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
        "out, or failed to fetch."
    ),
    "search_papers_smart": (
        "Agent-oriented concept and literature-review search. Starts from a broad "
        "concept, known item, author clue, or citation seed; runs grounded query "
        "expansion, multi-provider retrieval, deduplication, reranking, and stores "
        "the result set under searchSessionId for follow-up QA, landscape mapping, "
        "and graph expansion. Optional latencyProfile supports fast, balanced, and "
        "deep execution modes, providerBudget lets advanced clients cap total, "
        "per-provider, or paid usage for one smart search, and includeEnrichment "
        "adds Crossref + Unpaywall metadata only to the final returned hits after "
        "ranking is complete. Returns compact smart hits, strategyMetadata, "
        "agentHints, resourceUris, and a concrete next-step recommendation."
    ),
    "ask_result_set": (
        "Grounded follow-up over a saved searchSessionId. Answer a question using "
        "only the papers in that result set, returning evidence for every claim. "
        "answerMode supports qa, claim_check, and comparison, and latencyProfile "
        "lets callers choose faster deterministic synthesis when needed."
    ),
    "map_research_landscape": (
        "Cluster a saved searchSessionId into 3-5 themes, representative papers, "
        "gaps, disagreements, and suggested next searches. Use this when an "
        "agent needs a literature-review map rather than another flat result page. "
        "latencyProfile controls how much model work is spent on theme labeling."
    ),
    "expand_research_graph": (
        "Expand a saved search session or explicit paper seeds into a compact "
        "citation, reference, or author graph. Returns nodes, edges, a ranked "
        "frontier, agentHints, and resourceUris for continued exploration. "
        "latencyProfile controls graph scoring cost without changing the explicit "
        "seed, direction, or hop inputs."
    ),
}

__all__ = ["OPAQUE_CURSOR_CONTRACT", "TOOL_DESCRIPTIONS"]
