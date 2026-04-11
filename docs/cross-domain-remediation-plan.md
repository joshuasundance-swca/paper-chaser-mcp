# Cross-Domain Remediation Plan

This document turns the latest broad Paper Chaser tool exercise into a durable implementation plan. The exercise intentionally tested the guided workflow across environmental science, regulatory retrieval, environmental justice, cultural resources, archaeology, anthropology-adjacent stewardship questions, and known-item recovery. The goal is not just to fix individual failures. The goal is to make the guided public surface behave like a credible cross-domain research assistant for real practitioner workflows.

## Stress-Test Remediation Completed (Prerequisite Foundations)

Before starting the workstreams below, a comprehensive 8-phase stress-test
remediation addressed foundational issues in schema integrity, answer
validation, provider runtime contracts, payload efficiency, citation repair,
and regulatory routing. Key outcomes relevant to this plan:

- **Schema integrity**: `coverageSummary.providersSucceeded` now excludes
  zero-result providers; `failureSummary` contracts are internally consistent;
  `activeProviderSet` excludes suppressed providers; author dedup is reliable.
- **Answer validation**: LLM-backed answer-status validation with deterministic
  refusal-pattern fallback; `classify_answerability()` is now more trustworthy.
- **Provider runtime**: `canAnswerFollowUp` is capability-based; SerpApi venue
  parsing rejects author-pattern strings.
- **Payload efficiency**: follow-up responses only include referenced sources;
  null/empty fields stripped from evidence and source records.
- **Citation repair**: year-mismatch penalty tightened; upstream confidence
  bonus reduced; title-similarity length penalty added; confidence capped on
  title conflicts.
- **Regulatory routing**: verification-status default tightened to `unverified`
  for papers without identifiers; regulatory source-type vocabulary expanded
  from 4 to 13 entries.
- **Test coverage**: 117 new tests across 6 modules (962 total, all passing).

These fixes provide a stronger foundation for the workstreams below, especially
Workstreams B (relevance classification), C (query routing), D (regulatory
coverage), E (known-item recovery), and F (follow-up trust).

## Why this plan exists

The recent tool exercise showed that the current guided surface is already useful in some ways:

- grounded follow-up over a decent saved session is often the strongest part of the product
- abstention behavior is materially better than generic answer-shaped filler
- broad discovery can produce a viable first literature scout for many scientific topics
- source metadata and trust buckets provide enough structure to reason about evidence quality

The same exercise also showed recurring problems that now look systemic rather than incidental:

- reranking and relevance filtering still allow generic or only loosely related papers to survive too high in the stack
- mixed policy-plus-literature questions are frequently forced into overly regulatory or overly generic routes
- exact-title known-item recovery can still return the right paper as only a weak or partial match
- regulatory retrieval is strongest for species and air-quality style primary sources, but weak for cultural resources, tribal consultation, archaeology, and other human-dimensions workflows
- cultural-resource and heritage queries are partly supported in literature search, but they are not first-class citizens in the regulatory or subject-resolution layers
- when model-backed relevance classification fails, the fallback path can flatten too many items into `weak_match`, which degrades the quality of ranking, verification, and follow-up grounding at the same time

The product gap is no longer only about environmental-science drift. It is now about domain coverage, ranking quality, and the honesty of the guidance contract when queries sit between disciplines, agencies, and evidence types.

## Cross-domain synthesis

### What worked well

1. Grounded follow-up over saved sessions.
   When the saved corpus was reasonably on-topic, follow-up answers were often substantially better than the initial `research` summary. PFAS and assisted-migration follow-ups were the clearest examples.

2. Abstention safety.
   When the saved evidence set was clearly wrong for the question, follow-up abstained instead of hallucinating a synthesis.

3. Broad scientific discovery in several natural-science domains.
   Peatlands, PFAS, salt marshes, urban heat, harmful algal blooms, shellfish aquaculture, and sea-level-risk-to-heritage queries all returned enough useful material to start a review.

4. Literature coverage for heritage and Indigenous stewardship topics.
   The tool can already retrieve relevant archaeology, heritage-risk, Indigenous knowledge, and co-management papers well enough to support broad literature discovery.

### What only partially worked

1. Mixed regulatory-plus-literature questions.
   Questions that asked both what the regulation is and how it relates to science often collapsed into regulatory-primary-source mode and never really synthesized the literature side.

2. Broad interdisciplinary discovery.
   Queries that combined ecology, public health, governance, and management tradeoffs often returned plausible but incomplete corpora with several weak matches.

3. Wildlife or island-ecosystem regulatory retrieval.
   The system can find related primary sources, but subject specificity is still loose enough that nearby but not actually responsive notices remain common in leads.

4. Cultural-resource regulatory retrieval.
   The system can retrieve some related Federal Register items, but it does not yet behave like a domain-aware historic-preservation or tribal-consultation assistant.

### What failed or underperformed

1. Reranking for broad or ambiguous prompts.
   Generic high-authority or weakly related documents can remain near the top when broad-query bridge bonuses and softened penalties outweigh the actual topic mismatch.

2. Known-item recovery quality.
   Exact-title queries can still return the right paper with thin metadata, `weak_match` relevance, or noisy off-topic neighbors.

3. Relevance fallback quality.
   When model-backed relevance classification fails, the fallback path marks everything as `weak_match`, which is safe in one sense but damages ranking, trust summaries, and follow-up evidence selection.

4. Coverage for cultural resources and anthropology-adjacent regulation.
   There is no strong subject-anchor layer yet for archaeology, Section 106 style workflows, NHPA-like process questions, tribal consultation, sacred sites, or broader cultural-landscape decision workflows.

## Concrete product conclusions

The tests suggest the current guided surface is best understood as:

- a good cross-domain literature scout
- a good grounded follow-up assistant when the saved corpus is already decent
- a fair regulatory primary-source retriever in some domains
- a weak mixed-domain policy-science synthesizer
- a weak known-item resolver relative to what users expect from exact-title lookups

The stress-test remediation (Phases 1–8) strengthened the schema, answer
validation, payload efficiency, citation repair, and regulatory routing
foundations. The next cycle should now focus on the remaining product-level
gaps: reranking quality, relevance-classification resilience, cultural-resource
routing, and follow-up trust gating.

## Verified implementation hotspots

The observed failures map to concrete code areas.

- `paper_chaser_mcp/agentic/ranking.py`
  Smart ranking computes `finalScore` as a sum of fused provider rank, query similarity, concept bonuses, provider bonuses, citation or recency priors, and broad-query bridge bonuses minus drift and facet penalties. In broad-query mode it intentionally scales down provider and facet penalties, which helps coverage but also lets generic bridge-like hits survive too easily.

- `paper_chaser_mcp/agentic/graphs.py`
  Topic classification and filtering combine deterministic thresholds with optional LLM classifications. The deterministic thresholds are strict at the extremes but permissive in the middle zone. This is a good architecture, but it means degraded or missing LLM classifications matter a lot.

- `paper_chaser_mcp/agentic/provider_openai.py`
  Relevance-batch failure currently falls back to `weak_match` for every paper. That is safer than overclaiming, but it produces low-information rankings and low-information follow-up corpora. *(Stress-test Phase 2 added deterministic refusal detection as a complementary safeguard, but the all-`weak_match` fallback itself is still a Workstream B concern.)*

- `paper_chaser_mcp/agentic/planner.py`
  Specificity estimation and heuristic overrides can promote some queries into `known_item` or `regulatory` too aggressively. That behavior is especially risky for mixed queries and exact-title requests without strong identifiers.

- `paper_chaser_mcp/agentic/graphs.py`
  Regulatory document ranking is still driven mainly by subject-token overlap, priority-term overlap, facet overlap, and document-form bonuses. That is not enough for cultural resources, tribal consultation, or process-heavy regulatory questions.

- `paper_chaser_mcp/agentic/graphs.py` and `paper_chaser_mcp/dispatch.py`
  Guided follow-up quality depends heavily on the selected evidence pool. If ranking or relevance classification is weak upstream, follow-up can still be coherent but rest on a thin or noisy evidence set. *(Stress-test Phase 4 ensures follow-up now only includes sources referenced in `selectedEvidenceIds`, which helps but does not fix upstream pool quality.)*

## Workstream A: Reranking Quality

### Objective

Reduce the number of generic, weakly related, or merely adjacent papers that survive near the top of broad or interdisciplinary result sets.

### Problems to solve

- broad-query mode weakens penalties enough that generic bridging papers can outrank directly responsive papers
- provider consensus and citation priors can reward authority more than responsiveness
- title and anchor mismatches are not punished hard enough when the query is broad but still semantically specific

### Plan

1. Split broad-query handling into two regimes.
   Distinguish genuinely exploratory synthesis prompts from semantically anchored prompts that merely contain multiple facets. Do not soften penalties equally for both.

2. Add an anchored-intent penalty.
   If a query contains strong subject anchors, exact entities, or required intervention terms, apply a larger penalty when title and body anchor coverage remain low, even in broad-query mode.

3. Reduce the influence of provider and citation priors when semantic fit is weak.
   Provider bonuses and citation bonuses should be multiplicative or gated by a minimum semantic relevance threshold, not simply additive.

4. Make bridge bonuses conditional on direct responsiveness.
   A paper should not receive bridge credit unless it also clears a stronger minimum title or anchor match threshold.

5. Add ranking diagnostics in eval artifacts.
   Persist the pre-filter top candidates and score breakdowns for failed cases so reranking regressions can be debugged from captured traces.

### Acceptance scenarios

- nitrate plus headwater-stream queries promote nitrate-macroinvertebrate papers above generic stream-ecology papers
- pesticide-mixture plus pollinator prompts prioritize pollinator-relevant toxicology over generic pesticide-mixture studies
- wildfire plus cultural-resource prompts prefer archaeology or heritage-fire papers over generic archaeology stewardship papers

### Likely files

- `paper_chaser_mcp/agentic/ranking.py`
- `paper_chaser_mcp/agentic/graphs.py`
- `paper_chaser_mcp/eval_curation.py`
- `tests/test_smart_tools.py`

## Workstream B: Relevance Classification Resilience

### Objective

Prevent relevance-model failures from collapsing whole candidate pools into undifferentiated `weak_match` buckets.

### Problems to solve

- model failures currently produce all-`weak_match` fallback classifications
- this fallback damages ranking, evidence selection, trust summaries, and follow-up grounding all at once

### Plan

1. Replace all-`weak_match` fallback with deterministic tiering.
   Use deterministic title, anchor, and facet coverage thresholds to produce a three-way fallback classification, not a single default label.

2. Surface fallback provenance explicitly.
   Preserve whether classification came from the LLM, deterministic fallback, cached inference, or hybrid arbitration.

3. Add a degraded-mode cap.
   When the relevance classifier is in fallback mode, cap the number of `on_topic` candidates and force a stricter evidence sufficiency check before synthesis.

4. Add regression tests for relevance-failure scenarios.
   Simulate model failure and ensure obviously off-topic papers do not remain `weak_match` simply because the classifier is down.

### Acceptance scenarios

- when relevance classification fails, an unrelated Medicare or fish-movement document is not retained as a merely weak match for a fire or tree-migration query
- known-item recovery still promotes the title-exact paper above generic adjacent records under degraded classification

### Likely files

- `paper_chaser_mcp/agentic/provider_openai.py`
- `paper_chaser_mcp/agentic/provider_langchain.py`
- `paper_chaser_mcp/agentic/graphs.py`
- `tests/test_agentic_providers_extra.py`
- `tests/test_smart_tools.py`

## Workstream C: Query Routing And Intent Splitting

### Objective

Reduce early misrouting into `regulatory` or `known_item` when the user is really asking for a hybrid synthesis or a broad review with one anchored entity.

### Problems to solve

- exact-title or citation-like strings can push queries into known-item mode too early
- mixed policy-plus-science prompts can get trapped in regulatory mode before the literature side is explored
- cultural-resource and anthropology-adjacent prompts currently lack dedicated routing heuristics

### Plan

1. Add a first-class `hybrid_policy_science` retrieval hypothesis.
   Use this when the user asks both what recent actions exist and how they relate to evidence.

2. Add a first-class `heritage_cultural_resources` secondary intent family.
   Detect archaeology, historic preservation, cultural landscape, sacred-site, tribal-consultation, and heritage-risk prompts and keep them from collapsing into generic environmental regulation.

3. Make exact-title recovery require stronger evidence before overriding discovery.
   An exact-looking title without DOI, URL, or other identifier should remain eligible for known-item handling but should not prevent a small breadth pass when ambiguity remains.

4. Revisit specificity heuristics.
   Queries with several concrete nouns or intervention terms should not necessarily be treated as low-specificity just because they are broad in phrasing.

### Acceptance scenarios

- PFAS regulation plus evidence-base prompts trigger blended regulatory-plus-literature retrieval
- harmful-algal-bloom regulation plus science prompts do not devolve into generic drinking-water literature
- title-exact archaeology or conservation papers route as high-confidence known-item only when the title match is actually strong

### Likely files

- `paper_chaser_mcp/agentic/planner.py`
- `paper_chaser_mcp/agentic/models.py`
- `paper_chaser_mcp/agentic/graphs.py`
- `tests/test_smart_tools.py`
- `tests/test_dispatch.py`

## Workstream D: Regulatory Coverage Beyond Species And Air

### Objective

Improve regulatory usefulness for cultural resources, tribal consultation, historic preservation, and other process-heavy domains that practitioners care about.

### Problems to solve

- current regulatory ranking is optimized for species and general Federal Register notice retrieval, not heritage-process workflows
- cultural-resource questions retrieve generic environmental notices or sanctuary rules rather than consultation, preservation, or archaeology-oriented materials
- the system does not yet model document-family expectations for these workflows

### Plan

1. Add regulatory subject-card resolution.
   For regulatory prompts, resolve the query into a subject card with requested process, subject area, agencies, and expected document families.

2. Add document-family detection for cultural-resource workflows.
   Support categories such as consultation guidance, preservation regulation, programmatic agreement context, archaeology guidance, tribal policy, and site-protection rules.

3. Expand ranking features for process-centric regulation.
   Add bonuses for document families and agencies associated with historic preservation, tribal consultation, cultural resources, and related review processes.

4. Add blended retrieval for regulatory-plus-practice prompts.
   When the user asks what actions exist and how they affect environmental decision-making, return both authoritative primary sources and relevant practice or scholarship anchors.

5. Add direct eval coverage for cultural-resource regulation.
   Include prompts on archaeology, heritage risk, sacred sites, tribal consultation, and historic preservation in the benchmark suite.

### Acceptance scenarios

- cultural-resource regulation queries surface directly relevant regulatory or guidance materials rather than generic sanctuary or permitting notices
- archaeology plus tribal-consultation prompts produce more than just adjacent Federal Register items
- cultural-resource primary sources are clearly separated from literature or commentary leads

### Likely files

- `paper_chaser_mcp/agentic/graphs.py`
- `paper_chaser_mcp/dispatch.py`
- `paper_chaser_mcp/compat.py`
- `tests/test_dispatch.py`
- `tests/fixtures/ux_prompt_corpus.json`

## Workstream E: Known-Item Recovery Quality

### Objective

Make exact-title and near-exact-title retrieval behave like a high-confidence reference resolver instead of a partial discovery workflow.

### Problems to solve

- exact-title hits can still be labeled `weak_match`
- provider metadata for title-exact papers can remain thin
- off-topic neighbors remain too prominent in small known-item result sets

### Plan

1. Add exact-title confidence logic.
   If the normalized title is a very strong match and author or year metadata align, promote the match aggressively and demote generic neighbors.

2. Add a known-item metadata completion pass.
   After a likely exact match is found, enrich it before final classification so DOI, venue, canonical URL, and provider fields are more complete.

3. Make partial known-item responses explicit.
   Distinguish `resolved_exact`, `resolved_probable`, and `needs_disambiguation` style states in execution provenance.

### Acceptance scenarios

- exact-title urban-heat and assisted-migration queries return the intended paper as a strong match with minimal noisy neighbors
- known-item search no longer reports title-exact hits as merely partial without explanation

### Likely files

- `paper_chaser_mcp/agentic/planner.py`
- `paper_chaser_mcp/agentic/graphs.py`
- `paper_chaser_mcp/citation_repair.py` *(Stress-test Phase 5 tightened year
  penalty, added title-similarity length penalty, reduced upstream confidence
  bonus, and caps confidence on title conflicts. These changes help known-item
  recovery by improving candidate ranking quality.)*
- `tests/test_dispatch.py`

## Workstream F: Follow-Up Trust And Evidence Selection

### Objective

Keep the strong follow-up experience while making it less vulnerable to noisy initial corpora.

### Problems to solve

- good follow-up can still inherit ranking noise from the saved session
- selected evidence ids can look authoritative even when the upstream pool was degraded or fallback-classified

### Plan

1. Add evidence-pool quality checks before synthesis.
   If too much of the saved evidence pool is weak-match, force a narrower recovery or abstain.

2. Weight evidence selection by explanation quality.
   Prefer sources that are both on-topic and have explicit rationale for inclusion.

3. Add degraded-follow-up provenance.
   Preserve whether the answer relied on fallback classification, sparse evidence, or metadata-only rescue.

### Acceptance scenarios

- follow-up remains strong on PFAS and assisted migration positive controls
- follow-up becomes more conservative on noisy broad corpora instead of producing polished but thin answers

### Likely files

- `paper_chaser_mcp/agentic/graphs.py`
- `paper_chaser_mcp/dispatch.py`
- `tests/test_dispatch.py`
- `tests/test_smart_tools.py`

## Workstream G: Cross-Domain Eval Program

### Objective

Turn this review into a durable benchmark pack rather than a one-off manual exercise.

### Benchmark slices to add

1. Natural science.
   Peatlands, PFAS, HABs, urban heat, salt marshes, invasive species, pollinators.

2. Human dimensions.
   Environmental justice, Indigenous stewardship, co-management, rural drinking water risk.

3. Heritage and archaeology.
   Sea-level risk to archaeology, wildfire effects on archaeological sites, cultural-resource stewardship.

4. Regulation.
   Species-critical-habitat, PFAS drinking water, wildfire smoke and air management, cultural resources and consultation.

5. Known-item recovery.
   Exact-title and near-title retrieval across science and heritage domains.

### Eval expectations

- each slice should include positive controls, ambiguous cases, mixed-mode prompts, and explicit failure-expected prompts
- captured artifacts should preserve ranking diagnostics, fallback provenance, and routing summaries
- benchmark prompts should be structured so reranking, routing, and follow-up can be judged independently

### Likely files

- `tests/fixtures/ux_prompt_corpus.json`
- `tests/test_prompt_corpus.py`
- `tests/test_provider_benchmark_corpus.py`
- `paper_chaser_mcp/eval_curation.py`
- `docs/agent-handoff.md`

## Suggested sequencing

0. ~~Schema integrity, answer validation, payload efficiency, citation repair,
   regulatory routing foundations~~ — **completed** in stress-test remediation
   Phases 1–8 (117 new tests, 962 total).
1. Reranking quality and relevance fallback resilience
2. Query-routing and hybrid-intent fixes
3. Known-item recovery improvements
4. Regulatory expansion for cultural resources and process workflows
5. Follow-up evidence-pool trust gating
6. Cross-domain eval pack and external rerun

## Exit criteria for the next review

The next external rerun should aim for the following:

- broad cross-domain discovery feels reliable enough for first-pass review work
- known-item recovery behaves like a true resolver, not a partial scout
- mixed policy-plus-science prompts consistently run blended retrieval when needed
- cultural-resource and archaeology prompts feel like supported domains rather than accidental spillover from environmental search
- reranking diagnostics clearly explain why generic or adjacent papers were demoted
- follow-up answers remain strong, but only when the saved evidence set actually deserves confidence
