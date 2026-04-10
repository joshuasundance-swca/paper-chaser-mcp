# Environmental-Science Remediation Plan

This document turns the latest environmental-science evaluation into an implementation plan for the guided Paper Chaser workflow. It is intentionally LLM-first: the plan assumes the server should use the model for richer routing, synthesis, specificity judgment, and provenance explanation, while keeping deterministic logic as a guardrail rather than the primary experience.

## Why this plan exists

The latest environmental-science pass showed a recognizable pattern:

- broad discovery is now useful enough to scout literature, especially for interdisciplinary topics
- regulatory routing is safer than before, but still not specific enough for practitioner workflows
- grounded follow-up is the weakest step because the system can return polished answers after degrading into thin deterministic salvage
- source inspection helps, but it does not yet explain weak relevance or weak regulatory specificity clearly enough

The practical gap is no longer just retrieval drift. The larger product problem is that the public guided contract promises trust-gated follow-up and subject-anchored regulatory behavior, while several real environmental-science cases still degrade into weak synthesis, weak specificity, or overconfident source framing.

## Outcome targets

The target for the next remediation cycle is not just fewer failures. It is a different quality profile.

- `follow_up_research` should stop answering comparison or synthesis questions when only deterministic salvage is available.
- species and regulatory discovery should resolve to a species-specific dossier chain before they are allowed to claim `on_topic` primary-source support.
- `inspect_source` should explain *why* a source is strong, weak, or off-target instead of only labeling it.
- environmental-science evaluation should become a first-class regression harness, not a one-off review.

Suggested success thresholds for the next external rerun:

- broad discovery: keep at `8/10` or higher
- grounded follow-up: raise from `4/10` to at least `7/10`
- regulatory or primary-source discovery: raise from `5/10` to at least `7/10`
- species-plan or ECOS-style retrieval: raise from `3/10` to at least `7/10`
- provenance UX: raise from `6.5/10` to at least `8/10`

## Design principles

1. Use LLM judgment where the task is semantic.
   Deterministic heuristics should remain filters, validation checks, and fail-safe fallbacks. They should not be the default synthesis engine for comparison, policy-chain, or intervention-analysis questions.

2. Make fallback visibly lower trust.
   If the model-backed synthesis layer is unavailable or cannot support the question, the server should abstain or switch into an explicit metadata-summary mode. It should not produce answer-shaped text that looks equivalent to a grounded synthesis.

3. Treat regulatory and species workflows as entity-resolution problems, not keyword search problems.
   A species-planning query needs an identified subject, a dossier chain, and source-level specificity checks before it can claim grounded support.

4. Keep contract changes additive where possible.
   The guided public contract should preserve `answerStatus`, `answerability`, `evidence`, `leads`, and `executionProvenance`, while adding clearer confidence and rationale fields instead of breaking existing clients.

5. Evaluate the internal LLM roles directly.
   The repo already has the right philosophy in `docs/llm-evaluation-program-plan.md`: planner, synthesizer, trust gating, and provenance reasoning should be evaluated as distinct roles and then rechecked end to end.

## Verified code hotspots

These are the main implementation loci behind the observed failures.

- `paper_chaser_mcp/dispatch.py`
  Session-introspection follow-up answers are generated in `_answer_follow_up_from_session_state(...)`. That path can answer from saved metadata without the stronger degradation semantics used in model-backed follow-up.
- `paper_chaser_mcp/agentic/graphs.py`
  `ask_result_set` already downgrades deterministic follow-up answers with `degradationReason="deterministic_synthesis_fallback"`, but selected evidence ids can still make weak answers look more grounded than they feel.
- `paper_chaser_mcp/dispatch.py`
  `_direct_read_recommendations(...)` is provider-oriented and not confidence-aware.
- `paper_chaser_mcp/dispatch.py` and `paper_chaser_mcp/agentic/graphs.py`
  Topical relevance and trust summaries expose labels, counts, and status buckets, but not enough explanation for why a weak regulatory hit was retained or demoted.
- `paper_chaser_mcp/agentic/graphs.py`
  Regulatory routing already tries ECOS, Federal Register, and GovInfo with subject anchoring, but ECOS hits and accepted regulatory documents are promoted quickly once token-overlap checks pass.
- `paper_chaser_mcp/agentic/graphs.py`
  `_regulatory_document_matches_subject(...)` uses token overlap and priority overlap, which is safer than raw keyword retrieval but still too weak for species-specific planning and dossier-grade specificity.

## Workstream A: Follow-up Synthesis Integrity

### Objective

Fix the biggest current product failure: polished but weak follow-up answers.

### Problems to solve

- deterministic salvage can still look like a valid grounded answer for comparison-style questions
- session-introspection answers are useful for metadata questions, but they currently share the same surface as genuine synthesis answers
- selected evidence ids are not enough to prove that the answer actually responded to the user’s question

### Plan

1. Add an explicit follow-up answer mode decision before synthesis.
   The server should classify the follow-up into modes such as `metadata`, `relevance_triage`, `comparison`, `mechanism_summary`, `regulatory_chain`, and `intervention_tradeoff`.

2. Restrict deterministic fallback to metadata-safe modes.
   Deterministic or saved-session introspection should be allowed to answer questions like provider coverage, evidence gaps, saved source overview, and explicit source triage. It should not answer intervention comparisons, method comparisons, policy-chain questions, or question-specific synthesis asks.

3. Require an evidence use plan from the LLM before answering comparison-style questions.
   The synthesizer should first emit a compact structured plan such as:
   - what question subtype is being answered
   - which evidence ids are directly responsive
   - which parts are unsupported
   - whether retrieval is sufficient to compare

4. Add a synthesis sufficiency gate.
   If the plan shows fewer than the required responsive sources, or if evidence only supports a ranked recap rather than a comparison, the server should return `insufficient_evidence` instead of a recap disguised as synthesis.

5. Separate `metadata_answered` from `grounded_synthesis_answered` in provenance.
   Keep `answerStatus` backward-compatible, but add additive fields under `executionProvenance` and `confidenceSignals` so clients can tell whether the answer came from model-backed synthesis, saved-session metadata, or deterministic salvage.

6. Add a targeted recovery path before abstaining.
   For comparison questions over a thin saved set, allow one bounded LLM-guided retrieval refinement pass that asks for missing evidence shapes rather than broadening blindly.

### Acceptance scenarios

- eDNA follow-up answers the actual method-comparison question or abstains explicitly
- wildfire smoke follow-up stops returning source-list prose as if it were synthesis
- PFAS remediation follow-up compares adsorption, membranes, and destruction only when evidence actually supports that comparison
- mangrove blue carbon remains a positive control

### Likely files

- `paper_chaser_mcp/dispatch.py`
- `paper_chaser_mcp/agentic/graphs.py`
- `paper_chaser_mcp/agentic/provider_openai.py`
- `paper_chaser_mcp/agentic/provider_langchain.py`
- `tests/test_dispatch.py`
- `tests/test_smart_tools.py`

## Workstream B: Species And Regulatory Entity Grounding

### Objective

Make regulatory and ECOS-style retrieval practitioner-grade by grounding on the right entity and the right document chain.

### Problems to solve

- ECOS retrieval is present, but not yet strong enough as a species dossier resolver
- generic Federal Register notices can still look acceptable for a species-specific query if subject-term overlap is good enough
- the system does not yet distinguish well between a regulatory history request, a current-codified-text request, a species dossier request, and a guidance lookup

### Plan

1. Add a first-class regulatory intent split.
   The planner should distinguish:
   - `current_cfr_text`
   - `rulemaking_history`
   - `species_dossier`
   - `guidance_lookup`
   - `hybrid_regulatory_plus_literature`

2. Add an LLM-backed entity resolver for species and subject anchors.
   Before ranking ECOS, Federal Register, or GovInfo hits, resolve the query into a subject card:
   - common name
   - scientific name when available
   - agency or authority context
   - required document family, such as recovery plan, critical habitat, listing rule, or consultation guidance

3. Promote species dossiers as chains, not as isolated hits.
   For species workflows, the system should prefer an internally linked chain:
   - ECOS species profile
   - ECOS dossier documents
   - matching Federal Register or GovInfo items tied to the same subject

4. Add a stricter specificity scorer for regulatory documents.
   Replace or augment pure token-overlap gating with an LLM judgment that answers:
   - is this document specifically about the anchored subject
   - is it only context-adjacent
   - does it satisfy the requested document family
   - if weak, why

5. Make `on_topic` harder to earn for species-regulatory results.
   A generic wildlife notice should not be `on_topic` for desert tortoise or condor planning unless it explicitly addresses that species or the exact requested CFR chain.

6. Add subject-chain evidence gaps.
   When ECOS finds a profile but not dossier documents, or when Federal Register hits lack species-specific support, expose that gap directly instead of treating the primary-source lane as broadly successful.

### Acceptance scenarios

- desert tortoise queries do not treat generic endangered-species notices as grounded species support
- condor and northern long-eared bat remain positive regulatory controls
- PFAS drinking-water regulation cleanly separates primary federal materials from literature and unrelated items
- Section 106 plus offshore wind surfaces process or guidance materials instead of broad offshore-wind literature drift

### Likely files

- `paper_chaser_mcp/agentic/planner.py`
- `paper_chaser_mcp/agentic/graphs.py`
- `paper_chaser_mcp/dispatch.py`
- `paper_chaser_mcp/compat.py`
- `tests/test_dispatch.py`
- `tests/fixtures/ux_prompt_corpus.json`

## Workstream C: Confidence And Provenance UX

### Objective

Make weak evidence look weak for the right reasons.

### Problems to solve

- `inspect_source` is useful when the source is already good, but it does not explain the weakness of borderline hits well enough
- `trustSummary` counts sources but does not explain strength versus weakness
- `directReadRecommendations` are not calibrated to source quality

### Plan

1. Surface rationale for topical relevance.
   When the LLM or hybrid classifier marks a source as `weak_match` or `off_topic`, preserve a concise rationale and return it from `inspect_source`.

2. Add additive confidence fields instead of replacing `answerability`.
   Keep the current public enums, but add fields such as:
   - `confidenceSignals.evidenceQualityProfile`
   - `confidenceSignals.synthesisMode`
   - `confidenceSignals.trustRevisionReason`
   - `source.whyClassifiedAsWeakMatch`

3. Differentiate authoritative but weakly relevant from authoritative and directly responsive.
   A Federal Register or GovInfo source can be a real primary source and still be a weak answer to the user’s actual question.

4. Make direct-read recommendations quality-aware.
   Recommendations should say not just how to open the source, but why the user is being sent there and how much to trust it.

5. Improve trust summaries.
   Summaries should distinguish strong on-topic verified evidence from weak-match authority records and off-target leads.

### Acceptance scenarios

- `inspect_source` for a weak PFAS-regulation hit explains why it is only partially relevant
- `inspect_source` for a desert-tortoise false positive explicitly says the source is authoritative but not species-specific enough
- direct-read recommendations for good ECOS or GovInfo hits remain concise and useful

### Likely files

- `paper_chaser_mcp/dispatch.py`
- `paper_chaser_mcp/agentic/models.py`
- `paper_chaser_mcp/agentic/graphs.py`
- `tests/test_dispatch.py`

## Workstream D: Environmental-Science Eval Program

### Objective

Turn this review into an enduring LLM-evaluation slice that can steer iteration.

### Plan

1. Create an environmental-science benchmark pack.
   Add curated prompts and expected behaviors for:
   - literature discovery
   - regulatory discovery
   - grounded follow-up
   - source inspection
   - species dossier and ECOS-style retrieval

2. Use the trace-capture funnel for real failure mining.
   Capture guided `research`, `follow_up_research`, and `inspect_source` traces for environmental-science prompts, push them through review, and promote validated failures into durable eval rows.

3. Add role-level LLM judge rubrics.
   For the internal model, score separately:
   - planner specificity and route choice
   - follow-up responsiveness to the actual question
   - evidence sufficiency judgment
   - provenance honesty

4. Add regression slices for each failure family.
   Minimum tracked families:
   - thin deterministic salvage
   - species-specificity failure
   - regulatory primary-source mixing
   - archaeology or cultural-resource crossover drift
   - management-intervention comparison without enough evidence

5. Create an autopilot profile for this domain slice.
   Reuse the existing `run_eval_autopilot.py` and trace-promotion flow rather than inventing a separate harness.

### Acceptance artifacts

- new fixtures in `tests/fixtures/evals/`
- expanded prompt coverage in `tests/fixtures/ux_prompt_corpus.json`
- reviewable captured traces and promoted rows for environmental-science families
- focused test runs for guided contracts plus domain fixtures

## Phase plan

### Phase 0: Baseline capture and taxonomy

- convert the reviewed environmental-science prompts into repo fixtures
- define failure family labels and reviewer rubric
- record current scores as the baseline

### Phase 1: Follow-up safety before expressiveness

- stop deterministic salvage from answering synthesis questions
- add answer-mode classification and synthesis sufficiency gating
- tighten session-introspection behavior so it only answers metadata-safe asks

### Phase 2: Entity-grounded regulatory and species retrieval

- add subject-card resolution and species dossier routing
- strengthen per-document specificity judgment
- update regulatory acceptance tests and benchmark fixtures

### Phase 3: Provenance and trust explanation

- expose rationale fields in `inspect_source`
- add confidence-aware recommendation text and trust summaries
- ensure public docs explain these new signals

### Phase 4: LLM-eval hardening and score-driven iteration

- run the environmental-science pack repeatedly through the capture and promotion funnel
- use role-level judge scores to tune prompts, thresholds, and routing
- keep positive controls such as mangrove follow-up and broad interdisciplinary discovery in the pack to avoid overfitting only to failures

## Testing strategy

Run focused tests first:

- `pytest tests/test_dispatch.py tests/test_smart_tools.py -q`
- `pytest tests/test_prompt_corpus.py tests/test_provider_benchmark_corpus.py -q`

Then run the broader guided and eval stack:

- `pytest tests/test_agentic_workflow.py tests/test_eval_curation.py tests/test_eval_trace_promotion.py -q`

When behavior changes are stable, rerun the standard repo validation flow described in `docs/agent-handoff.md`.

## Product and documentation follow-through

If implementation changes the user-visible semantics of follow-up trust, provenance, or regulatory guidance, keep these surfaces aligned:

- `README.md`
- `docs/golden-paths.md`
- `docs/guided-smart-robustness.md`
- `docs/agent-handoff.md`
- tool descriptions in `paper_chaser_mcp/tool_specs/`

## Near-term recommendation

Do not start with a broad retrieval refactor. The fastest path to user-visible improvement is:

1. make weak follow-up synthesis abstain more honestly
2. add species-specific entity grounding for regulatory workflows
3. expose clearer trust rationale in `inspect_source`
4. institutionalize this environmental-science slice in the eval pipeline

That sequence addresses the biggest current quality gap without risking the strongest part of the product, which is already broad discovery.
