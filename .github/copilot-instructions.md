# GitHub Copilot Instructions

This repository is optimized for the **GitHub cloud coding agent** as the
primary development engine. Treat these instructions as the durable GitHub-side
counterpart to the in-app MCP onboarding resource and planning prompt.

## Start Here

Before changing code, read these files in order:

1. `README.md` for the public MCP surface, validation commands, and workflow notes.
2. `docs/golden-paths.md` for the default guided workflows and expert fallbacks.
3. `docs/guided-reset-migration-note.md` for the breaking guided-default reset.
4. `docs/agent-handoff.md` for current repo status, validation baseline, and next steps.

For release, publishing, or version-prep work, also read
`docs/release-publishing-plan.md` before changing workflows, tags, or package
metadata.

## Development Priorities

- Keep the **default tool surface small and obvious** for low-context agents.
- Treat the **guided tools as the public contract of record**:
  `research`, `follow_up_research`, `resolve_reference`, `inspect_source`, and
  `get_runtime_status`.
- Use **expert tools only intentionally** when there is a concrete control gap
  or the workflow is explicitly running with
  `PAPER_CHASER_TOOL_PROFILE=expert`.
- Make **abstention safe and explicit**. Weak evidence should produce
  `abstained`, `insufficient_evidence`, or `needs_disambiguation`, not
  answer-shaped filler.
- Keep **`leads` separate from `evidence`**. Weak, filtered, or off-topic items
  may be useful for auditability, but they must not quietly re-enter grounded
  evidence or trusted summaries. Legacy `unverifiedLeads` and
  `verifiedFindings` should be treated as compatibility views, not the primary
  contract.
- Keep **runtime truth internally consistent**. The top-level runtime summary,
  active/disabled provider sets, and smart-provider fields must agree with the
  detailed provider rows.
- **Minimize agent round trips**: every tool should give agents enough context
  in a single response to take the obvious next step. When a task requires 3+
  calls where 1-2 should suffice, that is a UX bug worth filing an issue for.
- **Prioritize intuitiveness**: tool names, parameter names, and response
  fields should be self-explanatory. Agents should not need to read the full
  docs to use a tool correctly on the first try.
- **Eliminate dead ends**: every response should include clear next-step
  guidance (`nextActions`, `clarification`, provenance hints, or direct-read
  recommendations).
- Preserve the workflow hierarchy:
  - `research` for discovery, literature review, known-item recovery, citation
    repair, and regulatory routing.
  - `follow_up_research` for one grounded follow-up over a saved
    `searchSessionId`.
  - `resolve_reference` for DOI/arXiv/URL/citation/reference cleanup.
  - `inspect_source` for source-level trust and provenance checks.
  - `get_runtime_status` for environment/profile/provider sanity checks.
  - Expert fallback only when needed:
    `search_papers_smart`, `ask_result_set`, `map_research_landscape`,
    `expand_research_graph`, `search_papers`, `search_papers_bulk`,
    provider-specific families, and direct regulatory primary-source tools.
- Prefer progressive disclosure over adding overlapping entry points.
- Keep provider complexity explainable but mostly hidden behind the guided and
  brokered paths.
- Keep continuation semantics explicit: `search_papers_bulk` is not a generic
  "next page" for guided discovery, and should be described as an expert pivot
  whenever it changes provider or retrieval semantics.
- Keep reusable result-set handles explicit: `searchSessionId` should always be
  enough to continue grounded QA, source inspection, clustering, or graph
  expansion without hidden session state.
- Keep provider-specific tool contracts honest: CORE, SerpApi, arXiv,
  OpenAlex, ScholarAPI, and regulatory tools should not casually advertise
  filters or guarantees they do not honor.

## Durable Planning And Progress

- When you change workflow guidance, update the durable docs in the repo, not
  just inline code strings.
- Keep these surfaces aligned:
  - `README.md`
  - `paper_chaser_mcp/server.py` (`SERVER_INSTRUCTIONS`,
    `AGENT_WORKFLOW_GUIDE`, `plan_paper_chaser_search`)
  - `docs/golden-paths.md`
  - `docs/guided-reset-migration-note.md`
  - `docs/agent-handoff.md`
- Keep the checked-in agentic workflow guidance aligned too:
  - `.github/workflows/test-paper-chaser.md`
  - `.github/workflows/test-paper-chaser.lock.yml`
- The workflow supports `smoke`, `comprehensive`, and `feature_probe` review
  modes plus a `tool_profile` input (`guided` by default, `expert` when you
  intentionally want raw/provider-specific coverage).
- If you edit `.github/workflows/test-paper-chaser.md`, rerun:

```bash
gh aw compile test-paper-chaser --dir .github/workflows
```

- Record meaningful follow-up work in `docs/agent-handoff.md` or
  `docs/golden-paths.md` so future agents inherit context instead of
  rediscovering it.

## Validation Expectations

Install development extras first:

```bash
pip install -e ".[all]"
```

Then run:

```bash
python -m pip check
pre-commit run --all-files
python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=87
python -m mypy --config-file pyproject.toml
python -m ruff check .
python -m bandit -c pyproject.toml -r paper_chaser_mcp
python -m build
python -m pip_audit . --progress-spinner off
```

If you prefer the hook entrypoint for the heavier checks, run
`pre-commit run --hook-stage manual --all-files` to execute the manual-stage
`pip check`, coverage, build, and audit hooks.

If you touch Azure IaC, deployment docs, the Dockerfile, the APIM policy, or
`.github/workflows/deploy-azure.yml`, also run:

```bash
python scripts/validate_psrule_azure.py
python scripts/validate_deployment.py --skip-docker
```

Use
`python scripts/validate_deployment.py --require-az --require-docker --image-tag paper-chaser-mcp:ci-validate`
when you need parity with the full `Deploy Azure` workflow.

Run focused tests early when making localized changes, then rerun the full stack
before finishing.

## Change Style

- Prefer the smallest complete change over broad refactors.
- Add or update tests when you change behavior or durable agent guidance.
- Keep README, onboarding resources, workflow docs, and tests synchronized when
  the MCP tool contract changes.
- Do not remove durable planning/progress/future-work docs; improve them instead.
