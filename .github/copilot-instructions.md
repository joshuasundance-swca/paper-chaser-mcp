# GitHub Copilot Instructions

This repository is optimized for the **GitHub cloud coding agent** as the
primary development engine. Treat these instructions as the durable GitHub-side
counterpart to the in-app MCP onboarding resource and planning prompt.

## Start Here

Before changing code, read these files in order:

1. `README.md` for the public MCP surface and validation commands.
2. `docs/golden-paths.md` for the primary user workflows, tool-routing defaults,
   and success signals.
3. `docs/agent-handoff.md` for current repo status, progress snapshots, known
   hotspots, and suggested next steps.

## Development Priorities

- Keep the **default tool surface small and obvious** for low-context agents.
- Keep the smart layer **additive, not replacement**: raw tools remain the
  stable contract of record, while smart tools (`search_papers_smart`,
  `ask_result_set`, `map_research_landscape`, `expand_research_graph`) should
  reduce round trips for concept-level research.
- **Minimize agent round trips**: every tool should give agents enough context
  in a single response to take the obvious next step. When a task requires 3+
  tool calls where 1-2 should suffice, that is a UX bug worth filing an issue
  for.
- **Prioritize intuitiveness**: tool names, parameter names, and response
  fields should be self-explanatory. Agents should not need to read the full
  docs to use a tool correctly on the first try.
- **Eliminate dead ends**: every response should include clear next-step
  guidance (e.g., `brokerMetadata.nextStepHint`, expansion IDs, pagination
  cursors). An agent that receives a response with no obvious follow-up action
  has hit a UX defect.
- Preserve the workflow hierarchy:
  - `search_papers_smart` for concept discovery, reusable result sets, and grounded follow-up
  - `search_papers` for quick literature discovery
  - `search_papers_bulk` for exhaustive or paginated retrieval
  - `resolve_citation` for broken references, incomplete bibliography lines, and almost-right citations
  - `search_papers_match` / `get_paper_details` for known-item lookup
  - `get_paper_citations` / `get_paper_references` for citation chasing
  - `search_authors` → `get_author_info` → `get_author_papers` for author pivots
  - `search_snippets` only as a recovery tool
- Prefer progressive disclosure over adding overlapping entry points.
- Keep provider complexity explainable but mostly hidden behind the brokered path.
- Keep OpenAlex explicit: use the dedicated `*_openalex` tools when OpenAlex
  citation, author, DOI/ID, or cursor semantics are required instead of trying
  to squeeze OpenAlex into the default brokered workflow.
- Keep continuation semantics explicit: `search_papers_bulk` is not a generic
  "next page" for brokered results, and should be described as a pivot whenever
  it changes provider or filter semantics.
- Keep reusable result-set handles explicit: `searchSessionId` should always be
  enough to continue grounded QA, clustering, or graph expansion without hidden
  session state.
- Keep provider-specific tool contracts honest: CORE, SerpApi, arXiv, and
  OpenAlex surfaces should not casually advertise filters they do not honor.

## Durable Planning and Progress

- When you change workflow guidance, update the durable docs in the repo, not
  just inline code strings.
- Keep these three surfaces aligned:
  - `paper_chaser_mcp/server.py` (`SERVER_INSTRUCTIONS`,
    `AGENT_WORKFLOW_GUIDE`, `plan_paper_chaser_search`)
  - `docs/golden-paths.md`
  - `docs/agent-handoff.md`
- Keep the checked-in agentic workflow guidance aligned too when the
  `workflow_dispatch` review modes (`smoke`, `comprehensive`, `feature_probe`)
  or focus-prompt contract changes.
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
python -m pytest --cov=paper_chaser_mcp --cov-report=term-missing --cov-fail-under=85
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

If you edit `.github/workflows/test-paper-chaser.md`, also rerun
`gh aw compile test-paper-chaser --dir .github/workflows` so the checked-in
`.lock.yml` stays synchronized with the editable Markdown workflow source, then
run the normal validation stack so pre-commit can normalize the generated file.

Run focused tests early when making localized changes, then rerun the full stack
before finishing.

## Change Style

- Prefer the smallest complete change over broad refactors.
- Add or update tests when you change behavior or durable agent guidance.
- Keep README, onboarding resources, and tests synchronized when the MCP tool
  contract changes.
- Do not remove durable planning/progress/future-work docs; improve them instead.
