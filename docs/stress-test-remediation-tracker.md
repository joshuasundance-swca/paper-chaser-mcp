# Stress-Test Remediation Tracker

This file is kept as a compatibility companion to
`docs/stress-test-remediation-plan.md`, which is the canonical durable
stress-test remediation plan referenced by the original implementation plan.

This document is the durable tracker for the stress-test remediation program that hardened the guided Paper Chaser contract across payload efficiency, citation confidence, degradation behavior, eval coverage, and supporting docs.

Use this alongside:

- `docs/agent-handoff.md` for current repo state and validation commands
- `docs/environmental-science-remediation-plan.md` for the environmental-science-specific follow-on work
- `docs/cross-domain-remediation-plan.md` for broader product workstreams that build on these foundations

## Status summary

- Phase 4.1 compact abstention responses: completed as a safe subset
  Outcome: guided `research` abstention responses and guided `follow_up_research` abstained or insufficient-evidence responses now suppress heavyweight source and legacy payload sections while preserving explicit recovery metadata.
- Phase 4.2 reduce duplicate serialization and legacy field bloat: partially complete
  Outcome: empty legacy compatibility fields are no longer serialized unnecessarily, and compact abstention paths now omit duplicate legacy/source sections. Full removal of compatibility views on successful responses remains intentionally deferred for compatibility.
- Phase 5.3 improve `bestMatch` confidence: completed
  Outcome: very high title-similarity fuzzy or citation-ranked matches are now promoted more aggressively so correct top-ranked papers are less likely to fall into alternatives-only responses.
- Phase 7.2 live integration tests with graceful skips: completed
  Outcome: live guided integration tests now exist and skip cleanly when no smart-provider key is configured.
- Phase 7.3 deterministic degradation integration coverage: completed
  Outcome: dedicated integration tests cover guided research and runtime-status behavior when the configured smart provider degrades to deterministic fallback.
- Phase 7.4 expand eval corpus and fixtures: completed as an in-repo subset
  Outcome: the stress scenarios now exist in the eval seed fixtures and the provider benchmark corpus. Full external replay data is still outside the repo.
- Phase 7.5 payload-size regression tests: completed
  Outcome: regression tests now enforce compact follow-up insufficient-evidence payloads and compact research abstention payloads.
- Phase 8 remaining docs: completed for the durable tracker and environmental-science plan update
  Outcome: this tracker is now checked in, and the environmental-science remediation plan reflects the current state of the landed work.

## Compatibility notes

- The repo still preserves compatibility views such as `verifiedFindings`, `sources`, `unverifiedLeads`, and `coverage` on non-abstention success paths.
- That duplication is still intentional until downstream consumers are ready to rely only on `evidence`, `leads`, `routingSummary`, and `coverageSummary`.
- The current payload-efficiency work therefore focuses on the highest-value safe subset: abstention and insufficient-evidence paths, where repeated source reserialization costs the most and helps the least.

## Validation anchors

Focused regression coverage for this tracker lives in:

- `tests/test_payload_efficiency.py`
- `tests/test_citation_repair_fixes.py`
- `tests/test_degradation_integration.py`
- `tests/test_integration_live.py`
- `tests/test_eval_curation.py`
- `tests/test_provider_benchmark_corpus.py`

Recommended focused command:

```bash
python -m pytest tests/test_payload_efficiency.py tests/test_citation_repair_fixes.py tests/test_degradation_integration.py tests/test_integration_live.py tests/test_eval_curation.py tests/test_provider_benchmark_corpus.py -q
```

## Remaining follow-on work

1. Decide when successful guided responses can stop serializing legacy compatibility views by default.
2. Strengthen entity-grounded regulatory and species routing beyond the schema and eval foundations landed here.
3. Expand live stress-scenario replay and trace promotion so the in-repo fixtures stay paired with fresh observed failures.
