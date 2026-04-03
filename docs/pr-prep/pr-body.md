# Guided-First Contract Reset, Trust Gating, and Regulatory Routing

## Summary

This PR resets Paper Chaser's default MCP experience around a guided-first public contract. Guided mode now advertises five tools by default: `research`, `follow_up_research`, `resolve_reference`, `inspect_source`, and `get_runtime_status`.

Under the hood, the branch productizes those flows as wrappers over the existing retrieval and smart-runtime substrate, tightens response semantics around trust gating and abstention, adds a dedicated regulatory primary-source path, and realigns docs, prompts, manifests, workflow assets, and tests around that contract.

## Why

The previous default surface was too easy for low-context agents to misuse. There were too many starting points, too much provider nuance exposed too early, and not enough hard separation between verified findings and plausible but weak evidence.

This PR makes the safe path more obvious and more enforceable:

- guided discovery starts from `research`
- grounded follow-up is explicit and session-based
- weak evidence can abstain instead of producing answer-shaped filler
- off-topic or filtered material stays in `unverifiedLeads` instead of trusted summaries
- runtime/profile truth is directly inspectable
- expert depth remains available, but only when intentionally selected

## Main changes

- switch default MCP advertisement to the five guided tools
- add profile-aware tool visibility and profile-filtered tool definitions
- implement guided wrappers for research, follow-up, reference resolution, source inspection, and runtime status
- strengthen trust-gated outputs with `verifiedFindings`, `unverifiedLeads`, `trustSummary`, coverage/failure metadata, and explicit next actions
- add answer gating for follow-up via `answerStatus=answered|abstained|insufficient_evidence`
- add a dedicated regulatory primary-source workflow and timeline-oriented output behavior
- normalize retrieval metadata so provider results can participate in trust gating consistently
- update docs, workflow guidance, manifests, and samples to codify the guided-versus-expert split more precisely

## Risks

- breaking default `list_tools` discovery for clients that expect old smart/raw tools
- wrapper drift between guided contracts and expert internals over time
- heuristic routing or threshold tuning causing over-abstention or over-routing
- future manifest/docs/workflow drift if the contract changes again without synchronized updates

## Notes for reviewers

- guided mode now genuinely filters advertised tools; this is a real discovery-contract change, not just a docs update
- guided docs and workflow instructions now consistently use `unverifiedLeads` for the public contract
- expert visibility language is now qualified by enabled providers and `PAPER_CHASER_HIDE_DISABLED_TOOLS`
- the checked-in Microsoft plugin sample is guided-first by default; expert packaging is still documented as an intentional switch
