"""Guided findings/leads extraction (Phase 3).

Extracted from :mod:`paper_chaser_mcp.dispatch._core`. The two pure helpers
here distill a list of guided source records into verified findings and
unverified leads.
"""

from __future__ import annotations

from typing import Any

def _guided_findings_from_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for source in sources:
        if source.get("topicalRelevance") != "on_topic":
            continue
        verification_status = str(source.get("verificationStatus") or "")
        if verification_status not in {"verified_primary_source", "verified_metadata"}:
            continue
        claim = str(source.get("title") or source.get("note") or source.get("sourceId") or "").strip()
        if not claim:
            continue
        findings.append(
            {
                "claim": claim,
                "supportingSourceIds": [source["sourceId"]],
                "trustLevel": "verified",
            }
        )
    return findings




def _guided_unverified_leads_from_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        if source.get("topicalRelevance") == "on_topic" and source.get("verificationStatus") in {
            "verified_primary_source",
            "verified_metadata",
        }:
            continue
        source_id = str(source.get("sourceId") or "").strip()
        if source_id and source_id in seen:
            continue
        if source_id:
            seen.add(source_id)
        leads.append(source)
    return leads[:6]


