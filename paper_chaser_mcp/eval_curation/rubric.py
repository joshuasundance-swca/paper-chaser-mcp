"""Environmental-science benchmark pack and LLM judge rubric scaffolding."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_SCI_JUDGE_ROLES: tuple[str, ...] = (
    "plannerSpecificity",
    "followUpResponsiveness",
    "evidenceSufficiency",
    "provenanceHonesty",
)


@dataclass
class EnvSciJudgeRubric:
    """Role-level rubric scores emitted by an LLM judge for an env-sci row.

    Each role score lives in ``[0.0, 1.0]``. ``overall`` is the unweighted mean
    of the four role scores. ``notes`` is a free-form list of short strings the
    judge can use to cite concrete evidence behind each score.
    """

    plannerSpecificity: float = 0.0
    followUpResponsiveness: float = 0.0
    evidenceSufficiency: float = 0.0
    provenanceHonesty: float = 0.0
    overall: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plannerSpecificity": float(self.plannerSpecificity),
            "followUpResponsiveness": float(self.followUpResponsiveness),
            "evidenceSufficiency": float(self.evidenceSufficiency),
            "provenanceHonesty": float(self.provenanceHonesty),
            "overall": float(self.overall),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EnvSciJudgeRubric:
        def _clamp(value: Any) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            if number < 0.0:
                return 0.0
            if number > 1.0:
                return 1.0
            return number

        planner = _clamp(payload.get("plannerSpecificity"))
        follow_up = _clamp(payload.get("followUpResponsiveness"))
        evidence = _clamp(payload.get("evidenceSufficiency"))
        provenance = _clamp(payload.get("provenanceHonesty"))
        overall_raw = payload.get("overall")
        if overall_raw is None:
            overall = (planner + follow_up + evidence + provenance) / 4.0
        else:
            overall = _clamp(overall_raw)
        notes_raw = payload.get("notes") or []
        notes = [str(item) for item in notes_raw if isinstance(item, str) and item.strip()]
        return cls(
            plannerSpecificity=planner,
            followUpResponsiveness=follow_up,
            evidenceSufficiency=evidence,
            provenanceHonesty=provenance,
            overall=overall,
            notes=notes,
        )


DEFAULT_ENV_SCI_BENCHMARK_PACK = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "evals" / "env_sci_benchmark_pack.json"
)
DEFAULT_ENV_SCI_JUDGE_RUBRIC = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "evals" / "env_sci_judge_rubric.json"
)


def load_env_sci_benchmark_pack(path: Path | None = None) -> dict[str, Any]:
    """Load the env-sci benchmark pack JSON and validate the minimum schema."""

    pack_path = path or DEFAULT_ENV_SCI_BENCHMARK_PACK
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"invalid env-sci benchmark pack at {pack_path}: missing 'rows'")
    return payload


def load_env_sci_judge_rubric_template(path: Path | None = None) -> dict[str, Any]:
    rubric_path = path or DEFAULT_ENV_SCI_JUDGE_RUBRIC
    payload = json.loads(rubric_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("template"), list):
        raise ValueError(f"invalid env-sci rubric template at {rubric_path}: missing 'template'")
    return payload


def build_env_sci_judge_prompt(
    captured_row: dict[str, Any],
    *,
    expected_row: dict[str, Any] | None = None,
    template: dict[str, Any] | None = None,
) -> str:
    """Render a judge prompt from a captured tool response and an expected-row.

    The returned prompt is a plain string suitable for handing to an LLM judge.
    It includes the original query, the compacted tool response, and the
    expected-behavior row (if supplied) so the judge can score each role
    against concrete guidance. The prompt intentionally asks for strict JSON
    output keyed by ``ENV_SCI_JUDGE_ROLES`` so round-tripping through
    :meth:`EnvSciJudgeRubric.from_dict` is safe.
    """

    rubric = template or load_env_sci_judge_rubric_template()
    lines = list(rubric.get("template") or [])

    query = ""
    if isinstance(captured_row, dict):
        query = str(
            captured_row.get("query")
            or captured_row.get("input", {}).get("query")
            or captured_row.get("payload", {}).get("arguments", {}).get("query")
            or ""
        )
    response_json = json.dumps(captured_row, indent=2, sort_keys=True, default=str)
    expected_json = json.dumps(expected_row or {}, indent=2, sort_keys=True, default=str)

    prompt = "\n".join(lines)
    prompt = (
        prompt.replace("{{query}}", query or "(no query captured)")
        .replace("{{response_json}}", response_json)
        .replace("{{expected_json}}", expected_json)
    )

    role_block_lines = ["", "Role scoring guidance:"]
    for role in rubric.get("roles") or []:
        if not isinstance(role, dict):
            continue
        role_block_lines.append(f"- {role.get('id')}: {role.get('purpose')}")
        for guidance in role.get("scoringGuidance") or []:
            role_block_lines.append(f"    * {guidance}")
    prompt = prompt + "\n".join(role_block_lines) + "\n"
    return prompt
