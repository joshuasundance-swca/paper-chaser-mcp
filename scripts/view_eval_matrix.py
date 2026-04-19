"""Serve a lightweight local viewer for live eval matrix results."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _item_live_evaluation(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("liveEvaluation")
    return value if isinstance(value, dict) else {}


def _executed_tool_sequence(live: dict[str, Any]) -> list[str]:
    sequence: list[str] = []
    for step in live.get("executedSteps") or []:
        if not isinstance(step, dict):
            continue
        tool_name = str(step.get("tool") or "").strip()
        if tool_name:
            sequence.append(tool_name)
    return sequence


def _search_session_lineage(live: dict[str, Any]) -> list[str]:
    lineage: list[str] = []
    top_level = str(live.get("searchSessionId") or "").strip()
    if top_level:
        lineage.append(top_level)
    for step in live.get("executedSteps") or []:
        if not isinstance(step, dict):
            continue
        session_id = str(step.get("searchSessionId") or "").strip()
        if session_id and session_id not in lineage:
            lineage.append(session_id)
    return lineage


def summarize_matrix_results(payload: dict[str, Any]) -> dict[str, Any]:
    scenarios = payload.get("scenarios") or []
    if not isinstance(scenarios, list):
        scenarios = []
    passed = 0
    failed = 0
    scenario_summaries: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        exit_code = int(scenario.get("exitCode") or 0)
        report = scenario.get("report") or {}
        summary = report.get("summary") or {}
        live_summary = report.get("liveSummary") or {}
        status = "passed" if exit_code == 0 else "failed"
        if status == "passed":
            passed += 1
        else:
            failed += 1
        env = scenario.get("env") or {}
        provider = str(env.get("PAPER_CHASER_AGENTIC_PROVIDER") or "default")
        scenario_summaries.append(
            {
                "name": scenario.get("name"),
                "status": status,
                "provider": provider,
                "family": scenario.get("family") or "all",
                "exitCode": exit_code,
                "failedItems": summary.get("failedItems", 0),
                "warningItems": summary.get("warningItems", 0),
                "liveFailed": live_summary.get("failed", 0),
                "itemCount": summary.get("itemCount", 0),
            }
        )
    return {
        "scenarioCount": len(scenario_summaries),
        "passedScenarios": passed,
        "failedScenarios": failed,
        "scenarios": scenario_summaries,
    }


def summarize_matrix_divergences(payload: dict[str, Any]) -> dict[str, Any]:
    scenarios = payload.get("scenarios") or []
    if not isinstance(scenarios, list):
        scenarios = []

    by_item: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        provider = str((scenario.get("env") or {}).get("PAPER_CHASER_AGENTIC_PROVIDER") or "default")
        for item in (scenario.get("report") or {}).get("items") or []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                continue
            live = _item_live_evaluation(item)
            entry = by_item.setdefault(
                item_id,
                {
                    "itemId": item_id,
                    "family": item.get("family") or "unknown",
                    "byProvider": {},
                },
            )
            entry["byProvider"][provider] = {
                "scenario": scenario.get("name"),
                "status": item.get("status"),
                "liveStatus": live.get("status"),
                "observedIntent": live.get("observedIntent"),
                "executedToolSequence": _executed_tool_sequence(live),
                "searchSessionLineage": _search_session_lineage(live),
                "errors": item.get("errors") or [],
                "warnings": item.get("warnings") or [],
                "reason": live.get("reason"),
            }

    divergences: list[dict[str, Any]] = []
    for item_id, entry in sorted(by_item.items()):
        by_provider = entry["byProvider"]
        statuses = {str(value.get("status") or "") for value in by_provider.values()}
        live_statuses = {str(value.get("liveStatus") or "") for value in by_provider.values()}
        observed_intents = {
            str(value.get("observedIntent") or "")
            for value in by_provider.values()
            if str(value.get("observedIntent") or "")
        }
        reasons = {str(value.get("reason") or "") for value in by_provider.values() if str(value.get("reason") or "")}
        tool_sequences = {
            tuple(str(step) for step in (value.get("executedToolSequence") or [])) for value in by_provider.values()
        }
        session_lineages = {
            tuple(str(session_id) for session_id in (value.get("searchSessionLineage") or []))
            for value in by_provider.values()
        }

        divergence_reasons: list[str] = []
        if len(statuses) > 1:
            divergence_reasons.append("item status differs")
        if len(live_statuses) > 1:
            divergence_reasons.append("live evaluation status differs")
        if len(observed_intents) > 1:
            divergence_reasons.append("observed intent differs")
        if len(reasons) > 1:
            divergence_reasons.append("skip or failure reason differs")
        if len(tool_sequences) > 1:
            divergence_reasons.append("executed tool sequence differs")
        if len(session_lineages) > 1:
            divergence_reasons.append("search session lineage differs")

        if divergence_reasons:
            divergences.append(
                {
                    "itemId": item_id,
                    "family": entry["family"],
                    "divergenceReasons": divergence_reasons,
                    "providers": by_provider,
                }
            )

    return {
        "itemCount": len(by_item),
        "divergentItemCount": len(divergences),
        "divergences": divergences,
    }


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _html_page(title: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f7f4eb;
      --ink: #16202b;
      --panel: #fffdf7;
      --line: #d8cfbf;
      --accent: #1b5d7a;
      --accent-soft: #d7edf6;
      --good: #2d6a4f;
      --bad: #9b2226;
      --muted: #67625d;
    }}
    body {{
      margin: 0;
      font-family: Georgia, 'Palatino Linotype', serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .shell {{
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      padding: 18px;
      background: linear-gradient(180deg, #f4efe2 0%, #fbf8f1 100%);
    }}
    .main {{ padding: 22px; }}
    .hero, .panel, .stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }}
    .hero, .panel {{ padding: 14px; margin-bottom: 14px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px; }}
    .stat {{ padding: 10px; }}
    .stat strong {{ display: block; font-size: 26px; }}
    .scenario-link {{
      display: block;
      width: 100%;
      text-align: left;
      margin-bottom: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      background: white;
      cursor: pointer;
    }}
    .scenario-link.active {{ border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }}
    .scenario-link small {{ display: block; color: var(--muted); margin-top: 4px; }}
    .divergence-link {{
      display: block;
      width: 100%;
      text-align: left;
      margin-bottom: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      background: #fff;
      cursor: pointer;
    }}
    .divergence-link small {{ display: block; color: var(--muted); margin-top: 4px; }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: var(--accent-soft);
      margin-right: 6px;
    }}
    .badge.good {{ background: var(--good); border-color: var(--good); color: white; }}
    .badge.bad {{ background: var(--bad); border-color: var(--bad); color: white; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .empty {{
      border: 1px dashed var(--line);
      background: #fff;
      border-radius: 10px;
      padding: 12px;
      color: var(--muted);
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 520px;
      overflow: auto;
      background: #fff;
      border: 1px solid var(--line);
      padding: 10px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid var(--line); text-align: left; }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <aside class=\"sidebar\">
      <div class=\"hero\">
        <h2>Matrix Viewer</h2>
        <p class="muted">Inspect provider comparison results without reading raw JSON.</p>
        <ul class="muted">
          <li>No divergence does not prove identical reasoning or identical hidden traces.</li>
          <li>Two providers can reach the same answer through different retrieval paths.</li>
          <li>Divergence here means the current captured eval signals differ enough to observe.</li>
        </ul>
      </div>
      <div class=\"summary-grid\">
        <div class=\"stat\"><span class=\"muted\">Scenarios</span><strong id=\"scenarioCount\">0</strong></div>
        <div class=\"stat\"><span class=\"muted\">Passed</span><strong id=\"passedCount\">0</strong></div>
        <div class=\"stat\"><span class=\"muted\">Failed</span><strong id=\"failedCount\">0</strong></div>
      </div>
        <p class="muted">
          This can catch same-answer, different-trace behavior when tool sequence or session lineage changes.
        </p>
      <div id=\"scenarios\"></div>
    </aside>
    <main class=\"main\">
      <div class=\"panel\">
        <h2 id=\"scenarioName\">Scenario</h2>
        <div>
          <span class=\"badge\" id=\"scenarioProvider\">provider</span>
          <span class=\"badge\" id=\"scenarioFamily\">family</span>
          <span class=\"badge\" id=\"scenarioStatus\">status</span>
        </div>
      </div>
      <div class=\"grid\">
        <section class=\"panel\">
          <h3>Summary</h3>
          <p class="muted">Use this as a triage view, not a proof of model internals or identical reasoning.</p>
          <table>
            <tbody id=\"summaryTable\"></tbody>
          </table>
        </section>
        <section class=\"panel\">
          <h3>Environment</h3>
          <pre id=\"envJson\"></pre>
        </section>
      </div>
      <div class=\"panel\">
        <h3>Divergence View</h3>
        <p class=\"muted\">Matched item IDs that diverged across providers.</p>
        <div class=\"summary-grid\">
          <div class=\"stat\"><span class=\"muted\">Items</span><strong id=\"itemCount\">0</strong></div>
          <div class=\"stat\"><span class=\"muted\">Divergent</span><strong id=\"divergentCount\">0</strong></div>
          <div class=\"stat\"><span class=\"muted\">Selected</span><strong id=\"selectedItem\">none</strong></div>
        </div>
        <div class=\"grid\">
          <div>
            <div id=\"divergences\"></div>
            <div id=\"noDivergences\" class=\"empty\">No material divergence detected across matched items.</div>
          </div>
          <div>
            <pre id=\"divergenceJson\"></pre>
          </div>
        </div>
      </div>
      <div class=\"grid\">
        <section class=\"panel\">
          <h3>Items</h3>
          <pre id=\"itemsJson\"></pre>
        </section>
        <section class=\"panel\">
          <h3>Raw Scenario</h3>
          <pre id=\"scenarioJson\"></pre>
        </section>
      </div>
    </main>
  </div>
  <script>
    let payload = {{ scenarios: [] }};
    let current = 0;
    let divergenceIndex = 0;

    async function loadPayload() {{
      const response = await fetch('/api/matrix');
      payload = await response.json();
      renderSidebar();
      renderCurrent();
      renderDivergences();
    }}

    function renderSidebar() {{
      document.getElementById('scenarioCount').textContent = payload.summary.scenarioCount;
      document.getElementById('passedCount').textContent = payload.summary.passedScenarios;
      document.getElementById('failedCount').textContent = payload.summary.failedScenarios;
      const container = document.getElementById('scenarios');
      container.innerHTML = '';
      payload.summary.scenarios.forEach((scenario, index) => {{
        const button = document.createElement('button');
        button.className = 'scenario-link' + (index === current ? ' active' : '');
        button.innerHTML = [
          `<strong>${{scenario.name}}</strong>`,
          `<small>${{scenario.provider}} | ${{scenario.family}} | ${{scenario.status}}</small>`
        ].join('');
        button.onclick = () => {{ current = index; renderSidebar(); renderCurrent(); }};
        container.appendChild(button);
      }});
    }}

    function renderCurrent() {{
      const scenario = payload.scenarios[current];
      const summary = payload.summary.scenarios[current];
      if (!scenario || !summary) return;
      document.getElementById('scenarioName').textContent = summary.name;
      document.getElementById('scenarioProvider').textContent = summary.provider;
      document.getElementById('scenarioFamily').textContent = summary.family;
      const statusBadge = document.getElementById('scenarioStatus');
      statusBadge.textContent = summary.status;
      statusBadge.className = 'badge ' + (summary.status === 'passed' ? 'good' : 'bad');
      const rows = [
        ['Exit Code', scenario.exitCode],
        ['Item Count', summary.itemCount],
        ['Failed Items', summary.failedItems],
        ['Warnings', summary.warningItems],
        ['Live Failed', summary.liveFailed],
      ];
      document.getElementById('summaryTable').innerHTML = rows
        .map(row => `<tr><th>${{row[0]}}</th><td>${{row[1]}}</td></tr>`)
        .join('');
      document.getElementById('envJson').textContent = JSON.stringify(scenario.env || {{}}, null, 2);
      document.getElementById('itemsJson').textContent = JSON.stringify((scenario.report || {{}}).items || [], null, 2);
      document.getElementById('scenarioJson').textContent = JSON.stringify(scenario, null, 2);
    }}

    function renderDivergences() {{
      const divergenceSummary = payload.divergences || {{ divergences: [], itemCount: 0, divergentItemCount: 0 }};
      const divergences = divergenceSummary.divergences || [];
      document.getElementById('itemCount').textContent = divergenceSummary.itemCount || 0;
      document.getElementById('divergentCount').textContent = divergenceSummary.divergentItemCount || 0;
      const empty = document.getElementById('noDivergences');
      const container = document.getElementById('divergences');
      container.innerHTML = '';
      if (!divergences.length) {{
        empty.style.display = 'block';
        document.getElementById('selectedItem').textContent = 'none';
        document.getElementById('divergenceJson').textContent = JSON.stringify({{
          message: 'No divergence detected',
          comparedItems: divergenceSummary.itemCount || 0
        }}, null, 2);
        return;
      }}
      empty.style.display = 'none';
      divergenceIndex = Math.min(divergenceIndex, divergences.length - 1);
      divergences.forEach((divergence, index) => {{
        const button = document.createElement('button');
        button.className = 'divergence-link' + (index === divergenceIndex ? ' active' : '');
        button.innerHTML = [
          `<strong>${{divergence.itemId}}</strong>`,
          `<small>${{divergence.family}} | ${{divergence.divergenceReasons.join('; ')}}</small>`
        ].join('');
        button.onclick = () => {{
          divergenceIndex = index;
          renderDivergences();
        }};
        container.appendChild(button);
      }});
      const selected = divergences[divergenceIndex];
      document.getElementById('selectedItem').textContent = selected.itemId;
      document.getElementById('divergenceJson').textContent = JSON.stringify(selected, null, 2);
    }}

    loadPayload();
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a lightweight local viewer for live eval matrix results.")
    parser.add_argument("--input", required=True, help="Path to matrix results JSON.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local viewer server.")
    parser.add_argument("--port", type=int, default=8766, help="Port to bind the local viewer server.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    summary = summarize_matrix_results(payload)
    divergences = summarize_matrix_divergences(payload)

    class MatrixHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/matrix":
                _json_response(
                    self,
                    {
                        "summary": summary,
                        "divergences": divergences,
                        "scenarios": payload.get("scenarios") or [],
                    },
                )
                return
            if self.path == "/":
                html = _html_page("Paper Chaser Eval Matrix")
                encoded = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return None

    server = ThreadingHTTPServer((args.host, args.port), MatrixHandler)
    print(f"Matrix viewer available at http://{args.host}:{args.port}")
    print(f"Input: {args.input}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
