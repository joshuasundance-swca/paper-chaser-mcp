"""Serve a lightweight local viewer for ranked generated topic pools."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def summarize_generated_topics(payload: dict[str, Any]) -> dict[str, Any]:
    existing_summary = payload.get("summary")
    if isinstance(existing_summary, dict) and existing_summary:
        return existing_summary

    topics = payload.get("topics") or []
    if not isinstance(topics, list):
        topics = []

    families: dict[str, int] = {}
    intents: dict[str, int] = {}
    quality_tiers: dict[str, int] = {}
    total_score = 0.0
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        family = str(topic.get("family") or "unknown")
        intent = str(topic.get("intent") or "unknown")
        tier = str(topic.get("qualityTier") or "unscored")
        families[family] = families.get(family, 0) + 1
        intents[intent] = intents.get(intent, 0) + 1
        quality_tiers[tier] = quality_tiers.get(tier, 0) + 1
        total_score += float(topic.get("qualityScore") or 0.0)

    topic_count = len([topic for topic in topics if isinstance(topic, dict)])
    return {
        "topicCount": topic_count,
        "averageQualityScore": round(total_score / topic_count, 1) if topic_count else 0.0,
        "families": families,
        "intents": intents,
        "qualityTiers": quality_tiers,
        "familyCount": len(families),
        "intentCount": len(intents),
        "topTopics": [
            {
                "query": topic.get("query"),
                "family": topic.get("family"),
                "qualityScore": topic.get("qualityScore"),
                "qualityTier": topic.get("qualityTier"),
            }
            for topic in topics[:10]
            if isinstance(topic, dict)
        ],
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
      --bg: #f4efe7;
      --ink: #17222d;
      --panel: #fffdf8;
      --line: #d9d0c3;
      --accent: #245d73;
      --accent-soft: #d7ebf2;
      --muted: #655f58;
    }}
    body {{
      margin: 0;
      font-family: Georgia, 'Palatino Linotype', serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .shell {{ display: grid; grid-template-columns: 380px 1fr; min-height: 100vh; }}
    .sidebar {{
      padding: 18px;
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, #f6f1e6 0%, #fbf8f2 100%);
      overflow-y: auto;
    }}
    .main {{ padding: 22px; }}
    .panel, .stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }}
    .panel {{ padding: 14px; margin-bottom: 14px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px; }}
    .stat {{ padding: 10px; }}
    .stat strong {{ display: block; font-size: 24px; }}
    .topic-link {{
      display: block;
      width: 100%;
      text-align: left;
      margin-bottom: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      background: white;
      cursor: pointer;
    }}
    .topic-link.active {{ border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: var(--accent-soft);
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 360px;
      overflow: auto;
      background: #fff;
      border: 1px solid var(--line);
      padding: 10px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    h4 {{ margin-bottom: 8px; }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <aside class=\"sidebar\">
      <div class=\"panel\">
        <h2>Topic Viewer</h2>
        <p class=\"muted\">
          Inspect ranked topics, blockers, warnings, and prune or cross-check signals without hand-reading JSON.
        </p>
      </div>
      <div class=\"summary-grid\">
        <div class=\"stat\"><span class=\"muted\">Topics</span><strong id=\"topicCount\">0</strong></div>
        <div class=\"stat\"><span class=\"muted\">Avg Score</span><strong id=\"avgScore\">0</strong></div>
        <div class=\"stat\"><span class=\"muted\">Families</span><strong id=\"familyCount\">0</strong></div>
      </div>
      <div class=\"panel\">
        <h3>Pool Review</h3>
        <p class=\"muted\" id=\"reviewRecommendation\">unknown</p>
        <h4>Risk Warnings</h4>
        <pre id=\"riskWarnings\"></pre>
        <h4>Hard Blockers</h4>
        <pre id=\"hardBlockers\"></pre>
        <h4>Generation Warnings</h4>
        <pre id=\"generationWarnings\"></pre>
        <h4>Prune Audit</h4>
        <pre id=\"pruneAuditJson\"></pre>
      </div>
      <div id=\"topics\"></div>
    </aside>
    <main class=\"main\">
      <div class=\"panel\">
        <h2 id=\"topicQuery\">Topic</h2>
        <div id=\"badges\"></div>
      </div>
      <div class=\"panel\">
        <h3>Summary</h3>
        <table><tbody id=\"summaryTable\"></tbody></table>
      </div>
      <div class=\"panel\">
        <h3>Quality Signals</h3>
        <pre id=\"signalsJson\"></pre>
      </div>
      <div class=\"panel\">
        <h3>Family Cross-Check</h3>
        <pre id=\"familyCrossCheckJson\"></pre>
      </div>
      <div class=\"panel\">
        <h3>Raw Topic</h3>
        <pre id=\"topicJson\"></pre>
      </div>
    </main>
  </div>
  <script>
    let payload = {{ topics: [] }};
    let current = 0;

    async function loadPayload() {{
      const response = await fetch('/api/topics');
      payload = await response.json();
      renderSidebar();
      renderCurrent();
    }}

    function renderSidebar() {{
      const summary = payload.summary || {{}};
      document.getElementById('topicCount').textContent = summary.topicCount || 0;
      document.getElementById('avgScore').textContent = summary.averageQualityScore || 0;
      document.getElementById('familyCount').textContent = Object.keys(summary.families || {{}}).length;
      document.getElementById('reviewRecommendation').textContent = payload.reviewRecommendation || 'unknown';
      document.getElementById('riskWarnings').textContent = JSON.stringify(payload.riskWarnings || [], null, 2);
      document.getElementById('hardBlockers').textContent = JSON.stringify(payload.hardBlockers || [], null, 2);
      document.getElementById('generationWarnings').textContent = JSON.stringify(
        payload.generationWarnings || [],
        null,
        2
      );
      document.getElementById('pruneAuditJson').textContent = JSON.stringify(
        (payload.pruneSummary || {{}}).audit || [],
        null,
        2
      );
      const container = document.getElementById('topics');
      container.innerHTML = '';
      (payload.topics || []).forEach((topic, index) => {{
        const button = document.createElement('button');
        button.className = 'topic-link' + (index === current ? ' active' : '');
        button.innerHTML = [
          `<strong>${{topic.query}}</strong>`,
          `<small>${{topic.family || 'unknown'}} | ${{topic.intent || 'unknown'}} | ` +
          `score ${{topic.qualityScore || 0}}</small>`
        ].join('');
        button.onclick = () => {{ current = index; renderSidebar(); renderCurrent(); }};
        container.appendChild(button);
      }});
    }}

    function renderCurrent() {{
      const topic = (payload.topics || [])[current];
      if (!topic) return;
      document.getElementById('topicQuery').textContent = topic.query || 'Topic';
      const badges = document.getElementById('badges');
      badges.innerHTML = [
        `<span class=\"badge\">${{topic.family || 'unknown family'}}</span>`,
        `<span class=\"badge\">${{topic.intent || 'unknown intent'}}</span>`,
        `<span class=\"badge\">score ${{topic.qualityScore || 0}}</span>`,
        `<span class=\"badge\">${{topic.qualityTier || 'unscored'}}</span>`
      ].join('') + ((topic.tags || []).map(tag => `<span class=\"badge\">${{tag}}</span>`).join(''));
      const rows = [
        ['Seed Query', topic.seedQuery],
        ['Follow-up Mode', topic.followUpMode],
        ['Provider Plan', (topic.providerPlan || []).join('; ')],
        ['Candidate Concepts', (topic.candidateConcepts || []).join('; ')],
        ['Success Criteria', (topic.successCriteria || []).join('; ')],
        ['Rationale', topic.rationale]
      ];
      document.getElementById('summaryTable').innerHTML = rows
        .map(row => `<tr><th>${{row[0]}}</th><td>${{row[1] || ''}}</td></tr>`)
        .join('');
      document.getElementById('signalsJson').textContent = JSON.stringify(topic.qualitySignals || [], null, 2);
      document.getElementById('familyCrossCheckJson').textContent = JSON.stringify(
        topic.familyCrossCheck || {{}},
        null,
        2
      );
      document.getElementById('topicJson').textContent = JSON.stringify(topic, null, 2);
    }}

    loadPayload();
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a lightweight local viewer for ranked generated topics.")
    parser.add_argument("--input", required=True, help="Path to generated-topics JSON.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local viewer server.")
    parser.add_argument("--port", type=int, default=8767, help="Port to bind the local viewer server.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    summary = summarize_generated_topics(payload)

    class TopicHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/topics":
                _json_response(
                    self,
                    {
                        "summary": summary,
                        "topics": payload.get("topics") or [],
                        "riskWarnings": payload.get("riskWarnings") or [],
                        "hardBlockers": payload.get("hardBlockers") or [],
                        "generationWarnings": payload.get("generationWarnings") or [],
                        "reviewRecommendation": payload.get("reviewRecommendation") or "unknown",
                        "pruneSummary": payload.get("pruneSummary") or {},
                    },
                )
                return
            if self.path == "/":
                html = _html_page("Paper Chaser Generated Topics")
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

    server = ThreadingHTTPServer((args.host, args.port), TopicHandler)
    print(f"Topic viewer available at http://{args.host}:{args.port}")
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
