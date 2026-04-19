"""Serve a lightweight local review UI for eval trace queues."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from paper_chaser_mcp.eval_trace_promotion import load_reviewed_trace_rows, write_promoted_rows


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def summarize_review_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = 0
    families: dict[str, int] = {}
    tools: dict[str, int] = {}
    for row in rows:
        review = row.get("review") or {}
        trace = row.get("trace") or {}
        if review.get("promote"):
            promoted += 1
        family = str(review.get("task_family") or "unknown").strip() or "unknown"
        source_tool = str(trace.get("source_tool") or "unknown").strip() or "unknown"
        families[family] = families.get(family, 0) + 1
        tools[source_tool] = tools.get(source_tool, 0) + 1
    return {
        "rowCount": len(rows),
        "promotedCount": promoted,
        "pendingCount": len(rows) - promoted,
        "families": families,
        "tools": tools,
    }


def _html_page(title: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --ink: #17222c;
      --panel: #fffaf0;
      --line: #d5c7a8;
      --accent: #8a4b23;
      --accent-soft: #eed8c9;
      --muted: #6c6b68;
      --good: #355b3d;
    }}
    body {{
      margin: 0;
      font-family: Georgia, 'Iowan Old Style', serif;
      background: linear-gradient(180deg, #efe6d1 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .shell {{
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      padding: 16px;
      background: rgba(255,250,240,0.9);
    }}
    .main {{ padding: 20px; }}
    .hero, .panel, .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .hero, .panel {{ padding: 14px; margin-bottom: 14px; }}
    .hero h2 {{ margin: 0 0 6px 0; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }}
    .stat {{ padding: 10px; }}
    .stat strong {{ display: block; font-size: 24px; }}
    .hint-list {{ margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; }}
    .row-link {{
      display: block;
      width: 100%;
      text-align: left;
      padding: 10px;
      margin-bottom: 8px;
      border: 1px solid var(--line);
      background: white;
      cursor: pointer;
    }}
    .row-link.active {{
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px var(--accent);
    }}
    .row-link small {{ display: block; color: var(--muted); margin-top: 4px; }}
    .badge {{
      display: inline-block;
      border: 1px solid var(--line);
      background: var(--accent-soft);
      padding: 2px 7px;
      font-size: 12px;
      margin-right: 6px;
      border-radius: 999px;
    }}
    .badge.good {{
      color: white;
      background: var(--good);
      border-color: var(--good);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    label {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    input[type=text], textarea {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      padding: 8px;
      background: white;
    }}
    textarea {{ min-height: 120px; font-family: Consolas, monospace; }}
    .toolbar {{ display: flex; gap: 10px; margin-bottom: 16px; }}
    .toolbar.wrap {{ flex-wrap: wrap; align-items: center; }}
    button {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      padding: 9px 14px;
      cursor: pointer;
    }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 420px;
      overflow: auto;
      background: #fff;
      border: 1px solid var(--line);
      padding: 10px;
    }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <aside class=\"sidebar\">
      <div class=\"hero\">
        <h2>Trace Review</h2>
        <p class=\"muted\">Review queue rows without editing JSONL by hand.</p>
        <ol class=\"hint-list\">
          <li>Pick a row from the left list.</li>
          <li>Set promote, family, expected JSON, and labels.</li>
          <li>Save to write one reviewed JSONL file.</li>
        </ol>
      </div>
      <div class=\"summary-grid\">
        <div class=\"stat\"><span class=\"muted\">Rows</span><strong id=\"rowCount\">0</strong></div>
        <div class=\"stat\"><span class=\"muted\">Promote</span><strong id=\"promotedCount\">0</strong></div>
        <div class=\"stat\"><span class=\"muted\">Pending</span><strong id=\"pendingCount\">0</strong></div>
      </div>
      <div class=\"toolbar\"><button id=\"saveAll\">Save All</button></div>
      <div id=\"rows\"></div>
    </aside>
    <main class=\"main\">
      <div class=\"toolbar wrap\">
        <button id=\"saveCurrent\">Save Current</button>
        <span class=\"badge\" id=\"currentFamily\">family</span>
        <span class=\"badge\" id=\"currentTool\">tool</span>
        <span class=\"badge\" id=\"currentPromote\">pending</span>
        <span id=\"status\" class=\"muted\"></span>
      </div>
      <div class=\"grid\">
        <section class=\"panel\">
          <label><input id=\"promote\" type=\"checkbox\" /> Promote</label>
          <label>Task Family</label><input id=\"task_family\" type=\"text\" />
          <label>Why It Matters</label><textarea id=\"why_it_matters\"></textarea>
          <label>Expected JSON</label><textarea id=\"expected\"></textarea>
          <label>Notes</label><textarea id=\"notes\"></textarea>
        </section>
        <section class=\"panel\">
          <label>Quick Summary</label><pre id=\"traceSummary\"></pre>
          <label>Trace Summary</label><pre id=\"trace\"></pre>
          <label>Labels JSON</label><textarea id=\"labels\"></textarea>
        </section>
      </div>
    </main>
  </div>
  <script>
    let rows = [];
    let current = 0;

    function updateSummary() {{
      const promoted = rows.filter(row => row.review && row.review.promote).length;
      document.getElementById('rowCount').textContent = rows.length;
      document.getElementById('promotedCount').textContent = promoted;
      document.getElementById('pendingCount').textContent = rows.length - promoted;
    }}

    async function loadRows() {{
      const response = await fetch('/api/rows');
      const payload = await response.json();
      rows = payload.rows;
      updateSummary();
      renderList();
      renderCurrent();
    }}

    function safeJson(value) {{
      try {{
        return JSON.parse(value || '{{}}');
      }} catch {{
        return null;
      }}
    }}

    function currentRow() {{
      return rows[current];
    }}

    function summarizeTrace(row) {{
      const trace = row.trace || {{}};
      const captured = trace.captured_output || {{}};
      return {{
        source_tool: trace.source_tool || null,
        query: trace.query || null,
        query_context: trace.query_context || null,
        search_session_id: trace.search_session_id || null,
        duration_ms: trace.duration_ms || null,
        source_count: captured.sourceCount || null,
        answer_status: captured.answerStatus || null,
        result_status: captured.resultStatus || captured.status || null,
      }};
    }}

    function renderList() {{
      const container = document.getElementById('rows');
      container.innerHTML = '';
      rows.forEach((row, index) => {{
        const button = document.createElement('button');
        const taskFamily = row.review.task_family || 'unknown';
        const sourceTool = row.trace.source_tool || 'unknown';
        const promote = row.review.promote ? 'promote' : 'pending';
        button.className = 'row-link' + (index === current ? ' active' : '');
        button.innerHTML = [
          `<strong>${{row.trace_id}}</strong>`,
          `<small>${{taskFamily}} | ${{sourceTool}} | ${{promote}}</small>`
        ].join('');
        button.onclick = () => {{
          current = index;
          renderList();
          renderCurrent();
        }};
        container.appendChild(button);
      }});
    }}

    function renderCurrent() {{
      const row = currentRow();
      if (!row) return;
      document.getElementById('promote').checked = !!row.review.promote;
      document.getElementById('task_family').value = row.review.task_family || '';
      document.getElementById('why_it_matters').value = row.review.why_it_matters || '';
      document.getElementById('expected').value = JSON.stringify(row.review.expected || {{}}, null, 2);
      document.getElementById('notes').value = row.review.notes || '';
      document.getElementById('labels').value = JSON.stringify(row.review.labels || {{}}, null, 2);
      document.getElementById('traceSummary').textContent = JSON.stringify(summarizeTrace(row), null, 2);
      document.getElementById('trace').textContent = JSON.stringify(row.trace || {{}}, null, 2);
      document.getElementById('currentFamily').textContent = row.review.task_family || 'unknown';
      document.getElementById('currentTool').textContent = row.trace.source_tool || 'unknown';
      const badge = document.getElementById('currentPromote');
      badge.textContent = row.review.promote ? 'promote' : 'pending';
      badge.className = 'badge' + (row.review.promote ? ' good' : '');
    }}

    function collectCurrent() {{
      const row = currentRow();
      row.review.promote = document.getElementById('promote').checked;
      row.review.task_family = document.getElementById('task_family').value;
      row.review.why_it_matters = document.getElementById('why_it_matters').value;
      row.review.notes = document.getElementById('notes').value;
      const expected = safeJson(document.getElementById('expected').value);
      const labels = safeJson(document.getElementById('labels').value);
      if (expected === null || labels === null) {{
        document.getElementById('status').textContent = 'Expected or labels JSON is invalid.';
        return false;
      }}
      row.review.expected = expected;
      row.review.labels = labels;
      updateSummary();
      document.getElementById('status').textContent = 'Unsaved changes staged locally.';
      return true;
    }}

    async function saveAll() {{
      if (!collectCurrent()) return;
      const response = await fetch('/api/save', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ rows }})
      }});
      const payload = await response.json();
      document.getElementById('status').textContent = payload.message;
    }}

    document.getElementById('saveCurrent').onclick = saveAll;
    document.getElementById('saveAll').onclick = saveAll;
    loadRows();
  </script>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a lightweight local review UI for eval traces.")
    parser.add_argument("--input", required=True, help="Path to review queue or reviewed traces JSONL.")
    parser.add_argument("--output", required=True, help="Path to write reviewed trace JSONL.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local review server.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the local review server.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = load_reviewed_trace_rows(Path(args.input))
    output_path = Path(args.output)

    class ReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/rows":
                summary = summarize_review_rows(rows)
                _json_response(self, {"rows": rows, "summary": summary})
                return
            if self.path == "/":
                html = _html_page("Paper Chaser Eval Review")
                encoded = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            incoming_rows = payload.get("rows") or []
            if not isinstance(incoming_rows, list):
                _json_response(self, {"message": "rows must be a list"}, status=400)
                return
            rows[:] = incoming_rows
            write_promoted_rows(output_path, rows)
            _json_response(self, {"message": f"Saved {len(rows)} reviewed row(s) to {output_path}."})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return None

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Review UI available at http://{args.host}:{args.port}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
