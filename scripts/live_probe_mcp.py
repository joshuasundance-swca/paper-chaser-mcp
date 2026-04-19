"""Live MCP client probe: spawns the paper-chaser server over stdio and
exercises guided tools across Phase 1/2 surfaces.

Run with the project venv:
    .venv\\Scripts\\python.exe scripts\\live_probe_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def _compact(value: object, limit: int = 600) -> str:
    text = json.dumps(value, indent=2, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


async def _call(session: ClientSession, name: str, args: dict[str, object]) -> dict[str, object] | None:
    try:
        result = await session.call_tool(name, args)
    except Exception as exc:  # pragma: no cover - probe script
        print(f"  ! {name} raised: {exc!r}")
        return None
    payload: object | None = None
    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = text
    return payload if isinstance(payload, dict) else {"raw": payload}


async def _probe() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    for key, value in _load_dotenv(repo_root / ".env").items():
        env.setdefault(key, value)
    env.setdefault("PAPER_CHASER_TOOL_PROFILE", "guided")

    python_exe = sys.executable
    params = StdioServerParameters(
        command=python_exe,
        args=["-m", "paper_chaser_mcp"],
        env=env,
        cwd=str(repo_root),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = sorted(t.name for t in tools.tools)
            print(f"Advertised tools ({len(tool_names)}):", tool_names)

            print("\n== get_runtime_status ==")
            status = await _call(session, "get_runtime_status", {})
            if status:
                print(
                    _compact(
                        {
                            "toolProfile": status.get("toolProfile"),
                            "smartProvider": status.get("smartProvider"),
                            "activeProviders": status.get("activeProviders"),
                            "disabledProviders": status.get("disabledProviders"),
                        }
                    )
                )

            probe_queries: list[tuple[str, str]] = [
                ("heritage", "Section 106 NHPA tribal consultation for cultural heritage sites"),
                ("known-item", "Attention Is All You Need Vaswani 2017"),
                ("broad", "graph neural networks for molecular property prediction"),
                ("env-sci", "environmental impact assessment for offshore wind"),
                ("cultural-lit", "archaeological resources act compliance literature review"),
            ]
            for label, q in probe_queries:
                print(f"\n== research [{label}]: {q!r} ==")
                resp = await _call(session, "research", {"query": q, "limit": 3})
                if not resp:
                    continue
                print("  top-level keys:", sorted(resp.keys()))
                meta = resp.get("strategy") or resp.get("searchStrategy") or resp.get("strategyMetadata") or {}
                print("  strategy keys:", sorted(meta.keys()) if isinstance(meta, dict) else type(meta).__name__)
                print(_compact(resp, limit=1400))

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_probe()))
