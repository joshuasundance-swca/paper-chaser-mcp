#!/usr/bin/env python3
"""Load the repo .env file, then launch the paper_chaser_mcp module."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv


def load_env_file(dotenv_path: Path) -> None:
    """Populate os.environ from a .env file without overriding existing values."""
    if not dotenv_path.exists():
        print(f"Warning: .env file not found at {dotenv_path}", file=sys.stderr)
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    load_env_file(repo_root / ".env")
    runpy.run_module("paper_chaser_mcp", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
