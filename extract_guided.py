"""Helper script to extract functions from _core.py to guided submodules.

Usage: python extract_guided.py <submodule> <func1> <func2> ...

Reads paper_chaser_mcp/dispatch/_core.py, extracts the named functions
(using AST for exact line ranges), writes them to
paper_chaser_mcp/dispatch/guided/<submodule>.py (appending), and removes
them from _core.py.

Also emits to stdout the list of removed function names so the caller
can add re-imports to _core.py's bottom.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "paper_chaser_mcp" / "dispatch" / "_core.py"
GUIDED_DIR = ROOT / "paper_chaser_mcp" / "dispatch" / "guided"


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python extract_guided.py <submodule> <func1> [func2 ...]", file=sys.stderr)
        sys.exit(1)
    submodule = sys.argv[1]
    target_funcs = set(sys.argv[2:])

    src = CORE.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    # Find spans: (start_line, end_line) 1-indexed inclusive. Include
    # leading blank lines + preceding decorator lines (already in node.lineno).
    spans: list[tuple[int, int, str]] = []
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in target_funcs:
            start = node.lineno
            # Back up over decorators (already included in node.lineno by ast).
            # Back up over preceding blank lines + comment block directly above.
            while start > 1 and lines[start - 2].strip() == "":
                start -= 1
            end = node.end_lineno or node.lineno
            # Forward to include any trailing blank lines before the next def/class.
            spans.append((start, end, node.name))
            found.add(node.name)

    missing = target_funcs - found
    if missing:
        print(f"ERROR: functions not found in _core.py: {sorted(missing)}", file=sys.stderr)
        sys.exit(2)

    # Sort spans by start line
    spans.sort()

    # Build extracted text (preserving original order)
    extracted_chunks = []
    for start, end, name in spans:
        # Include trailing blank lines up to next non-blank (limit 2)
        trail_end = end
        trailing_blanks = 0
        while trail_end < len(lines) and lines[trail_end].strip() == "" and trailing_blanks < 2:
            trail_end += 1
            trailing_blanks += 1
        chunk = "".join(lines[start - 1 : trail_end])
        extracted_chunks.append((start, trail_end, name, chunk))

    # Compose new _core.py lines (skip extracted spans)
    skip_ranges = [(s, e) for s, e, _, _ in extracted_chunks]
    # Merge overlapping ranges (shouldn't happen but safe)
    skip_ranges.sort()
    new_lines: list[str] = []
    i = 0
    skip_idx = 0
    while i < len(lines):
        if skip_idx < len(skip_ranges) and i + 1 >= skip_ranges[skip_idx][0] and i + 1 <= skip_ranges[skip_idx][1]:
            # Skip this line (part of extracted function)
            if i + 1 == skip_ranges[skip_idx][1]:
                skip_idx += 1
            i += 1
            continue
        new_lines.append(lines[i])
        i += 1

    # Collapse any triple-blank runs in new_lines
    collapsed = []
    blank_run = 0
    for ln in new_lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                collapsed.append(ln)
        else:
            blank_run = 0
            collapsed.append(ln)

    CORE.write_text("".join(collapsed), encoding="utf-8")

    # Append to guided submodule
    target_file = GUIDED_DIR / f"{submodule}.py"
    if not target_file.exists():
        print(f"ERROR: {target_file} does not exist; create it with a header first.", file=sys.stderr)
        sys.exit(3)

    existing = target_file.read_text(encoding="utf-8")
    if not existing.endswith("\n"):
        existing += "\n"
    if not existing.endswith("\n\n"):
        existing += "\n"
    # Append all extracted chunks in their original order
    extracted_text = "".join(chunk for _, _, _, chunk in extracted_chunks)
    target_file.write_text(existing + extracted_text, encoding="utf-8")

    # Print the names of extracted symbols in original order for re-import generation
    for _, _, name, _ in extracted_chunks:
        print(name)


if __name__ == "__main__":
    main()
