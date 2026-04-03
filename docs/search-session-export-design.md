# Search Session Export Design Stub

This wave does not implement session export. The follow-up target is a guided or
expert MCP tool with this shape:

```text
export_search_session(searchSessionId="...", format="ris|bibtex|csv")
```

## Locked design assumptions

- Export will read from the saved `searchSessionId` instead of re-running search.
- Supported formats will be `ris`, `bibtex`, and `csv`.
- Export will use guided-v2 `sources[*].citation` plus source-level metadata so
  export can ship without another public schema rewrite.
- `citationText`, `citation`, `openAccessRoute`, `canonicalUrl`, and
  `retrievedUrl` are part of the forward-compatible data contract for this
  reason.

## Out of scope for this wave

- No new MCP tool is added yet.
- No RIS/BibTeX/CSV renderer is implemented yet.
- No search-session export tests are added yet.
