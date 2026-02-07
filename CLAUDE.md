# Research Intelligence Platform

Unified research pipeline combining Perplexity AI, NotebookLM, and Claude Code.

**Created**: 2026-02-07
**Location**: `~/research-platform/`

---

## Architecture

```
Streamlit UI (:8503) → Research Pipeline → Perplexity API
                                         → Qdrant Vector DB
                                         → MCP Server (5 tools)
```

## Components

| Module | Description |
|--------|-------------|
| `models.py` | Pydantic: Citation, SearchResult, ResearchProject, PerplexityModel |
| `perplexity_client.py` | httpx client, rate limiting, citations parsing |
| `storage.py` | Qdrant + OpenAI embeddings (text-embedding-3-small, 1536 dim) |
| `pipeline.py` | Orchestrator: quick_search, deep_research, find_related |
| `mcp_server.py` | FastMCP server with 5 tools |
| `streamlit_app.py` | 4 tabs: Search, Projects, Knowledge Base, History |

## MCP Tools

| Tool | Description |
|------|-------------|
| `research_search` | Quick search via Perplexity (sonar/sonar-pro/reasoning/deep-research) |
| `research_deep` | Multi-query deep research |
| `research_find` | Semantic search over past results (Qdrant) |
| `research_history` | Search history by project |
| `research_stats` | Storage statistics |

## Running

```bash
# Streamlit UI
./run_streamlit.sh

# MCP Server (registered in ~/.claude.json)
python -m research_platform
```

## Environment Variables

- `PERPLEXITY_API_KEY` — Perplexity API key (required)
- `OPENAI_API_KEY` — OpenAI key for embeddings (required)
- `QDRANT_URL` — Qdrant URL (default: http://localhost:6333)

## Tests

```bash
pytest tests/ -v --cov=research_platform
# 51 passed, 81% coverage
```

## Dependencies

- Perplexity MCP: `@perplexity-ai/mcp-server`
- NotebookLM MCP: `notebooklm-mcp@latest` (v1.1.0, 16 tools, patchright browser)
- Qdrant collection: `research_results`

## Testing Results (2026-02-07)

### Perplexity Search — PASSED
- `sonar` model: 510 tokens, 6 citations
- `sonar-pro` model: 582 tokens, 10 citations

### Full Pipeline — PASSED
- Perplexity search → OpenAI embeddings → Qdrant store → semantic search
- 3 results stored, semantic similarity ranking works
- Project filter by `project_id` works
- Qdrant-client v1.16+ compatibility fixes applied

### NotebookLM MCP — PASSED
- Server: Connected, 16 tools, authenticated=true
- `setup_auth` — Google auth via Chrome/WSLg (patchright browser)
- `get_health`, `list_notebooks`, `add_notebook` — work correctly
- `ask_question` — PASSED (tested with notebook_id and notebook_url)
  - Session-based: reuse `session_id` for contextual follow-up questions
  - Detailed answers with numbered citations from notebook sources
- Library: 1 notebook registered ("Enterprise Ontology Infrastructure")
  - Topics: enterprise ontology, PowerBI, OntoGuard, Fabric IQ, MCP
  - ID: `enterprise-ontology-infrastruc`

### NotebookLM MCP Usage Flow
1. Create notebook on https://notebooklm.google.com (add sources)
2. Copy notebook URL: `https://notebooklm.google.com/notebook/xxx`
3. Register: `add_notebook(url=NOTEBOOK_URL, name=..., topics=[...])`
4. Query: `ask_question(question=..., notebook_id=...)`

### NotebookLM MCP Env Vars
```bash
HEADLESS=false              # Show Chrome window (for auth)
DISPLAY=:0                  # WSLg display
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome  # System Chrome (critical for WSL)
NOTEBOOK_URL=               # Default notebook URL (optional)
```

### MCP SDK v1.26.0 Notes
- Protocol version: `2025-11-25`
- Transport: NDJSON (newline-delimited JSON), NOT Content-Length headers
- patchright v1.57.0 (Playwright fork with anti-detection)
