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
- NotebookLM MCP: `notebooklm-mcp@latest`
- Qdrant collection: `research_results`
