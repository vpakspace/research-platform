# Research Intelligence Platform

Unified research pipeline combining **Perplexity AI**, **NotebookLM**, and **Claude Code** via Model Context Protocol (MCP).

Search the web with Perplexity Pro, store results in Qdrant vector database with OpenAI embeddings, find related past research via semantic search, and access everything through Claude Code MCP tools or Streamlit UI.

## Architecture

```
                          +-----------------+
                          |  Perplexity AI  |
                          |  (sonar/pro/    |
                          |   deep-research)|
                          +--------+--------+
                                   |
+----------------+        +--------v--------+        +-----------------+
|                |        |                 |        |                 |
| Streamlit UI   +------->+ Research        +------->+ Qdrant          |
| (port 8503)    |        | Pipeline        |        | Vector DB       |
|                |        |                 |        | (port 6333)     |
+----------------+        +--------+--------+        +-----------------+
                                   |                         ^
+----------------+        +--------v--------+                |
|                |        |                 |    OpenAI       |
| Claude Code    +------->+ FastMCP Server  |  embeddings    |
| (MCP client)   |        | (5 tools)       +----------------+
|                |        |                 |  text-embedding-
+----------------+        +-----------------+  3-small (1536d)

+----------------+
| NotebookLM     |  Curated research notebooks
| (Google)       |  queried via NotebookLM MCP
+----------------+
```

## Features

- **Web Search** — Search via Perplexity AI with 4 model tiers (sonar, sonar-pro, sonar-reasoning-pro, sonar-deep-research)
- **Automatic Storage** — Every search result is embedded and stored in Qdrant for future retrieval
- **Semantic Search** — Find related past research using cosine similarity over OpenAI embeddings
- **Research Projects** — Organize multi-query deep research into named projects with tags
- **MCP Integration** — 5 MCP tools accessible from Claude Code for seamless AI-assisted research
- **NotebookLM** — Query curated Google NotebookLM notebooks for deep document analysis
- **Streamlit UI** — 4-tab web interface (Search, Projects, Knowledge Base, History)

## Quick Start

### Prerequisites

- Python 3.9+
- [Qdrant](https://qdrant.tech/) running locally (Docker recommended)
- Perplexity API key ([pplx.ai](https://www.perplexity.ai/settings/api))
- OpenAI API key ([platform.openai.com](https://platform.openai.com/api-keys))

### 1. Clone and Install

```bash
git clone https://github.com/vpakspace/research-platform.git
cd research-platform
pip install -e ".[dev]"
```

### 2. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Or if already created:
docker start qdrant
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
#   PERPLEXITY_API_KEY=pplx-xxxx
#   OPENAI_API_KEY=sk-xxxx
#   QDRANT_URL=http://localhost:6333
```

### 4. Run

**Streamlit UI:**

```bash
./run_streamlit.sh
# Open http://localhost:8503
```

**MCP Server (for Claude Code):**

```bash
python -m research_platform
```

## MCP Tools

The platform exposes 5 tools via [FastMCP](https://github.com/jlowin/fastmcp) for use in Claude Code:

| Tool | Description | Parameters |
|------|-------------|------------|
| `research_search` | Search the web via Perplexity AI and store results | `query`, `model` (default: sonar-pro) |
| `research_deep` | Multi-query deep research on a topic | `topic`, `questions` (list) |
| `research_find` | Semantic search over past research results in Qdrant | `query`, `limit` (default: 5) |
| `research_history` | Get search history, optionally filtered by project | `project_id`, `limit` |
| `research_stats` | Get storage statistics (collection size, status) | — |

### Claude Code MCP Configuration

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "research-platform": {
      "command": "python",
      "args": ["-m", "research_platform"],
      "cwd": "/path/to/research-platform",
      "env": {
        "PERPLEXITY_API_KEY": "pplx-xxxx",
        "OPENAI_API_KEY": "sk-xxxx"
      }
    }
  }
}
```

## Companion MCP Servers

The platform works alongside two companion MCP servers:

### Perplexity MCP

Direct access to Perplexity AI from Claude Code.

```bash
claude mcp add perplexity -s user -e PERPLEXITY_API_KEY=pplx-xxxx -- \
  npx -y @perplexity-ai/mcp-server
```

### NotebookLM MCP

Query curated research notebooks on Google NotebookLM via browser automation.

```bash
claude mcp add notebooklm -s user -- npx -y notebooklm-mcp@latest
```

**Usage flow:**
1. Create a notebook at https://notebooklm.google.com (add sources)
2. Authenticate: `setup_auth` (opens Chrome for Google login)
3. Register: `add_notebook(url=NOTEBOOK_URL, name=..., topics=[...])`
4. Query: `ask_question(question=..., notebook_id=...)`

**Env vars for WSL:**
```bash
HEADLESS=false
DISPLAY=:0
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome
```

## Project Structure

```
research-platform/
├── research_platform/           # Core Python package
│   ├── __init__.py              # Package init, version
│   ├── __main__.py              # Entry point: python -m research_platform
│   ├── models.py                # Pydantic models (SearchResult, Citation, etc.)
│   ├── perplexity_client.py     # Async HTTP client for Perplexity API
│   ├── storage.py               # Qdrant + OpenAI embeddings storage layer
│   ├── pipeline.py              # Research orchestrator (search → store → find)
│   └── mcp_server.py            # FastMCP server with 5 tools
├── config/
│   └── platform_config.yaml     # Platform configuration (models, URLs, etc.)
├── tests/                       # Test suite (51 tests, 81% coverage)
│   ├── conftest.py              # Shared fixtures
│   ├── test_models.py           # Model validation tests
│   ├── test_perplexity_client.py# Client tests (mocked HTTP)
│   ├── test_storage.py          # Storage layer tests (mocked Qdrant/OpenAI)
│   ├── test_pipeline.py         # Pipeline integration tests
│   └── test_mcp_server.py       # MCP tool implementation tests
├── streamlit_app.py             # Streamlit web UI (4 tabs)
├── run_streamlit.sh             # Streamlit launch script
├── pyproject.toml               # Build config, dependencies, tool settings
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Dev dependencies (pytest, black, mypy)
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
└── CLAUDE.md                    # Claude Code project memory
```

## Components

### Models (`models.py`)

Pydantic v2 data models:

- **`PerplexityModel`** — Enum of available models: `sonar`, `sonar-pro`, `sonar-reasoning-pro`, `sonar-deep-research`
- **`Citation`** — Source reference with `title`, `url`, `snippet`
- **`SearchResult`** — Query result with answer, citations, token usage, timestamps
- **`ResearchProject`** — Named research project with tags, queries, and collected results

### Perplexity Client (`perplexity_client.py`)

Synchronous HTTP client for Perplexity API:

- Built on `httpx` with configurable timeout
- Rate limiting (default: 1 request/sec via `time.monotonic()`)
- Citation parsing from API response (URL strings and dict formats)
- Convenience methods: `search()`, `research()` (deep-research), `reason()` (reasoning-pro)

### Storage (`storage.py`)

Vector storage layer combining Qdrant and OpenAI:

- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Vector DB**: Qdrant with cosine distance
- **Auto-collection**: Creates collection on first use
- **Operations**: `store_result()`, `search_similar()`, `get_project_results()`, `get_stats()`

### Pipeline (`pipeline.py`)

Research orchestrator connecting search and storage:

- **`quick_search()`** — Search + auto-store
- **`deep_research()`** — Multi-question research on a topic with context prefix
- **`find_related()`** — Semantic search over past results
- **`get_project_summary()`** — Aggregated project statistics
- **`get_history()`** — Filtered search history

### MCP Server (`mcp_server.py`)

FastMCP server exposing 5 tools:

- YAML configuration loading
- Lazy pipeline initialization (singleton)
- API key validation at startup
- Testable implementation functions separated from FastMCP decorators

### Streamlit UI (`streamlit_app.py`)

4-tab web interface on port 8503:

| Tab | Description |
|-----|-------------|
| **Search** | Single query with model selector, answer display, source links |
| **Research Projects** | Create projects, run multi-question research, view results |
| **Knowledge Base** | Semantic search over stored results, storage stats |
| **History** | Session search history table with JSON export |

## Configuration

### `config/platform_config.yaml`

```yaml
perplexity:
  default_model: "sonar-pro"
  timeout: 60
  min_request_interval: 1.0

qdrant:
  url: "http://localhost:6333"
  collection_name: "research_results"
  vector_size: 1536

embeddings:
  model: "text-embedding-3-small"
  dimensions: 1536

ui:
  port: 8503
  title: "Research Intelligence Platform"

logging:
  level: "INFO"
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PERPLEXITY_API_KEY` | Yes | — | Perplexity AI API key |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for embeddings |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant server URL |
| `RESEARCH_PLATFORM_CONFIG` | No | `config/platform_config.yaml` | Config file path |

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=research_platform --cov-report=term-missing

# Current status: 51 passed, 81% coverage
```

All external dependencies (Perplexity API, OpenAI, Qdrant) are mocked in tests.

## Research Workflow

The recommended workflow for using the platform:

```
1. Discover    →  Perplexity Pro (Deep Research) for web search
2. Curate      →  Save key findings to Google NotebookLM notebooks
3. Store       →  Results auto-stored in Qdrant via the pipeline
4. Retrieve    →  Semantic search finds related past research
5. Context     →  Claude Code auto-loads relevant context via auto-context skill
```

### Auto-Context Skill

The companion `auto-context` Claude Code skill (`~/.claude/skills/auto-context/`) automatically:
1. Lists registered NotebookLM notebooks
2. Matches task keywords against notebook topics
3. Queries matching notebooks for relevant context
4. Presents structured context before starting work

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Search API | Perplexity AI | Web search with citations |
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dim vector representations |
| Vector DB | Qdrant | Semantic similarity search |
| MCP Framework | FastMCP | Claude Code tool integration |
| Web UI | Streamlit | Interactive research interface |
| Data Models | Pydantic v2 | Validation and serialization |
| HTTP Client | httpx | Perplexity API communication |
| Config | PyYAML | YAML configuration |
| Notebook AI | Google NotebookLM | Curated document analysis |

## License

MIT

## Author

[@vpakspace](https://github.com/vpakspace)
