"""MCP Server for Research Intelligence Platform.

Usage:
    python -m research_platform

    Or configure in MCP client:
    {
        "mcpServers": {
            "research-platform": {
                "command": "python",
                "args": ["-m", "research_platform"],
                "cwd": "/home/vladspace_ubuntu24/research-platform",
                "env": {
                    "PERPLEXITY_API_KEY": "pplx-xxx",
                    "OPENAI_API_KEY": "sk-xxx"
                }
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError("fastmcp is required. Install with: pip install fastmcp")

from .models import PerplexityModel
from .perplexity_client import PerplexityClient
from .storage import ResearchStorage
from .pipeline import ResearchPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_pipeline: Optional[ResearchPipeline] = None
_config: Dict[str, Any] = {}


def load_config() -> Dict[str, Any]:
    """Load platform configuration from YAML."""
    config_path = os.getenv(
        "RESEARCH_PLATFORM_CONFIG",
        str(Path(__file__).parent.parent / "config" / "platform_config.yaml"),
    )

    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info("Config loaded from %s", config_path)
        return config

    logger.warning("Config not found at %s, using defaults", config_path)
    return {}


def get_pipeline() -> ResearchPipeline:
    """Lazy-initialize and return the pipeline singleton."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if not perplexity_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable is required")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    perplexity_conf = _config.get("perplexity", {})
    qdrant_conf = _config.get("qdrant", {})

    client = PerplexityClient(
        api_key=perplexity_key,
        timeout=perplexity_conf.get("timeout", 60),
        min_request_interval=perplexity_conf.get("min_request_interval", 1.0),
    )
    storage = ResearchStorage(
        qdrant_url=qdrant_conf.get("url", "http://localhost:6333"),
        collection_name=qdrant_conf.get("collection_name", "research_results"),
        openai_api_key=openai_key,
        vector_size=qdrant_conf.get("vector_size", 1536),
    )

    _pipeline = ResearchPipeline(perplexity=client, storage=storage)
    return _pipeline


# --- Implementation functions (testable without FastMCP) ---


def _research_search_impl(query: str, model: str = "sonar-pro") -> Dict[str, Any]:
    """Search the web via Perplexity AI and store results."""
    pipeline = get_pipeline()
    result = pipeline.quick_search(query, model=model)
    return {
        "id": result.id,
        "query": result.query,
        "answer": result.answer,
        "citations": [c.model_dump() for c in result.citations],
        "model_used": result.model_used,
        "tokens_used": result.tokens_used,
    }


def _research_deep_impl(topic: str, questions: List[str]) -> Dict[str, Any]:
    """Perform deep multi-query research on a topic."""
    pipeline = get_pipeline()
    results = pipeline.deep_research(topic, questions)
    return {
        "topic": topic,
        "results": [
            {
                "query": r.query,
                "answer": r.answer,
                "citations": [c.model_dump() for c in r.citations],
                "tokens_used": r.tokens_used,
            }
            for r in results
        ],
        "total_questions": len(results),
    }


def _research_find_impl(query: str, limit: int = 5) -> Dict[str, Any]:
    """Semantic search over past research results stored in Qdrant."""
    pipeline = get_pipeline()
    results = pipeline.find_related(query, limit=limit)
    return {
        "query": query,
        "results": [
            {
                "id": r.id,
                "original_query": r.query,
                "answer": r.answer[:300],
                "citations_count": len(r.citations),
                "model_used": r.model_used,
            }
            for r in results
        ],
        "total_found": len(results),
    }


def _research_history_impl(project_id: str = "", limit: int = 20) -> Dict[str, Any]:
    """Get research history, optionally filtered by project."""
    pipeline = get_pipeline()
    if project_id:
        results = pipeline.storage.get_project_results(project_id, limit=limit)
    else:
        results = []

    return {
        "project_id": project_id or "all",
        "results": [
            {
                "id": r.id,
                "query": r.query,
                "answer": r.answer[:200],
                "model_used": r.model_used,
                "tokens_used": r.tokens_used,
            }
            for r in results
        ],
        "total": len(results),
    }


def _research_stats_impl() -> Dict[str, Any]:
    """Get storage statistics for the research platform."""
    pipeline = get_pipeline()
    return pipeline.storage.get_stats()


# --- FastMCP server with tool wrappers ---

mcp = FastMCP("ResearchPlatform")


@mcp.tool()
def research_search(query: str, model: str = "sonar-pro") -> Dict[str, Any]:
    """Search the web via Perplexity AI and store results."""
    return _research_search_impl(query, model=model)


@mcp.tool()
def research_deep(topic: str, questions: List[str]) -> Dict[str, Any]:
    """Perform deep multi-query research on a topic."""
    return _research_deep_impl(topic, questions)


@mcp.tool()
def research_find(query: str, limit: int = 5) -> Dict[str, Any]:
    """Semantic search over past research results stored in Qdrant."""
    return _research_find_impl(query, limit=limit)


@mcp.tool()
def research_history(project_id: str = "", limit: int = 20) -> Dict[str, Any]:
    """Get research history, optionally filtered by project."""
    return _research_history_impl(project_id=project_id, limit=limit)


@mcp.tool()
def research_stats() -> Dict[str, Any]:
    """Get storage statistics for the research platform."""
    return _research_stats_impl()


def main():
    """Entry point for MCP server."""
    global _config

    _config = load_config()

    log_level = _config.get("logging", {}).get("level", "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

    logger.info("Starting Research Intelligence Platform MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
