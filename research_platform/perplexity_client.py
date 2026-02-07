"""Async HTTP client for Perplexity AI API."""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from .models import Citation, PerplexityModel, SearchResult

logger = logging.getLogger(__name__)


class PerplexityClient:
    """Async client for Perplexity AI search API."""

    BASE_URL = "https://api.perplexity.ai"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        min_request_interval: float = 1.0,
    ):
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self._min_interval = min_request_interval
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _parse_citations(self, response_data: dict) -> list[Citation]:
        """Extract citations from Perplexity API response."""
        citations = []
        raw_citations = response_data.get("citations", [])

        if isinstance(raw_citations, list):
            for item in raw_citations:
                if isinstance(item, str):
                    citations.append(Citation(url=item))
                elif isinstance(item, dict):
                    citations.append(
                        Citation(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("snippet", ""),
                        )
                    )
        return citations

    def search(
        self,
        query: str,
        model: str = PerplexityModel.SONAR_PRO,
    ) -> SearchResult:
        """
        Perform a search query via Perplexity API.

        Args:
            query: The search query.
            model: Model to use (default: sonar-pro).

        Returns:
            SearchResult with answer and citations.
        """
        if isinstance(model, PerplexityModel):
            model = model.value

        self._rate_limit()

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info("Perplexity search: model=%s, query=%s", model, query[:80])

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        answer = ""
        tokens_used = 0

        choices = data.get("choices", [])
        if choices:
            answer = choices[0].get("message", {}).get("content", "")

        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)

        citations = self._parse_citations(data)

        return SearchResult(
            query=query,
            answer=answer,
            citations=citations,
            provider="perplexity",
            model_used=model,
            tokens_used=tokens_used,
        )

    def research(self, query: str) -> SearchResult:
        """Deep research using sonar-deep-research model."""
        return self.search(query, model=PerplexityModel.SONAR_DEEP_RESEARCH)

    def reason(self, query: str) -> SearchResult:
        """Reasoning search using sonar-reasoning-pro model."""
        return self.search(query, model=PerplexityModel.SONAR_REASONING_PRO)
