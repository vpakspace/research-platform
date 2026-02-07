"""Research Pipeline — orchestrates search, storage, and retrieval."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import PerplexityModel, ResearchProject, SearchResult
from .perplexity_client import PerplexityClient
from .storage import ResearchStorage

logger = logging.getLogger(__name__)


class ResearchPipeline:
    """Orchestrator that connects Perplexity search with Qdrant storage."""

    def __init__(self, perplexity: PerplexityClient, storage: ResearchStorage):
        self.perplexity = perplexity
        self.storage = storage

    def quick_search(
        self,
        query: str,
        model: str = PerplexityModel.SONAR_PRO,
        project_id: Optional[str] = None,
    ) -> SearchResult:
        """
        Perform a quick search and store the result.

        Args:
            query: Search query.
            model: Perplexity model to use.
            project_id: Optional project to associate result with.

        Returns:
            SearchResult with answer and citations.
        """
        result = self.perplexity.search(query, model=model)
        result.project_id = project_id

        try:
            self.storage.store_result(result, project_id=project_id)
        except Exception as e:
            logger.warning("Failed to store result: %s", e)

        return result

    def deep_research(
        self,
        topic: str,
        questions: List[str],
        project_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform multi-query deep research on a topic.

        Args:
            topic: The research topic (used as context).
            questions: List of specific questions.
            project_id: Optional project to associate results with.

        Returns:
            List of SearchResults for each question.
        """
        results = []
        for question in questions:
            full_query = f"[Research topic: {topic}] {question}"
            result = self.perplexity.search(
                full_query, model=PerplexityModel.SONAR_PRO
            )
            result.project_id = project_id

            try:
                self.storage.store_result(result, project_id=project_id)
            except Exception as e:
                logger.warning("Failed to store result: %s", e)

            results.append(result)

        logger.info("Deep research completed: %d questions on '%s'", len(results), topic)
        return results

    def find_related(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Find related past research using semantic search.

        Args:
            query: Query to find related results for.
            limit: Maximum number of results.

        Returns:
            List of semantically similar past results.
        """
        return self.storage.search_similar(query, limit=limit)

    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a research project.

        Args:
            project_id: The project ID.

        Returns:
            Summary dict with counts and totals.
        """
        results = self.storage.get_project_results(project_id)
        total_tokens = sum(r.tokens_used for r in results)
        total_citations = sum(len(r.citations) for r in results)
        models_used = list({r.model_used for r in results})

        return {
            "project_id": project_id,
            "total_queries": len(results),
            "total_tokens": total_tokens,
            "total_citations": total_citations,
            "models_used": models_used,
        }

    def get_history(
        self,
        project_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[SearchResult]:
        """
        Get search history, optionally filtered by project.

        Args:
            project_id: Optional project filter.
            limit: Maximum number of results.

        Returns:
            List of SearchResults.
        """
        if project_id:
            return self.storage.get_project_results(project_id, limit=limit)
        # Without project filter, return recent via scroll
        return self.storage.get_project_results("", limit=0)[:0]  # empty if no filter
