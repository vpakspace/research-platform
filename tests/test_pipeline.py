"""Tests for Research Pipeline orchestrator."""

import pytest
from unittest.mock import MagicMock, patch

from research_platform.pipeline import ResearchPipeline
from research_platform.models import SearchResult, Citation


def _make_result(query="q", answer="a", tokens=50):
    return SearchResult(query=query, answer=answer, tokens_used=tokens)


class TestQuickSearch:
    def test_search_and_store(self):
        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_client.search.return_value = _make_result("What is AI?", "AI is...")

        pipeline = ResearchPipeline(perplexity=mock_client, storage=mock_storage)
        result = pipeline.quick_search("What is AI?")

        assert result.query == "What is AI?"
        assert result.answer == "AI is..."
        mock_client.search.assert_called_once()
        mock_storage.store_result.assert_called_once()

    def test_search_stores_with_project_id(self):
        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_client.search.return_value = _make_result()

        pipeline = ResearchPipeline(perplexity=mock_client, storage=mock_storage)
        result = pipeline.quick_search("q", project_id="proj1")

        assert result.project_id == "proj1"
        mock_storage.store_result.assert_called_once_with(result, project_id="proj1")

    def test_search_continues_on_storage_failure(self):
        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_client.search.return_value = _make_result()
        mock_storage.store_result.side_effect = Exception("Qdrant down")

        pipeline = ResearchPipeline(perplexity=mock_client, storage=mock_storage)
        result = pipeline.quick_search("q")

        assert result.answer == "a"  # Search succeeded despite storage failure


class TestDeepResearch:
    def test_multi_query(self):
        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_client.search.side_effect = [
            _make_result("q1", "a1"),
            _make_result("q2", "a2"),
        ]

        pipeline = ResearchPipeline(perplexity=mock_client, storage=mock_storage)
        results = pipeline.deep_research("AI", ["q1", "q2"])

        assert len(results) == 2
        assert mock_client.search.call_count == 2
        assert mock_storage.store_result.call_count == 2

    def test_empty_questions(self):
        mock_client = MagicMock()
        mock_storage = MagicMock()

        pipeline = ResearchPipeline(perplexity=mock_client, storage=mock_storage)
        results = pipeline.deep_research("AI", [])

        assert results == []
        mock_client.search.assert_not_called()


class TestFindRelated:
    def test_delegates_to_storage(self):
        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_storage.search_similar.return_value = [_make_result()]

        pipeline = ResearchPipeline(perplexity=mock_client, storage=mock_storage)
        results = pipeline.find_related("test", limit=3)

        assert len(results) == 1
        mock_storage.search_similar.assert_called_once_with("test", limit=3)


class TestProjectSummary:
    def test_summary_counts(self):
        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_storage.get_project_results.return_value = [
            _make_result("q1", "a1", 100),
            _make_result("q2", "a2", 200),
        ]

        pipeline = ResearchPipeline(perplexity=mock_client, storage=mock_storage)
        summary = pipeline.get_project_summary("proj1")

        assert summary["project_id"] == "proj1"
        assert summary["total_queries"] == 2
        assert summary["total_tokens"] == 300

    def test_summary_empty_project(self):
        mock_client = MagicMock()
        mock_storage = MagicMock()
        mock_storage.get_project_results.return_value = []

        pipeline = ResearchPipeline(perplexity=mock_client, storage=mock_storage)
        summary = pipeline.get_project_summary("empty")

        assert summary["total_queries"] == 0
        assert summary["total_tokens"] == 0
