"""Tests for Qdrant storage layer."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from research_platform.storage import ResearchStorage
from research_platform.models import Citation, SearchResult


class TestResearchStorageInit:
    def test_init_without_key_raises(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            ResearchStorage(openai_api_key="")

    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_init_success(self, mock_openai, mock_qdrant):
        s = ResearchStorage(openai_api_key="sk-test")
        assert s.collection_name == "research_results"
        assert s.vector_size == 1536

    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_custom_params(self, mock_openai, mock_qdrant):
        s = ResearchStorage(
            qdrant_url="http://custom:6334",
            collection_name="custom_coll",
            openai_api_key="sk-test",
            vector_size=768,
        )
        assert s.collection_name == "custom_coll"
        assert s.vector_size == 768


class TestEnsureCollection:
    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_creates_if_not_exists(self, mock_openai, mock_qdrant_cls):
        mock_qdrant = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_qdrant.get_collections.return_value = mock_collections
        mock_qdrant_cls.return_value = mock_qdrant

        s = ResearchStorage(openai_api_key="sk-test")
        s.ensure_collection()

        mock_qdrant.create_collection.assert_called_once()

    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_skips_if_exists(self, mock_openai, mock_qdrant_cls):
        mock_qdrant = MagicMock()
        mock_coll = MagicMock()
        mock_coll.name = "research_results"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_coll]
        mock_qdrant.get_collections.return_value = mock_collections
        mock_qdrant_cls.return_value = mock_qdrant

        s = ResearchStorage(openai_api_key="sk-test")
        s.ensure_collection()

        mock_qdrant.create_collection.assert_not_called()


class TestEmbed:
    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_embed_calls_openai(self, mock_openai_cls, mock_qdrant):
        mock_openai = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_resp = MagicMock()
        mock_resp.data = [mock_embedding]
        mock_openai.embeddings.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_openai

        s = ResearchStorage(openai_api_key="sk-test")
        result = s._embed("test text")

        assert len(result) == 1536
        mock_openai.embeddings.create.assert_called_once()


class TestStoreResult:
    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_store_success(self, mock_openai_cls, mock_qdrant_cls):
        mock_openai = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_resp = MagicMock()
        mock_resp.data = [mock_embedding]
        mock_openai.embeddings.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_openai

        mock_qdrant = MagicMock()
        mock_coll = MagicMock()
        mock_coll.name = "research_results"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_coll]
        mock_qdrant.get_collections.return_value = mock_collections
        mock_qdrant_cls.return_value = mock_qdrant

        s = ResearchStorage(openai_api_key="sk-test")
        result = SearchResult(query="q", answer="a")
        point_id = s.store_result(result)

        assert point_id == result.id
        mock_qdrant.upsert.assert_called_once()


class TestSearchSimilar:
    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_search_returns_results(self, mock_openai_cls, mock_qdrant_cls):
        mock_openai = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_resp = MagicMock()
        mock_resp.data = [mock_embedding]
        mock_openai.embeddings.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_openai

        mock_hit = MagicMock()
        mock_hit.payload = {
            "id": "abc",
            "query": "test query",
            "answer": "test answer",
            "citations": [],
            "provider": "perplexity",
            "model_used": "sonar-pro",
            "tokens_used": 100,
        }

        mock_qdrant = MagicMock()
        mock_coll = MagicMock()
        mock_coll.name = "research_results"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_coll]
        mock_qdrant.get_collections.return_value = mock_collections
        mock_query_response = MagicMock()
        mock_query_response.points = [mock_hit]
        mock_qdrant.query_points.return_value = mock_query_response
        mock_qdrant_cls.return_value = mock_qdrant

        s = ResearchStorage(openai_api_key="sk-test")
        results = s.search_similar("test")

        assert len(results) == 1
        assert results[0].query == "test query"

    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_search_empty(self, mock_openai_cls, mock_qdrant_cls):
        mock_openai = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_resp = MagicMock()
        mock_resp.data = [mock_embedding]
        mock_openai.embeddings.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_openai

        mock_qdrant = MagicMock()
        mock_coll = MagicMock()
        mock_coll.name = "research_results"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_coll]
        mock_qdrant.get_collections.return_value = mock_collections
        mock_query_response = MagicMock()
        mock_query_response.points = []
        mock_qdrant.query_points.return_value = mock_query_response
        mock_qdrant_cls.return_value = mock_qdrant

        s = ResearchStorage(openai_api_key="sk-test")
        results = s.search_similar("nothing")

        assert results == []


class TestGetStats:
    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_stats_success(self, mock_openai, mock_qdrant_cls):
        mock_qdrant = MagicMock()
        mock_info = MagicMock()
        mock_info.points_count = 42
        mock_info.indexed_vectors_count = 42
        mock_info.status = "green"
        mock_qdrant.get_collection.return_value = mock_info
        mock_qdrant_cls.return_value = mock_qdrant

        s = ResearchStorage(openai_api_key="sk-test")
        stats = s.get_stats()

        assert stats["points_count"] == 42

    @patch("research_platform.storage.QdrantClient")
    @patch("research_platform.storage.openai.OpenAI")
    def test_stats_not_found(self, mock_openai, mock_qdrant_cls):
        mock_qdrant = MagicMock()
        mock_qdrant.get_collection.side_effect = Exception("Not found")
        mock_qdrant_cls.return_value = mock_qdrant

        s = ResearchStorage(openai_api_key="sk-test")
        stats = s.get_stats()

        assert stats["points_count"] == 0
        assert stats["status"] == "not_found"
