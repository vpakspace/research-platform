"""Tests for Perplexity API client."""

import pytest
from unittest.mock import patch, MagicMock

from research_platform.perplexity_client import PerplexityClient
from research_platform.models import PerplexityModel


class TestPerplexityClientInit:
    def test_init_with_key(self):
        c = PerplexityClient(api_key="test-key")
        assert c.api_key == "test-key"
        assert c.base_url == "https://api.perplexity.ai"

    def test_init_without_key_raises(self):
        with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
            PerplexityClient(api_key="")

    def test_custom_base_url(self):
        c = PerplexityClient(api_key="k", base_url="http://localhost:8080")
        assert c.base_url == "http://localhost:8080"

    def test_custom_timeout(self):
        c = PerplexityClient(api_key="k", timeout=30.0)
        assert c.timeout == 30.0


class TestParseCitations:
    def test_parse_string_urls(self):
        c = PerplexityClient(api_key="k")
        data = {"citations": ["https://a.com", "https://b.com"]}
        result = c._parse_citations(data)
        assert len(result) == 2
        assert result[0].url == "https://a.com"

    def test_parse_dict_citations(self):
        c = PerplexityClient(api_key="k")
        data = {
            "citations": [
                {"title": "Title", "url": "https://x.com", "snippet": "Snip"}
            ]
        }
        result = c._parse_citations(data)
        assert len(result) == 1
        assert result[0].title == "Title"
        assert result[0].snippet == "Snip"

    def test_parse_empty_citations(self):
        c = PerplexityClient(api_key="k")
        result = c._parse_citations({})
        assert result == []

    def test_parse_no_citations_key(self):
        c = PerplexityClient(api_key="k")
        result = c._parse_citations({"other": "data"})
        assert result == []


class TestSearch:
    @patch("research_platform.perplexity_client.httpx.Client")
    def test_search_success(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "RAG is..."}}],
            "usage": {"total_tokens": 150},
            "citations": ["https://arxiv.org"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = PerplexityClient(api_key="test-key", min_request_interval=0)
        result = client.search("What is RAG?")

        assert result.query == "What is RAG?"
        assert result.answer == "RAG is..."
        assert result.tokens_used == 150
        assert len(result.citations) == 1
        assert result.provider == "perplexity"
        assert result.model_used == "sonar-pro"

    @patch("research_platform.perplexity_client.httpx.Client")
    def test_search_custom_model(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Answer"}}],
            "usage": {"total_tokens": 50},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = PerplexityClient(api_key="test-key", min_request_interval=0)
        result = client.search("query", model="sonar")

        assert result.model_used == "sonar"

    @patch("research_platform.perplexity_client.httpx.Client")
    def test_search_empty_response(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [], "usage": {}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = PerplexityClient(api_key="test-key", min_request_interval=0)
        result = client.search("query")

        assert result.answer == ""
        assert result.tokens_used == 0


class TestConvenienceMethods:
    @patch("research_platform.perplexity_client.httpx.Client")
    def test_research_uses_deep_model(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Deep research"}}],
            "usage": {"total_tokens": 300},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = PerplexityClient(api_key="test-key", min_request_interval=0)
        result = client.research("topic")

        assert result.model_used == "sonar-deep-research"

    @patch("research_platform.perplexity_client.httpx.Client")
    def test_reason_uses_reasoning_model(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Reasoning"}}],
            "usage": {"total_tokens": 200},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = PerplexityClient(api_key="test-key", min_request_interval=0)
        result = client.reason("problem")

        assert result.model_used == "sonar-reasoning-pro"
