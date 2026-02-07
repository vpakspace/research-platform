"""Tests for MCP Server."""

import pytest
from unittest.mock import MagicMock, patch

from research_platform.mcp_server import load_config, _research_search_impl, _research_find_impl


class TestLoadConfig:
    @patch("research_platform.mcp_server.Path")
    def test_load_existing_config(self, mock_path):
        mock_path_inst = MagicMock()
        mock_path_inst.exists.return_value = True
        mock_path.return_value = mock_path_inst

        with patch("builtins.open", create=True) as mock_open:
            with patch("research_platform.mcp_server.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"perplexity": {"timeout": 30}}
                config = load_config()

        assert config.get("perplexity", {}).get("timeout") == 30

    @patch("research_platform.mcp_server.Path")
    def test_load_missing_config_returns_defaults(self, mock_path):
        mock_path_inst = MagicMock()
        mock_path_inst.exists.return_value = False
        mock_path.return_value = mock_path_inst

        with patch.dict("os.environ", {}, clear=False):
            config = load_config()

        assert config == {}


class TestResearchSearchTool:
    @patch("research_platform.mcp_server.get_pipeline")
    def test_search_returns_dict(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "abc"
        mock_result.query = "What is RAG?"
        mock_result.answer = "RAG is..."
        mock_result.citations = []
        mock_result.model_used = "sonar-pro"
        mock_result.tokens_used = 100
        mock_get_pipeline.return_value = mock_pipeline
        mock_pipeline.quick_search.return_value = mock_result

        result = _research_search_impl("What is RAG?")

        assert result["id"] == "abc"
        assert result["answer"] == "RAG is..."
        assert result["tokens_used"] == 100

    @patch("research_platform.mcp_server.get_pipeline")
    def test_search_with_custom_model(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "x"
        mock_result.query = "q"
        mock_result.answer = "a"
        mock_result.citations = []
        mock_result.model_used = "sonar"
        mock_result.tokens_used = 50
        mock_get_pipeline.return_value = mock_pipeline
        mock_pipeline.quick_search.return_value = mock_result

        result = _research_search_impl("q", model="sonar")

        mock_pipeline.quick_search.assert_called_once_with("q", model="sonar")


class TestResearchFindTool:
    @patch("research_platform.mcp_server.get_pipeline")
    def test_find_returns_results(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "r1"
        mock_result.query = "RAG"
        mock_result.answer = "RAG answer that is long enough"
        mock_result.citations = []
        mock_result.model_used = "sonar-pro"
        mock_get_pipeline.return_value = mock_pipeline
        mock_pipeline.find_related.return_value = [mock_result]

        result = _research_find_impl("RAG")

        assert result["total_found"] == 1
        assert len(result["results"]) == 1

    @patch("research_platform.mcp_server.get_pipeline")
    def test_find_empty(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline
        mock_pipeline.find_related.return_value = []

        result = _research_find_impl("nothing")

        assert result["total_found"] == 0
        assert result["results"] == []
