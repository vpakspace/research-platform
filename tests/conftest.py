"""Shared fixtures for tests."""

import pytest


@pytest.fixture
def sample_search_result():
    """Sample search result for testing."""
    return {
        "query": "What is RAG?",
        "answer": "RAG stands for Retrieval-Augmented Generation...",
        "citations": [
            {"title": "RAG Paper", "url": "https://arxiv.org/abs/2005.11401", "snippet": "..."},
        ],
        "model_used": "sonar-pro",
        "tokens_used": 150,
    }


@pytest.fixture
def sample_project():
    """Sample research project for testing."""
    return {
        "name": "AI Research",
        "description": "Research on modern AI techniques",
        "tags": ["ai", "ml", "rag"],
    }
