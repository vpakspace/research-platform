"""Tests for Pydantic models."""

import pytest
from datetime import datetime

from research_platform.models import (
    Citation,
    PerplexityModel,
    ResearchProject,
    SearchResult,
)


class TestCitation:
    def test_create_with_url_only(self):
        c = Citation(url="https://example.com")
        assert c.url == "https://example.com"
        assert c.title == ""
        assert c.snippet == ""

    def test_create_full(self):
        c = Citation(title="Test", url="https://test.com", snippet="A snippet")
        assert c.title == "Test"
        assert c.snippet == "A snippet"

    def test_serialization(self):
        c = Citation(title="T", url="https://x.com", snippet="S")
        d = c.model_dump()
        assert d["title"] == "T"
        assert d["url"] == "https://x.com"


class TestPerplexityModel:
    def test_enum_values(self):
        assert PerplexityModel.SONAR.value == "sonar"
        assert PerplexityModel.SONAR_PRO.value == "sonar-pro"
        assert PerplexityModel.SONAR_REASONING_PRO.value == "sonar-reasoning-pro"
        assert PerplexityModel.SONAR_DEEP_RESEARCH.value == "sonar-deep-research"

    def test_enum_from_value(self):
        m = PerplexityModel("sonar-pro")
        assert m == PerplexityModel.SONAR_PRO


class TestSearchResult:
    def test_create_minimal(self):
        r = SearchResult(query="test", answer="answer")
        assert r.query == "test"
        assert r.answer == "answer"
        assert r.provider == "perplexity"
        assert r.model_used == "sonar-pro"
        assert r.tokens_used == 0
        assert r.citations == []
        assert r.id  # auto-generated

    def test_create_full(self):
        r = SearchResult(
            query="q",
            answer="a",
            citations=[Citation(url="https://x.com")],
            provider="perplexity",
            model_used="sonar",
            tokens_used=100,
            project_id="proj1",
        )
        assert len(r.citations) == 1
        assert r.tokens_used == 100
        assert r.project_id == "proj1"

    def test_unique_ids(self):
        r1 = SearchResult(query="q1", answer="a1")
        r2 = SearchResult(query="q2", answer="a2")
        assert r1.id != r2.id

    def test_serialization(self):
        r = SearchResult(query="q", answer="a", tokens_used=50)
        d = r.model_dump()
        assert d["query"] == "q"
        assert d["tokens_used"] == 50

    def test_created_at_auto(self):
        r = SearchResult(query="q", answer="a")
        assert isinstance(r.created_at, datetime)


class TestResearchProject:
    def test_create_minimal(self):
        p = ResearchProject(name="Test Project")
        assert p.name == "Test Project"
        assert p.description == ""
        assert p.tags == []
        assert p.queries == []
        assert p.results == []

    def test_create_full(self):
        p = ResearchProject(
            name="AI Research",
            description="Exploring AI",
            tags=["ai", "ml"],
            queries=["What is AI?"],
        )
        assert p.description == "Exploring AI"
        assert len(p.tags) == 2
        assert len(p.queries) == 1

    def test_unique_ids(self):
        p1 = ResearchProject(name="P1")
        p2 = ResearchProject(name="P2")
        assert p1.id != p2.id
