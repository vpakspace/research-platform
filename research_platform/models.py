"""Pydantic models for the Research Intelligence Platform."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PerplexityModel(str, Enum):
    """Available Perplexity AI models."""

    SONAR = "sonar"
    SONAR_PRO = "sonar-pro"
    SONAR_REASONING_PRO = "sonar-reasoning-pro"
    SONAR_DEEP_RESEARCH = "sonar-deep-research"


class Citation(BaseModel):
    """A single citation/source from search results."""

    title: str = ""
    url: str
    snippet: str = ""


class SearchResult(BaseModel):
    """Result of a single search query."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    provider: str = "perplexity"
    model_used: str = "sonar-pro"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tokens_used: int = 0
    project_id: Optional[str] = None


class ResearchProject(BaseModel):
    """A research project containing multiple queries and results."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    queries: List[str] = Field(default_factory=list)
    results: List[SearchResult] = Field(default_factory=list)
