"""Qdrant storage layer with OpenAI embeddings for research results."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .models import Citation, SearchResult

logger = logging.getLogger(__name__)


class ResearchStorage:
    """Storage backend using Qdrant for vector search and OpenAI for embeddings."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "research_results",
        openai_api_key: Optional[str] = None,
        vector_size: int = 1536,
    ):
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._openai = openai.OpenAI(api_key=openai_api_key)
        self._qdrant = QdrantClient(url=qdrant_url)

    def ensure_collection(self) -> None:
        """Create collection if it does not exist."""
        collections = self._qdrant.get_collections().collections
        names = [c.name for c in collections]

        if self.collection_name not in names:
            self._qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection: %s", self.collection_name)

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        resp = self._openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding

    def store_result(self, result: SearchResult, project_id: Optional[str] = None) -> str:
        """
        Store a search result in Qdrant.

        Args:
            result: The SearchResult to store.
            project_id: Optional project ID to associate with.

        Returns:
            The point ID.
        """
        self.ensure_collection()

        text = f"{result.query}\n\n{result.answer}"
        vector = self._embed(text)

        payload: Dict[str, Any] = {
            "id": result.id,
            "query": result.query,
            "answer": result.answer,
            "citations": [c.model_dump() for c in result.citations],
            "provider": result.provider,
            "model_used": result.model_used,
            "created_at": result.created_at.isoformat(),
            "tokens_used": result.tokens_used,
        }
        if project_id:
            payload["project_id"] = project_id

        import uuid as _uuid

        try:
            point_id = str(_uuid.UUID(result.id))
        except ValueError:
            point_id = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, result.id))

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )
        self._qdrant.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        logger.info("Stored result %s for query: %s", result.id, result.query[:60])
        return result.id

    def search_similar(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Search for similar past results using semantic search.

        Args:
            query: The query to find similar results for.
            limit: Maximum number of results.

        Returns:
            List of similar SearchResults.
        """
        self.ensure_collection()

        vector = self._embed(query)
        response = self._qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
        )
        hits = response.points

        results = []
        for hit in hits:
            p = hit.payload or {}
            citations = [Citation(**c) for c in p.get("citations", [])]
            results.append(
                SearchResult(
                    id=p.get("id", ""),
                    query=p.get("query", ""),
                    answer=p.get("answer", ""),
                    citations=citations,
                    provider=p.get("provider", ""),
                    model_used=p.get("model_used", ""),
                    tokens_used=p.get("tokens_used", 0),
                    project_id=p.get("project_id"),
                )
            )
        return results

    def get_project_results(self, project_id: str, limit: int = 100) -> List[SearchResult]:
        """Get all results for a given project."""
        self.ensure_collection()

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        hits = self._qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project_id),
                    )
                ]
            ),
            limit=limit,
        )

        results = []
        for point in hits[0]:
            p = point.payload or {}
            citations = [Citation(**c) for c in p.get("citations", [])]
            results.append(
                SearchResult(
                    id=p.get("id", ""),
                    query=p.get("query", ""),
                    answer=p.get("answer", ""),
                    citations=citations,
                    provider=p.get("provider", ""),
                    model_used=p.get("model_used", ""),
                    tokens_used=p.get("tokens_used", 0),
                    project_id=p.get("project_id"),
                )
            )
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            info = self._qdrant.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "points_count": info.points_count,
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
                "status": str(info.status),
            }
        except Exception:
            return {
                "collection": self.collection_name,
                "points_count": 0,
                "vectors_count": 0,
                "status": "not_found",
            }
