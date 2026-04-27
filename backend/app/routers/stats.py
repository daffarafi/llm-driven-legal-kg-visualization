"""Stats API router — KG overview statistics with optional document filtering."""

from fastapi import APIRouter, Query
from app.services.neo4j_service import Neo4jService

router = APIRouter()


@router.get("/stats")
async def get_stats(
    doc_id: str | None = Query(None, description="Filter stats to a specific source_document_id"),
):
    """Get KG overview: node/edge counts, type distributions. Optionally filtered by document."""
    if doc_id:
        return Neo4jService.get_stats(doc_id=doc_id)
    return Neo4jService.get_stats()
