"""Search API router — keyword and semantic search."""

from fastapi import APIRouter, Query
from app.services.neo4j_service import Neo4jService

router = APIRouter()


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    mode: str = Query("keyword", description="keyword | semantic | hybrid"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search nodes by keyword or semantic similarity."""
    results = Neo4jService.search(q, mode=mode, limit=limit)
    return {"results": results, "total": len(results)}
