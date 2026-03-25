"""Stats API router — KG overview statistics."""

from fastapi import APIRouter
from app.services.neo4j_service import Neo4jService

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """Get KG overview: node/edge counts, type distributions."""
    return Neo4jService.get_stats()
