"""Graph API router — subgraph retrieval, node detail, expand."""

from fastapi import APIRouter, Query, HTTPException
from app.services.neo4j_service import Neo4jService

router = APIRouter()


@router.get("/graph")
async def get_graph(
    types: str | None = Query(None, description="Comma-separated node types"),
    relations: str | None = Query(None, description="Comma-separated relation types"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get subgraph with optional node type and relation type filters."""
    node_types = types.split(",") if types else None
    relation_types = relations.split(",") if relations else None

    result = Neo4jService.get_graph(
        node_types=node_types,
        relation_types=relation_types,
        limit=limit,
    )
    return result


@router.get("/node/{node_id}")
async def get_node(node_id: str):
    """Get single node with all properties and relations."""
    result = Neo4jService.get_node(node_id)
    if not result:
        raise HTTPException(status_code=404, detail="Node not found")
    return result


@router.get("/node/{node_id}/subgraph")
async def get_node_subgraph(
    node_id: str,
    depth: int = Query(1, ge=1, le=3),
):
    """Get subgraph around a node up to given depth."""
    result = Neo4jService.get_node_subgraph(node_id, depth)
    return result
