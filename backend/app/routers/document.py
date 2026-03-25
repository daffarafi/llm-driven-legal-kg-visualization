"""Document API router — document viewer with hierarchy."""

from fastapi import APIRouter, HTTPException
from app.services.neo4j_service import Neo4jService

router = APIRouter()


@router.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """Get document with hierarchical structure (Bab → Pasal → Ayat)."""
    result = Neo4jService.get_document(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@router.get("/documents")
async def list_documents():
    """List all available documents (UndangUndang nodes)."""
    from app.services.neo4j_service import Neo4jService
    with Neo4jService.get_driver().session() as s:
        results = s.run("""
            MATCH (u:UndangUndang)
            OPTIONAL MATCH (u)-[:MEMUAT]->(b:Bab)
            WITH u, count(b) AS bab_count
            OPTIONAL MATCH (u)-[:MEMUAT]->(:Bab)-[:MEMUAT|MEMILIKI_PASAL]->(p:Pasal)
            RETURN elementId(u) AS id,
                   u.label AS label,
                   bab_count,
                   count(p) AS pasal_count
        """).data()
    return {"documents": results}
