"""Document API router — multi-document regulation cluster support."""

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
    """List all available documents — Regulasi nodes."""
    regulasi = Neo4jService.get_regulations()
    return {
        "regulations": regulasi,
        "documents": regulasi,
    }



@router.get("/regulations/graph")
async def get_regulation_graph():
    """Get inter-document relationship graph (Peraturan → Peraturan)."""
    with Neo4jService.get_driver().session() as s:
        result = s.run("""
            MATCH (a:Peraturan)-[r]->(b:Peraturan)
            RETURN collect(DISTINCT {
                id: a.id,
                label: a.short_name,
                full_label: a.label,
                type: a.regulation_type,
                year: a.year,
                status: a.status
            }) + collect(DISTINCT {
                id: b.id,
                label: b.short_name,
                full_label: b.label,
                type: b.regulation_type,
                year: b.year,
                status: b.status
            }) AS nodes,
            collect(DISTINCT {
                source: a.id,
                target: b.id,
                type: type(r),
                description: r.description
            }) AS edges
        """).single()

    # Deduplicate nodes
    nodes_map = {}
    for n in (result["nodes"] or []):
        if n and n.get("id"):
            nodes_map[n["id"]] = n

    return {
        "nodes": list(nodes_map.values()),
        "edges": [e for e in (result["edges"] or []) if e and e.get("source")],
    }


@router.get("/regulations/amendments")
async def get_amendments():
    """Get amendment version tracking (VersiPasal nodes)."""
    with Neo4jService.get_driver().session() as s:
        versions = s.run("""
            MATCH (v:VersiPasal)
            OPTIONAL MATCH (v)-[:DIAMANDEMEN_MENJADI]->(v2:VersiPasal)
            RETURN v.id AS id,
                   v.label AS label,
                   v.version AS version,
                   v.status AS status,
                   v.source_document_id AS source_doc,
                   v2.id AS amended_to_id,
                   v2.label AS amended_to_label
            ORDER BY v.source_document_id, v.version
        """).data()

    return {"amendments": versions}
