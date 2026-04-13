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
    """List all available documents — both UndangUndang and Peraturan nodes."""
    with Neo4jService.get_driver().session() as s:
        # Get Peraturan nodes (multi-document cluster)
        peraturan = s.run("""
            MATCH (p:Peraturan)
            OPTIONAL MATCH (n:Entity {source_document_id: p.id})
            WITH p, count(n) AS entity_count
            RETURN p.id AS doc_id,
                   p.label AS label,
                   p.short_name AS short_name,
                   p.regulation_type AS regulation_type,
                   p.number AS number,
                   p.year AS year,
                   p.status AS status,
                   entity_count
            ORDER BY p.year
        """).data()

        # Fallback: also get UndangUndang nodes not in Peraturan
        uu_fallback = s.run("""
            MATCH (u:UndangUndang)
            WHERE NOT EXISTS { MATCH (p:Peraturan {id: u.source_document_id}) }
            OPTIONAL MATCH (u)-[:MEMUAT]->(b:Bab)
            WITH u, count(b) AS bab_count
            OPTIONAL MATCH (u)-[:MEMUAT]->(:Bab)-[:MEMUAT|MEMILIKI_PASAL]->(p:Pasal)
            RETURN elementId(u) AS id,
                   u.label AS label,
                   bab_count,
                   count(p) AS pasal_count
        """).data()

    return {
        "regulations": peraturan,
        "documents": uu_fallback,  # backward compatible
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
