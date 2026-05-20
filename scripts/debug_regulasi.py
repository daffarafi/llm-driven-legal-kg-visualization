"""Debug script: inspect Regulasi nodes in Neo4j."""
from app.services.neo4j_service import Neo4jService
import json

print("=== 1. Check if get_session works ===")
try:
    with Neo4jService.get_session() as s:
        print("  get_session() OK")
except Exception as e:
    print(f"  get_session() FAILED: {e}")

print("\n=== 2. Check if get_driver works ===")
try:
    driver = Neo4jService.get_driver()
    print(f"  get_driver() returned: {type(driver)}")
    with driver.session() as s:
        print("  driver.session() OK")
except Exception as e:
    print(f"  get_driver() FAILED: {e}")

print("\n=== 3. Count Regulasi nodes ===")
with Neo4jService.get_session() as s:
    count = s.run("MATCH (r:Regulasi) RETURN count(r) AS cnt").single()
    print(f"  Regulasi count: {count['cnt']}")

print("\n=== 4. Regulasi node properties (first 3) ===")
with Neo4jService.get_session() as s:
    nodes = s.run("MATCH (r:Regulasi) RETURN properties(r) AS props LIMIT 3").data()
    for i, n in enumerate(nodes):
        print(f"  Node {i}: {json.dumps(n['props'], indent=2, ensure_ascii=False, default=str)}")

print("\n=== 5. Test get_regulations() method ===")
try:
    regs = Neo4jService.get_regulations()
    print(f"  get_regulations() returned {len(regs)} items")
    for r in regs[:3]:
        print(f"    {json.dumps(r, ensure_ascii=False, default=str)}")
except Exception as e:
    print(f"  get_regulations() FAILED: {e}")

print("\n=== 6. Test the EXACT router query ===")
try:
    with Neo4jService.get_driver().session() as s:
        regulasi = s.run("""
            MATCH (r:Regulasi)
            OPTIONAL MATCH (n:Entity {source_document_id: r.source_document_id})
            WHERE n <> r
            WITH r, count(n) AS entity_count
            RETURN r.id AS doc_id,
                   r.label AS label,
                   r.source_document_id AS source_document_id,
                   r.jenis AS regulation_type,
                   r.node_type AS node_type,
                   entity_count
            ORDER BY r.label
        """).data()
        print(f"  Router query returned {len(regulasi)} items")
        for r in regulasi[:3]:
            print(f"    {json.dumps(r, ensure_ascii=False, default=str)}")
except Exception as e:
    print(f"  Router query FAILED: {e}")
