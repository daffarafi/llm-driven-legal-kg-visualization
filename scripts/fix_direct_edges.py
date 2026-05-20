"""Delete all wrong Regulasi -> Pasal direct MEMUAT edges.
These Pasal should only be connected via Bab (or Bab->Bagian).
For UU_19_2016 which has no Bab, Regulasi->Pasal is actually correct.
"""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv(".env")
driver = GraphDatabase.driver(os.getenv("NEO4J_URI","bolt://localhost:7687"), auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASSWORD","")))
DB = os.getenv("NEO4J_DATABASE","neo4j")

def run(q, params=None):
    with driver.session(database=DB) as s:
        return s.run(q, params or {}).data()

def run_single(q, params=None):
    with driver.session(database=DB) as s:
        return s.run(q, params or {}).single()

# ═══════════════════════════════════════════════════════════════
# STEP 1: Verify which Pasal already have correct Bab path
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Check if Pasal also reachable via Bab (safe to delete direct)")
print("=" * 70)

# POJK_11_2022 direct edges
print("\n[POJK_11_2022]")
pojk_direct = run("""
    MATCH (reg:Regulasi {source_document_id: 'POJK_11_2022'})-[:MEMUAT]->(p:Pasal)
    OPTIONAL MATCH path = (reg)-[:MEMUAT]->(:Bab)-[:MEMUAT*1..2]->(p)
    RETURN p.label AS pasal, path IS NOT NULL AS has_bab_path
    ORDER BY p.label
""")
for p in pojk_direct:
    status = "OK (has Bab path)" if p["has_bab_path"] else "WARNING (no Bab path!)"
    print(f"  {p['pasal']:20s} {status}")

# UU_19_2016 direct edges - this doc has no Bab structure
print("\n[UU_19_2016]")
uu19_direct = run("""
    MATCH (reg:Regulasi {source_document_id: 'UU_19_2016'})-[:MEMUAT]->(p:Pasal)
    OPTIONAL MATCH path = (reg)-[:MEMUAT]->(:Bab)-[:MEMUAT*1..2]->(p)
    RETURN p.label AS pasal, path IS NOT NULL AS has_bab_path
    ORDER BY p.label
""")
for p in uu19_direct:
    status = "OK (has Bab path)" if p["has_bab_path"] else "NO Bab path (amendment doc - expected)"
    print(f"  {p['pasal']:20s} {status}")

# Check if UU_19_2016 even has Bab nodes
uu19_babs = run("MATCH (b:Bab {source_document_id: 'UU_19_2016'}) RETURN b.label AS bab")
print(f"\n  UU_19_2016 Bab count: {len(uu19_babs)}")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Delete redundant Regulasi->Pasal for POJK (has Bab path)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Delete redundant Regulasi->Pasal edges for POJK_11_2022")
print("=" * 70)

result = run_single("""
    MATCH (reg:Regulasi {source_document_id: 'POJK_11_2022'})-[r:MEMUAT]->(p:Pasal)
    WHERE EXISTS { (reg)-[:MEMUAT]->(:Bab)-[:MEMUAT*1..2]->(p) }
    DELETE r
    RETURN count(r) AS deleted
""")
print(f"  Deleted {result['deleted']} redundant POJK edges")

# For UU_19_2016: keep Regulasi->Pasal since this is an amendment doc with no Bab
print("\n  UU_19_2016: Keeping Regulasi->Pasal edges (amendment doc, no Bab hierarchy)")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Verify
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Verify remaining Regulasi->Pasal edges")
print("=" * 70)

remaining = run("""
    MATCH (reg:Regulasi)-[r:MEMUAT]->(p:Pasal)
    RETURN reg.source_document_id AS doc, p.label AS pasal
    ORDER BY doc, p.label
""")
print(f"\nRemaining Regulasi->Pasal edges: {len(remaining)}")
current_doc = None
for r in remaining:
    if r["doc"] != current_doc:
        current_doc = r["doc"]
        print(f"\n  [{current_doc}]")
    print(f"    -> {r['pasal']}")

# Also verify POJK pasal are still reachable
print("\n\nPOJK Pasal 29-39 reachability check:")
for pasal in ["Pasal 29", "Pasal 30", "Pasal 31", "Pasal 32", "Pasal 35", "Pasal 36", "Pasal 37", "Pasal 39"]:
    path = run("""
        MATCH path = (:Regulasi {source_document_id: 'POJK_11_2022'})-[:MEMUAT]->(:Bab)-[:MEMUAT*1..2]->(p:Pasal {label: $pasal, source_document_id: 'POJK_11_2022'})
        RETURN [n IN nodes(path) | n.label] AS labels
    """, {"pasal": pasal})
    if path:
        print(f"  {pasal}: {' -> '.join(path[0]['labels'])}")
    else:
        print(f"  {pasal}: NOT REACHABLE!")

driver.close()
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
