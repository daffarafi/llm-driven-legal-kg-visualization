"""Find and fix Pasal nodes directly connected to Regulasi (should be via Bab/Bagian)."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv(".env")
driver = GraphDatabase.driver(os.getenv("NEO4J_URI","bolt://localhost:7687"), auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASSWORD","")))
DB = os.getenv("NEO4J_DATABASE","neo4j")

def run(q, params=None):
    with driver.session(database=DB) as s:
        return s.run(q, params or {}).data()

# Find all Regulasi -> Pasal direct edges (ALL docs)
print("=" * 70)
print("Regulasi -> Pasal direct MEMUAT edges (WRONG - should be via Bab)")
print("=" * 70)

direct = run("""
    MATCH (reg:Regulasi)-[r:MEMUAT]->(p:Pasal)
    RETURN reg.label AS regulasi, reg.source_document_id AS doc,
           p.label AS pasal, elementId(reg) AS reg_id, elementId(p) AS p_id
    ORDER BY doc, p.label
""")

print(f"\nFound {len(direct)} direct Regulasi->Pasal edges:\n")
current_doc = None
for d in direct:
    if d["doc"] != current_doc:
        current_doc = d["doc"]
        print(f"  [{current_doc}]")
    print(f"    Regulasi -> {d['pasal']}")

# Also check Regulasi -> Bagian (should be via Bab)
print("\n" + "=" * 70)
print("Regulasi -> Bagian direct MEMUAT edges (WRONG - should be via Bab)")
print("=" * 70)

direct_bg = run("""
    MATCH (reg:Regulasi)-[r:MEMUAT]->(bg:Bagian)
    RETURN reg.label AS regulasi, reg.source_document_id AS doc,
           bg.label AS bagian
    ORDER BY doc
""")
print(f"\nFound {len(direct_bg)} direct Regulasi->Bagian edges")
for d in direct_bg:
    print(f"  [{d['doc']}] Regulasi -> {d['bagian']}")

# Also check Regulasi -> Ayat
print("\n" + "=" * 70)
print("Regulasi -> Ayat direct MEMUAT edges (WRONG)")
print("=" * 70)

direct_ayat = run("""
    MATCH (reg:Regulasi)-[r:MEMUAT]->(a:Ayat)
    RETURN reg.source_document_id AS doc, a.label AS ayat
    ORDER BY doc
""")
print(f"\nFound {len(direct_ayat)} direct Regulasi->Ayat edges")
for d in direct_ayat:
    print(f"  [{d['doc']}] Regulasi -> {d['ayat']}")

# Check Bab -> Pasal that should be via Bagian
print("\n" + "=" * 70)
print("Bab -> Pasal where Bagian exists (potential redundant edges)")
print("=" * 70)

redundant = run("""
    MATCH (bab:Bab)-[r1:MEMUAT]->(p:Pasal)
    WHERE EXISTS { (bab)-[:MEMUAT]->(:Bagian)-[:MEMUAT]->(p) }
    RETURN bab.source_document_id AS doc, bab.label AS bab, p.label AS pasal
    ORDER BY doc, bab.label, p.label
""")
print(f"\nFound {len(redundant)} redundant Bab->Pasal edges (Pasal also reachable via Bagian)")
for r in redundant:
    print(f"  [{r['doc']}] {r['bab']} -> {r['pasal']}")

driver.close()
