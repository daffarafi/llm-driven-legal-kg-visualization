"""Deep edge audit for UU_11_2008 in Neo4j Knowledge Graph."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(".env")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
)
DB = os.getenv("NEO4J_DATABASE", "neo4j")
DOC = "UU_11_2008"

def run(q, params=None):
    with driver.session(database=DB) as s:
        return s.run(q, params or {}).data()

print("=" * 70)
print(f"EDGE AUDIT: {DOC}")
print("=" * 70)

# 1. Edge type distribution
print("\n--- 1. EDGE TYPE DISTRIBUTION ---")
edges = run("""
    MATCH (a)-[r]->(b)
    WHERE a.source_document_id = $doc OR b.source_document_id = $doc
    RETURN type(r) AS type, count(r) AS count
    ORDER BY count DESC
""", {"doc": DOC})
total_edges = sum(e["count"] for e in edges)
for e in edges:
    pct = e["count"] / total_edges * 100
    print(f"  {e['type']:30s} {e['count']:5d}  ({pct:.1f}%)")
print(f"  {'TOTAL':30s} {total_edges:5d}")

# 2. MEMUAT hierarchy check (Regulasi->Bab->Bagian->Pasal)
print("\n--- 2. MEMUAT HIERARCHY CHECK ---")

# 2a. Regulasi -> Bab
reg_to_bab = run("""
    MATCH (r:Regulasi)-[:MEMUAT]->(b:Bab)
    WHERE r.source_document_id = $doc
    RETURN r.label AS regulasi, b.label AS bab
    ORDER BY b.label
""", {"doc": DOC})
print(f"\n  Regulasi -> Bab: {len(reg_to_bab)} edges")
for r in reg_to_bab:
    print(f"    {r['bab']}")

# 2b. Bab -> Pasal (direct, no Bagian)
bab_to_pasal = run("""
    MATCH (b:Bab)-[:MEMUAT]->(p:Pasal)
    WHERE b.source_document_id = $doc
    RETURN b.label AS bab, p.label AS pasal
    ORDER BY b.label, p.label
""", {"doc": DOC})
print(f"\n  Bab -> Pasal (direct): {len(bab_to_pasal)} edges")
current_bab = None
for r in bab_to_pasal:
    if r["bab"] != current_bab:
        current_bab = r["bab"]
        print(f"    [{current_bab}]")
    print(f"      -> {r['pasal']}")

# 2c. Bab -> Bagian
bab_to_bagian = run("""
    MATCH (b:Bab)-[:MEMUAT]->(bg:Bagian)
    WHERE b.source_document_id = $doc
    RETURN b.label AS bab, bg.label AS bagian
    ORDER BY b.label, bg.label
""", {"doc": DOC})
print(f"\n  Bab -> Bagian: {len(bab_to_bagian)} edges")
for r in bab_to_bagian:
    print(f"    [{r['bab']}] -> {r['bagian']}")

# 2d. Bagian -> Pasal
bagian_to_pasal = run("""
    MATCH (bg:Bagian)-[:MEMUAT]->(p:Pasal)
    WHERE bg.source_document_id = $doc
    RETURN bg.label AS bagian, p.label AS pasal
    ORDER BY bg.label, p.label
""", {"doc": DOC})
print(f"\n  Bagian -> Pasal: {len(bagian_to_pasal)} edges")
current_bg = None
for r in bagian_to_pasal:
    if r["bagian"] != current_bg:
        current_bg = r["bagian"]
        print(f"    [{current_bg}]")
    print(f"      -> {r['pasal']}")

# 3. MEMILIKI_AYAT check
print("\n--- 3. MEMILIKI_AYAT CHECK ---")
pasal_ayat = run("""
    MATCH (p:Pasal)-[:MEMILIKI_AYAT]->(a:Ayat)
    WHERE p.source_document_id = $doc
    RETURN p.label AS pasal, count(a) AS ayat_count
    ORDER BY p.label
""", {"doc": DOC})
print(f"  Pasal with Ayat: {len(pasal_ayat)}")
for p in pasal_ayat:
    print(f"    {p['pasal']:25s} -> {p['ayat_count']} ayat")

# 3b. Orphan Ayat (no MEMILIKI_AYAT incoming)
orphan_ayat = run("""
    MATCH (a:Ayat)
    WHERE a.source_document_id = $doc
      AND NOT ()-[:MEMILIKI_AYAT]->(a)
    RETURN a.label AS label, elementId(a) AS eid
""", {"doc": DOC})
print(f"\n  Orphan Ayat (no parent Pasal): {len(orphan_ayat)}")
for o in orphan_ayat:
    print(f"    {o['label']}")

# 4. Pasal without ANY edges
print("\n--- 4. PASAL WITHOUT ANY OUTGOING SEMANTIC EDGES ---")
pasal_no_out = run("""
    MATCH (p:Pasal)
    WHERE p.source_document_id = $doc
    AND NOT (p)-[:MENGATUR|BERLAKU_UNTUK|MENETAPKAN_SANKSI|MENDEFINISIKAN]->()
    AND NOT (p)-[:MEMILIKI_AYAT]->()
    RETURN p.label AS label
    ORDER BY p.label
""", {"doc": DOC})
print(f"  Pasal with no outgoing semantic edges: {len(pasal_no_out)}")
for p in pasal_no_out:
    print(f"    {p['label']}")

# 5. MENGATUR / BERLAKU_UNTUK / MENETAPKAN_SANKSI breakdown
print("\n--- 5. SEMANTIC EDGE BREAKDOWN ---")
for edge_type in ["MENGATUR", "BERLAKU_UNTUK", "MENETAPKAN_SANKSI", "MENDEFINISIKAN", "MERUJUK"]:
    sem = run(f"""
        MATCH (src)-[r:{edge_type}]->(tgt)
        WHERE src.source_document_id = $doc
        WITH labels(src) AS src_labels, labels(tgt) AS tgt_labels, count(r) AS cnt
        RETURN src_labels, tgt_labels, cnt
        ORDER BY cnt DESC
    """, {"doc": DOC})
    total = sum(s["cnt"] for s in sem)
    print(f"\n  {edge_type}: {total} edges")
    for s in sem:
        src_type = [l for l in s["src_labels"] if l != "Entity"]
        tgt_type = [l for l in s["tgt_labels"] if l != "Entity"]
        print(f"    {src_type} -> {tgt_type}: {s['cnt']}")

# 6. Cross-references (MERUJUK between documents)
print("\n--- 6. CROSS-DOCUMENT REFERENCES ---")
cross = run("""
    MATCH (a)-[r:MERUJUK]->(b)
    WHERE a.source_document_id = $doc AND b.source_document_id <> $doc
    RETURN a.label AS from_node, b.label AS to_node, b.source_document_id AS to_doc
    ORDER BY to_doc, to_node
""", {"doc": DOC})
print(f"  Cross-doc references FROM {DOC}: {len(cross)}")
for c in cross:
    print(f"    {c['from_node']} -> [{c['to_doc']}] {c['to_node']}")

# 6b. Internal references
internal = run("""
    MATCH (a)-[r:MERUJUK]->(b)
    WHERE a.source_document_id = $doc AND b.source_document_id = $doc
    RETURN a.label AS from_node, b.label AS to_node
    ORDER BY from_node
""", {"doc": DOC})
print(f"\n  Internal MERUJUK within {DOC}: {len(internal)}")
for i in internal[:30]:
    print(f"    {i['from_node']} -> {i['to_node']}")
if len(internal) > 30:
    print(f"    ... and {len(internal) - 30} more")

# 7. Duplicate edges check
print("\n--- 7. DUPLICATE EDGES CHECK ---")
dupes = run("""
    MATCH (a)-[r]->(b)
    WHERE a.source_document_id = $doc
    WITH a, b, type(r) AS rtype, count(r) AS cnt
    WHERE cnt > 1
    RETURN a.label AS from_node, rtype, b.label AS to_node, cnt
    ORDER BY cnt DESC
    LIMIT 20
""", {"doc": DOC})
print(f"  Duplicate edges found: {len(dupes)}")
for d in dupes:
    print(f"    {d['from_node']} -[{d['rtype']}]-> {d['to_node']}  (x{d['cnt']})")

# 8. Self-referencing edges
print("\n--- 8. SELF-REFERENCING EDGES ---")
self_ref = run("""
    MATCH (a)-[r]->(a)
    WHERE a.source_document_id = $doc
    RETURN a.label AS node, type(r) AS type
""", {"doc": DOC})
print(f"  Self-referencing edges: {len(self_ref)}")
for s in self_ref:
    print(f"    {s['node']} -[{s['type']}]-> (self)")

# 9. Bab without Pasal
print("\n--- 9. BAB WITHOUT ANY PASAL ---")
bab_no_pasal = run("""
    MATCH (b:Bab)
    WHERE b.source_document_id = $doc
    AND NOT (b)-[:MEMUAT]->(:Pasal)
    AND NOT (b)-[:MEMUAT]->(:Bagian)-[:MEMUAT]->(:Pasal)
    RETURN b.label AS label
""", {"doc": DOC})
print(f"  Bab without Pasal (direct or via Bagian): {len(bab_no_pasal)}")
for b in bab_no_pasal:
    print(f"    {b['label']}")

driver.close()
print("\n" + "=" * 70)
print("EDGE AUDIT COMPLETE")
print("=" * 70)
