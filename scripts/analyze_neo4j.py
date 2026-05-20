"""Deep analysis of Neo4j KG data — Entity label issue and overall integrity."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "")
DB = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(URI, auth=(USER, PWD))

def run(query, params=None):
    with driver.session(database=DB) as s:
        return s.run(query, params or {}).data()

print("=" * 70)
print("NEO4J KNOWLEDGE GRAPH — DEEP ANALYSIS")
print("=" * 70)

# 1. Overall stats
print("\n--- 1. OVERALL DATABASE STATS ---")
totals = run("MATCH (n) RETURN count(n) AS nodes")[0]
edges_total = run("MATCH ()-[r]->() RETURN count(r) AS edges")[0]
print(f"Total nodes: {totals['nodes']}")
print(f"Total edges: {edges_total['edges']}")

# 2. All unique label combinations
print("\n--- 2. UNIQUE LABEL COMBINATIONS ---")
combos = run("""
    MATCH (n)
    RETURN labels(n) AS label_combo, count(n) AS count
    ORDER BY count DESC
""")
for c in combos:
    print(f"  {str(c['label_combo']):50s} → {c['count']} nodes")

# 3. Nodes WITHOUT "Entity" label
print("\n--- 3. NODES WITHOUT 'Entity' LABEL ---")
no_entity = run("""
    MATCH (n)
    WHERE NOT n:Entity
    RETURN labels(n) AS labels, n.label AS label, n.source_document_id AS doc_id,
           elementId(n) AS eid
    ORDER BY doc_id, labels
""")
print(f"Found {len(no_entity)} nodes without 'Entity' label:")
for n in no_entity:
    print(f"  [{n['doc_id']}] labels={n['labels']}, label=\"{n['label']}\", eid={n['eid']}")

# 4. Per-document breakdown
print("\n--- 4. PER-DOCUMENT BREAKDOWN ---")
per_doc = run("""
    MATCH (n)
    WHERE n.source_document_id IS NOT NULL
    RETURN n.source_document_id AS doc_id,
           count(n) AS total,
           sum(CASE WHEN n:Entity THEN 1 ELSE 0 END) AS with_entity,
           sum(CASE WHEN NOT n:Entity THEN 1 ELSE 0 END) AS without_entity
    ORDER BY doc_id
""")
for d in per_doc:
    pct = (d['with_entity'] / d['total'] * 100) if d['total'] > 0 else 0
    print(f"  {d['doc_id']:25s} total={d['total']:4d}  Entity={d['with_entity']:4d}  missing={d['without_entity']:4d}  ({pct:.1f}%)")

# 5. Nodes with NO source_document_id
print("\n--- 5. ORPHAN NODES (no source_document_id) ---")
orphans = run("""
    MATCH (n)
    WHERE n.source_document_id IS NULL
    RETURN labels(n) AS labels, n.label AS label, elementId(n) AS eid
    LIMIT 20
""")
print(f"Found {len(orphans)} orphan nodes (showing max 20):")
for o in orphans:
    print(f"  labels={o['labels']}, label=\"{o['label']}\", eid={o['eid']}")

# 6. Per-document, per-type breakdown (to see distribution)
print("\n--- 6. PER-DOCUMENT NODE TYPE DISTRIBUTION ---")
dist = run("""
    MATCH (n)
    WHERE n.source_document_id IS NOT NULL
    WITH n.source_document_id AS doc_id, labels(n) AS lbls
    UNWIND lbls AS lbl
    WITH doc_id, lbl
    WHERE lbl <> 'Entity'
    RETURN doc_id, lbl AS node_type, count(*) AS count
    ORDER BY doc_id, count DESC
""")
current_doc = None
for d in dist:
    if d['doc_id'] != current_doc:
        current_doc = d['doc_id']
        print(f"\n  [{current_doc}]")
    print(f"    {d['node_type']:25s} {d['count']}")

# 7. Nodes with ONLY "Entity" label (no semantic type)
print("\n\n--- 7. NODES WITH ONLY 'Entity' LABEL (no semantic type) ---")
entity_only = run("""
    MATCH (n:Entity)
    WHERE size(labels(n)) = 1
    RETURN n.label AS label, n.source_document_id AS doc_id, elementId(n) AS eid
    LIMIT 20
""")
print(f"Found {len(entity_only)} nodes with only 'Entity' label (showing max 20):")
for e in entity_only:
    print(f"  [{e['doc_id']}] label=\"{e['label']}\", eid={e['eid']}")

# 8. Edge types overview
print("\n--- 8. EDGE TYPE DISTRIBUTION ---")
edge_dist = run("""
    MATCH ()-[r]->()
    RETURN type(r) AS type, count(r) AS count
    ORDER BY count DESC
""")
for e in edge_dist:
    print(f"  {e['type']:30s} {e['count']}")

# 9. Disconnected nodes
print("\n--- 9. DISCONNECTED NODES (no relationships) ---")
disconn = run("""
    MATCH (n)
    WHERE NOT (n)--()
    RETURN labels(n) AS labels, n.label AS label, n.source_document_id AS doc_id
    LIMIT 20
""")
print(f"Found {len(disconn)} disconnected nodes (showing max 20):")
for d in disconn:
    print(f"  [{d['doc_id']}] labels={d['labels']}, label=\"{d['label']}\"")

# 10. VersiPasal nodes check
print("\n--- 10. VERSIPASAL NODES CHECK ---")
vp = run("""
    MATCH (n:VersiPasal)
    RETURN n.label AS label, n.source_document_id AS doc_id, labels(n) AS labels
    LIMIT 10
""")
print(f"Found {len(vp)} VersiPasal nodes:")
for v in vp:
    print(f"  [{v['doc_id']}] labels={v['labels']}, label=\"{v['label']}\"")

driver.close()
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
