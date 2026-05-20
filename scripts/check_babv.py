"""Check BAB V connection and API limits."""
import os, json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(".env")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
)
DB = os.getenv("NEO4J_DATABASE", "neo4j")

def run(q, params=None):
    with driver.session(database=DB) as s:
        return s.run(q, params or {}).data()

# 1. Check BAB V connection in Neo4j
print("=== BAB V connections in Neo4j ===")
results = run("""
    MATCH (reg:Regulasi {source_document_id: 'UU_11_2008'})-[r:MEMUAT]->(bab:Bab)
    WHERE bab.label CONTAINS 'BAB V'
    RETURN reg.label AS reg, bab.label AS bab, elementId(reg) AS reg_eid, elementId(bab) AS bab_eid
""")
for x in results:
    print(f"  {x['reg'][:60]}...")
    print(f"    -> {x['bab']}")
    print(f"    reg_eid={x['reg_eid']}")
    print(f"    bab_eid={x['bab_eid']}")

# 2. Total nodes and edges
print("\n=== Totals ===")
total_nodes = run("MATCH (n) RETURN count(n) AS total")[0]["total"]
total_edges = run("MATCH ()-[r]->() RETURN count(r) AS total")[0]["total"]
print(f"All nodes: {total_nodes}")
print(f"All edges: {total_edges}")

# 3. Check what the API returns (simulate limit=1000)
print("\n=== Simulating API graph query (limit=1000) ===")
# Check the actual backend query
api_nodes = run("""
    MATCH (n:Entity)
    RETURN elementId(n) AS id, n.label AS label, labels(n) AS labels,
           n.source_document_id AS doc
    LIMIT 1000
""")
print(f"Nodes returned with LIMIT 1000: {len(api_nodes)}")

# Check if BAB V TRANSAKSI ELEKTRONIK node is in the result
bab_v_in_result = [n for n in api_nodes if n["label"] and "BAB V" in n["label"]]
print(f"BAB V nodes in result: {len(bab_v_in_result)}")
for b in bab_v_in_result:
    print(f"  {b['label']} (id={b['id']}, doc={b['doc']})")

# 4. Check the edge between Regulasi and BAB V specifically
print("\n=== Edge check: Regulasi -> BAB V TRANSAKSI ELEKTRONIK ===")
edge_check = run("""
    MATCH (reg:Regulasi {source_document_id: 'UU_11_2008'})-[r:MEMUAT]->(bab:Bab {label: 'BAB V TRANSAKSI ELEKTRONIK'})
    RETURN type(r) AS type, elementId(reg) AS src_id, elementId(bab) AS tgt_id
""")
if edge_check:
    print(f"  Edge EXISTS: {edge_check[0]['src_id']} -[{edge_check[0]['type']}]-> {edge_check[0]['tgt_id']}")
else:
    print("  Edge MISSING!")

driver.close()
