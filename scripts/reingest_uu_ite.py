"""Re-ingest deduped UU ITE triples into Neo4j.

This script:
1. Deletes existing UU_11_2008 nodes from the database (not all data!)
2. Re-loads the deduped triples with the fixed single Regulasi node
"""
import json
import os
from dotenv import load_dotenv
from pipeline.load.neo4j_loader import Neo4jLoader

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

INPUT_PATH = "data/deduped/UU_11_2008_PROMPT_3_triples.json"

print(f"Neo4j URI: {NEO4J_URI}")
print(f"Database: {NEO4J_DATABASE}")
print(f"Input: {INPUT_PATH}")

# Load data
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

regulasi_nodes = [n for n in data["nodes"] if n["type"] == "Regulasi"]
print(f"\nRegulasi nodes in file: {len(regulasi_nodes)}")
for r in regulasi_nodes:
    print(f"  {r['id']}: {r['label']}")

print(f"\nTotal nodes: {data['total_nodes']}")
print(f"Total edges: {data['total_edges']}")

# Connect
loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database=NEO4J_DATABASE)

try:
    # Step 1: Delete only UU_11_2008 nodes (preserve POJK data)
    print("\n--- Step 1: Removing old UU_11_2008 nodes ---")
    with loader._session() as session:
        result = session.run("""
            MATCH (n:Entity)
            WHERE n.source_document_id = 'UU_11_2008'
            DETACH DELETE n
            RETURN count(n) AS deleted
        """).single()
        print(f"  Deleted {result['deleted']} old UU_11_2008 nodes")

    # Step 2: Re-load nodes
    print(f"\n--- Step 2: Loading {len(data['nodes'])} nodes ---")
    loader.load_nodes(data["nodes"])

    # Step 3: Re-load edges
    print(f"\n--- Step 3: Loading {len(data['edges'])} edges ---")
    loader.load_edges(data["edges"])

    # Step 4: Verify
    print("\n--- Step 4: Verification ---")
    stats = loader.get_stats()
    print(f"Total nodes in DB: {stats['total_nodes']}")
    print(f"Total edges in DB: {stats['total_edges']}")
    print(f"\nNode labels:")
    for label, count in sorted(stats["node_labels"].items(), key=lambda x: -x[1]):
        print(f"  {label:20s}: {count}")

    # Check Regulasi count
    with loader._session() as session:
        regs = session.run("MATCH (r:Regulasi) RETURN r.id AS id, r.label AS label, r.source_document_id AS src").data()
        print(f"\nRegulasi nodes in DB: {len(regs)}")
        for r in regs:
            print(f"  {r['id']}: {r['label']} (src: {r['src']})")

    print("\n✅ Re-ingestion complete!")

finally:
    loader.close()
