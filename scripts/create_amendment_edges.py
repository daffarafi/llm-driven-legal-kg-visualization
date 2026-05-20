"""
Step 4: Create Cross-Document MENGAMANDEMEN Edges

Auto-detect Regulasi nodes whose label contains "Perubahan atas"
and create MENGAMANDEMEN edges to the target Regulasi.

Direction: (amender) -[:MENGAMANDEMEN]-> (target)

Run AFTER 03_neo4j_ingestion.ipynb has loaded all documents.
Requires: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE in .env
"""

import re
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
load_dotenv()

from pipeline.load.neo4j_loader import Neo4jLoader

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print(f"Connecting to Neo4j: {NEO4J_URI} / {NEO4J_DATABASE}")
loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)

with loader._session() as session:
    # Find Regulasi nodes that are amendments
    result = session.run("""
        MATCH (r:Regulasi)
        WHERE toLower(r.label) CONTAINS 'perubahan atas'
        RETURN r.id AS id, r.label AS label, r.source_document_id AS doc_id
    """)
    
    records = list(result)
    print(f"\nFound {len(records)} amendment Regulasi node(s)\n")
    
    for record in records:
        amender_label = record["label"]
        amender_id = record["id"]
        amender_doc = record["doc_id"]
        
        # Extract target UU number + year from label
        match = re.search(r'Nomor\s+(\d+)\s+Tahun\s+(\d{4})', amender_label)
        if match:
            number, year = match.group(1), match.group(2)
            target_doc = f"UU_{number}_{year}"
            
            result2 = session.run("""
                MATCH (amender:Regulasi {id: $amender_id})
                MATCH (target:Regulasi)
                WHERE target.source_document_id = $target_doc
                MERGE (amender)-[:MENGAMANDEMEN]->(target)
                RETURN target.label AS target_label
            """, amender_id=amender_id, target_doc=target_doc)
            
            for r2 in result2:
                print(f"  {amender_doc} -[MENGAMANDEMEN]-> {target_doc}")
                print(f"    {amender_label}")
                print(f"    -> {r2['target_label']}\n")
        else:
            print(f"  SKIP (no regex match): {amender_label}")
    
    if not records:
        print("No amendment Regulasi found — skipping.")

# Verify
print("=" * 60)
print("Verification:")
with loader._session() as session:
    result = session.run("""
        MATCH (a:Regulasi)-[:MENGAMANDEMEN]->(b:Regulasi)
        RETURN a.source_document_id AS amender, b.source_document_id AS target,
               a.label AS amender_label, b.label AS target_label
    """)
    records = list(result)
    if records:
        for r in records:
            print(f"  {r['amender']} -> {r['target']}")
            print(f"    {r['amender_label']}")
            print(f"    {r['target_label']}")
    else:
        print("  No MENGAMANDEMEN edges found.")

loader.close()
print("\nDone.")
