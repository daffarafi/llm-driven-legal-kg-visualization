"""Fix missing MEMUAT edges for UU PDP (UU_27_2022).

Pasal 19-56 exist but are not connected to their Bab.
Based on actual UU PDP structure:
  BAB V:  Pasal 16-34 (Pemrosesan Data Pribadi)
  BAB VI: Pasal 35-50 (Kewajiban Pengendali & Prosesor)
  BAB VII: Pasal 51-56 (Transfer Data Pribadi)
Also: Pasal 1 missing from BAB I.
"""
import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Correct mapping based on actual UU 27/2022 structure
FIXES = {
    "Bab_V_Pemrosesan_Data_Pribadi": list(range(19, 35)),     # Pasal 19-34
    "Bab_VI_Kewajiban_Pengendali_dan_Prosesor_Data_Pribadi": list(range(35, 51)),  # Pasal 35-50
    "Bab_VII_Transfer_Data_Pribadi": list(range(51, 57)),       # Pasal 51-56
}

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
print(f"Connected to {NEO4J_URI} (db: {NEO4J_DATABASE})")

added = 0
with driver.session(database=NEO4J_DATABASE) as session:
    for bab_id, pasal_nums in FIXES.items():
        for num in pasal_nums:
            pasal_id = f"Pasal_{num}"
            result = session.run("""
                MATCH (b:Bab {id: $bab_id})
                MATCH (p:Pasal {id: $pasal_id})
                WHERE p.source_document_id = 'UU_27_2022'
                MERGE (b)-[r:MEMUAT]->(p)
                SET r.source_document_id = 'UU_27_2022',
                    r.created_at = datetime(),
                    r.fix_note = 'added by fix script'
                RETURN b.label AS bab, p.label AS pasal
            """, bab_id=bab_id, pasal_id=pasal_id).data()
            
            if result:
                added += 1
                print(f"  + {result[0]['bab']} -> {result[0]['pasal']}")
            else:
                print(f"  ! NOT FOUND: {bab_id} or {pasal_id}")

print(f"\nAdded {added} MEMUAT edges to Neo4j")

# Verify
print("\n=== Verification ===")
with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run("""
        MATCH (b:Bab)-[:MEMUAT]->(p:Pasal)
        WHERE b.source_document_id = 'UU_27_2022'
        RETURN b.label AS bab, count(p) AS pasal_count
        ORDER BY b.label
    """).data()
    for r in result:
        print(f"  {r['bab']}: {r['pasal_count']} pasal")

    orphans = session.run("""
        MATCH (p:Pasal)
        WHERE p.source_document_id = 'UU_27_2022'
          AND NOT (:Bab)-[:MEMUAT]->(p)
        RETURN count(p) AS orphan_count
    """).single()
    print(f"\n  Pasal tanpa Bab: {orphans['orphan_count']}")

driver.close()

# Also fix the deduped triples file
TRIPLES_PATH = "data/deduped/UU_27_2022_PROMPT_3_triples.json"
if os.path.exists(TRIPLES_PATH):
    data = json.load(open(TRIPLES_PATH, "r", encoding="utf-8"))
    existing_edges = {(e.get('source_id',''), e.get('target_id',''), e['type']) for e in data['edges']}
    
    new_edges = 0
    for bab_id, pasal_nums in FIXES.items():
        for num in pasal_nums:
            pasal_id = f"Pasal_{num}"
            key = (bab_id, pasal_id, 'MEMUAT')
            if key not in existing_edges:
                data['edges'].append({
                    "source_id": bab_id,
                    "target_id": pasal_id,
                    "type": "MEMUAT",
                    "provenance": {
                        "source_document_id": "UU_27_2022",
                        "extraction_model": "manual_fix"
                    }
                })
                new_edges += 1
    
    data['total_edges'] = len(data['edges'])
    with open(TRIPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nAdded {new_edges} edges to {TRIPLES_PATH}")
    print(f"Total edges now: {data['total_edges']}")
