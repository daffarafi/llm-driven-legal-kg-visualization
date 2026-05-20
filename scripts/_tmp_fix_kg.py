"""
Fix: Split shared 'Bagian Kesatu Umum' node in POJK_11_2022 KG.

Root cause: BAB II and BAB IV both have "Bagian Kesatu Umum" as sub-section.
The deduplicator merged them into one node, causing cross-contamination.

This script fixes it directly in Neo4j by:
1. Finding the shared Bagian node
2. Creating a new separate node for BAB IV
3. Re-wiring Pasal edges correctly
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(r'd:\TA\llm-driven-legal-kg-visualization')
os.chdir(PROJECT_ROOT)
load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', ''))
)
db = os.getenv('NEO4J_DATABASE', 'neo4j')

with driver.session(database=db) as s:
    # Step 1: Diagnose - find the shared Bagian node connected to both BAB II and BAB IV
    print("=== DIAGNOSE ===")
    result = s.run("""
        MATCH (b2:Bab)-[:MEMUAT]->(bg:Bagian)<-[:MEMUAT]-(b4:Bab)
        WHERE bg.source_document_id = 'POJK_11_2022'
          AND bg.label CONTAINS 'Kesatu Umum'
          AND b2.label CONTAINS 'BAB II'
          AND b4.label CONTAINS 'BAB IV'
        OPTIONAL MATCH (bg)-[:MEMUAT]->(p:Pasal)
        RETURN bg.id AS bg_id, bg.label AS bg_label,
               b2.label AS bab2, b4.label AS bab4,
               COLLECT(p.label) AS pasals
    """)
    rows = [dict(r) for r in result]
    
    if not rows:
        print("No shared Bagian node found — may already be fixed!")
        # Verify current state
        result = s.run("""
            MATCH (r:Regulasi)-[:MEMUAT]->(b:Bab)
            WHERE r.source_document_id = 'POJK_11_2022'
              AND (b.label CONTAINS 'BAB II' OR b.label CONTAINS 'BAB IV')
              AND NOT b.label CONTAINS 'BAB III'
            OPTIONAL MATCH (b)-[:MEMUAT]->(bg:Bagian)-[:MEMUAT]->(p1:Pasal)
            OPTIONAL MATCH (b)-[:MEMUAT]->(p2:Pasal)
            WITH b, COLLECT(DISTINCT COALESCE(p1.label, p2.label)) AS pasal_list
            RETURN b.label AS bab, SIZE(pasal_list) AS count, pasal_list
            ORDER BY bab
        """)
        for r in result:
            rec = dict(r)
            print(f"  {rec['bab']}: {rec['count']} pasal — {sorted(rec['pasal_list'])}")
        driver.close()
        sys.exit(0)
    
    row = rows[0]
    shared_id = row['bg_id']
    print(f"  Shared node: {row['bg_label']} (id={shared_id})")
    print(f"  Connected to: {row['bab2']} AND {row['bab4']}")
    print(f"  Pasal under shared: {sorted(row['pasals'])}")
    
    # BAB II should have: Pasal 2, 3 (from Bagian Kesatu Umum)
    # BAB IV should have: Pasal 15 (from Bagian Kesatu Umum)
    bab4_pasals = ['Pasal 15']
    
    # Step 2: Create new Bagian node for BAB IV
    print("\n=== FIX ===")
    new_id = "POJK_11_2022__Bagian_Kesatu_Umum__BAB_IV"
    
    s.run("""
        MATCH (bg:Bagian {id: $shared_id})
        CREATE (new:Entity:Bagian {
            id: $new_id,
            label: bg.label,
            content: COALESCE(bg.content, ''),
            node_type: 'Bagian',
            source_document_id: 'POJK_11_2022'
        })
    """, shared_id=shared_id, new_id=new_id)
    print(f"  Created new Bagian node: {new_id}")
    
    # Step 3: Connect BAB IV → new Bagian
    s.run("""
        MATCH (b:Bab), (new:Bagian {id: $new_id})
        WHERE b.source_document_id = 'POJK_11_2022'
          AND b.label CONTAINS 'BAB IV'
          AND NOT b.label CONTAINS 'BAB XIV'
        MERGE (b)-[:MEMUAT]->(new)
    """, new_id=new_id)
    print(f"  Connected BAB IV → new Bagian")
    
    # Step 4: Move Pasal 15 edge: old Bagian → new Bagian
    for pl in bab4_pasals:
        s.run("""
            MATCH (new:Bagian {id: $new_id})
            MATCH (p:Pasal {source_document_id: 'POJK_11_2022'})
            WHERE p.label = $pl
            MERGE (new)-[:MEMUAT]->(p)
        """, new_id=new_id, pl=pl)
        
        s.run("""
            MATCH (bg:Bagian {id: $shared_id})-[r:MEMUAT]->(p:Pasal)
            WHERE p.label = $pl
            DELETE r
        """, shared_id=shared_id, pl=pl)
        print(f"  Moved {pl}: old → new Bagian")
    
    # Step 5: Disconnect BAB IV from shared Bagian
    s.run("""
        MATCH (b:Bab)-[r:MEMUAT]->(bg:Bagian {id: $shared_id})
        WHERE b.source_document_id = 'POJK_11_2022'
          AND b.label CONTAINS 'BAB IV'
          AND NOT b.label CONTAINS 'BAB XIV'
        DELETE r
    """, shared_id=shared_id)
    print(f"  Disconnected BAB IV from shared Bagian")
    
    # Step 6: Check if Pasal 20 is missing from BAB IV  
    result = s.run("""
        MATCH (b:Bab)
        WHERE b.source_document_id = 'POJK_11_2022'
          AND b.label CONTAINS 'BAB IV'
          AND NOT b.label CONTAINS 'BAB XIV'
        OPTIONAL MATCH (b)-[:MEMUAT]->(bg:Bagian)-[:MEMUAT]->(p1:Pasal)
        OPTIONAL MATCH (b)-[:MEMUAT]->(p2:Pasal)
        WITH COLLECT(DISTINCT COALESCE(p1.label, p2.label)) AS pasal_list
        RETURN pasal_list
    """)
    bab4_actual = [dict(r) for r in result]
    if bab4_actual:
        pasals = bab4_actual[0]['pasal_list']
        print(f"\n  BAB IV current pasal: {sorted(pasals)}")
        if 'Pasal 20' not in pasals:
            print("  [WARN] Pasal 20 missing! Checking if it exists...")
            result = s.run("""
                MATCH (p:Pasal {source_document_id: 'POJK_11_2022'})
                WHERE p.label = 'Pasal 20'
                RETURN p.id, p.label
            """)
            p20 = [dict(r) for r in result]
            if p20:
                print(f"  Found Pasal 20, connecting to BAB IV Bagian Kedua...")
                # Find Bagian Kedua under BAB IV
                s.run("""
                    MATCH (b:Bab)-[:MEMUAT]->(bg:Bagian)
                    WHERE b.source_document_id = 'POJK_11_2022'
                      AND b.label CONTAINS 'BAB IV'
                      AND NOT b.label CONTAINS 'BAB XIV'
                      AND bg.label CONTAINS 'Kedua'
                    MATCH (p:Pasal {source_document_id: 'POJK_11_2022', label: 'Pasal 20'})
                    MERGE (bg)-[:MEMUAT]->(p)
                """)
                print("  Connected Pasal 20 to Bagian Kedua of BAB IV")
    
    # Step 7: Verify
    print("\n=== VERIFY ===")
    result = s.run("""
        MATCH (r:Regulasi)-[:MEMUAT]->(b:Bab)
        WHERE r.source_document_id = 'POJK_11_2022'
          AND (b.label CONTAINS 'BAB II' OR b.label CONTAINS 'BAB IV')
          AND NOT b.label CONTAINS 'BAB III'
          AND NOT b.label CONTAINS 'BAB XIV'
        OPTIONAL MATCH (b)-[:MEMUAT]->(bg:Bagian)-[:MEMUAT]->(p1:Pasal)
        OPTIONAL MATCH (b)-[:MEMUAT]->(p2:Pasal)
        WITH b, 
             COLLECT(DISTINCT bg.label) AS bagian_list,
             COLLECT(DISTINCT COALESCE(p1.label, p2.label)) AS pasal_list
        RETURN b.label AS bab, bagian_list, SIZE(pasal_list) AS count, pasal_list
        ORDER BY bab
    """)
    
    all_ok = True
    for r in result:
        rec = dict(r)
        expected = 9 if 'BAB II' in rec['bab'] and 'BAB IV' not in rec['bab'] else 6
        ok = rec['count'] == expected
        if not ok: all_ok = False
        icon = '[OK]' if ok else '[FAIL]'
        print(f"  {icon} {rec['bab']}: {rec['count']} pasal (expected {expected})")
        print(f"      Bagian: {rec['bagian_list']}")
        print(f"      Pasal: {sorted(rec['pasal_list'])}")
    
    print(f"\n{'[OK] ALL FIXED!' if all_ok else '[FAIL] Still issues — check manually'}")

driver.close()
