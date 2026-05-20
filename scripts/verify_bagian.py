"""Verify Bagian Kesatu Umum connections are correct."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv(".env")
driver = GraphDatabase.driver(os.getenv("NEO4J_URI","bolt://localhost:7687"), auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PASSWORD","")))
with driver.session(database=os.getenv("NEO4J_DATABASE","neo4j")) as s:
    r = s.run("""
        MATCH (bab:Bab {source_document_id: 'POJK_11_2022'})-[:MEMUAT]->(bg:Bagian {label: 'Bagian Kesatu Umum'})-[:MEMUAT]->(p:Pasal)
        RETURN bab.label AS bab, bg.label AS bagian, p.label AS pasal, elementId(bg) AS bg_id
        ORDER BY bab.label, p.label
    """).data()
    print("Bagian Kesatu Umum -> Pasal (grouped by parent Bab):")
    for x in r:
        print(f"  {x['bab'][:55]} -> {x['pasal']}  (bg_id={x['bg_id']})")
driver.close()
