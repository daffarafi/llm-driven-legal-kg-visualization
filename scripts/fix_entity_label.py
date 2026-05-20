"""Fix: Add Entity label to all nodes missing it."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(".env")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
)

with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
    # Show nodes before fix
    before = s.run("MATCH (n) WHERE NOT n:Entity RETURN n.label AS label, labels(n) AS labels").data()
    print(f"Nodes without Entity label BEFORE fix: {len(before)}")
    for b in before:
        print(f"  - {b['label']} (labels: {b['labels']})")

    # Fix: add Entity label
    result = s.run("MATCH (n) WHERE NOT n:Entity SET n:Entity RETURN count(n) AS fixed").single()
    print(f"\nFixed {result['fixed']} nodes")

    # Verify
    after = s.run("MATCH (n) WHERE NOT n:Entity RETURN count(n) AS remaining").single()
    print(f"Nodes without Entity label AFTER fix: {after['remaining']}")

driver.close()
print("\nDone!")
