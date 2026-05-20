"""Validate test_data.csv: check all expected Cypher queries execute on Neo4j."""
import csv, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "passwd123"))

with open("finetuning/query_model/data/test_data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total test pairs: {len(rows)}\n")

ok, fail = 0, 0
categories = {}
for i, row in enumerate(rows):
    q = row["question"]
    cypher = row["expected_cypher"]
    cat = row["category"]
    pat = row["pattern"]

    categories.setdefault(cat, {"ok": 0, "fail": 0, "total": 0})
    categories[cat]["total"] += 1

    with driver.session(database="experiment") as s:
        try:
            results = s.run(cypher).data()
            n = len(results)
            if n > 0:
                print(f"  [{i+1:2d}] OK  ({n:2d} rows) [{cat}/{pat}] {q[:60]}")
                ok += 1
                categories[cat]["ok"] += 1
            else:
                print(f"  [{i+1:2d}] EMPTY       [{cat}/{pat}] {q[:60]}")
                ok += 1  # Query valid, just no data
                categories[cat]["ok"] += 1
        except Exception as e:
            print(f"  [{i+1:2d}] FAIL        [{cat}/{pat}] {q[:60]}")
            print(f"         ERROR: {str(e)[:100]}")
            fail += 1
            categories[cat]["fail"] += 1

print(f"\n{'='*60}")
print(f"Results: {ok} OK, {fail} FAIL / {len(rows)} total")
print(f"\nPer category:")
for cat, d in sorted(categories.items()):
    print(f"  {cat:12s}: {d['ok']}/{d['total']} OK")

driver.close()
