"""Fix 4 failing Q2C test cases with exact-match queries and update CSV."""
import csv
import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "passwd123"))

CSV_PATH = r'd:\TA\[SFT] Regulation Fine Tuning for Knowledge Graph - QUESTION_TO_CYPHER_QUERY_DATA_TEST.csv'

# Read CSV
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Loaded {len(rows)} rows\n")

# Define fixes: TEST_ID -> new EXPECTED_CYPHER_QUERY
fixes = {
    "Q2C_011": (
        "MATCH (b:Bab) WHERE toLower(b.label) = 'bab vii' "
        "OPTIONAL MATCH (b)-[:MEMUAT]->(p1:Pasal) "
        "OPTIONAL MATCH (b)-[:MEMUAT]->(bg:Bagian)-[:MEMUAT]->(p2:Pasal) "
        "WITH b, COLLECT(DISTINCT p1) + COLLECT(DISTINCT p2) AS pasals "
        "UNWIND pasals AS p "
        "RETURN b.label AS bab, p.label AS pasal, p.content AS isi "
        "ORDER BY p.label LIMIT 25"
    ),
    "Q2C_013": (
        "MATCH (b:Bab) WHERE toLower(b.label) = 'bab xi' "
        "OPTIONAL MATCH (b)-[:MEMUAT]->(p1:Pasal) "
        "OPTIONAL MATCH (b)-[:MEMUAT]->(bg:Bagian)-[:MEMUAT]->(p2:Pasal) "
        "WITH b, COLLECT(DISTINCT p1) + COLLECT(DISTINCT p2) AS pasals "
        "UNWIND pasals AS p "
        "RETURN b.label AS bab, p.label AS pasal, p.content AS isi "
        "ORDER BY p.label LIMIT 25"
    ),
    "Q2C_036": (
        "MATCH (p:Pasal) WHERE toLower(p.label) = 'pasal 1' "
        "RETURN p.label AS pasal, p.content AS isi LIMIT 25"
    ),
    "Q2C_038": (
        "MATCH (b:Bab)-[:MEMUAT]->(p:Pasal) WHERE toLower(p.label) = 'pasal 3' "
        "RETURN b.label AS bab, p.label AS pasal LIMIT 25"
    ),
}

def truncate_content(results, max_len=200):
    long_fields = ('content', 'isi', 'detail', 'definisi')
    truncated = []
    for row in results:
        new_row = {}
        for k, v in row.items():
            if isinstance(v, str) and k in long_fields and len(v) > max_len:
                new_row[k] = v[:max_len] + '...'
            else:
                new_row[k] = v
        truncated.append(new_row)
    return truncated

# Apply fixes
for row in rows:
    test_id = row['TEST_ID']
    if test_id in fixes:
        new_cypher = fixes[test_id]
        
        # Run the corrected query
        with driver.session(database="experiment") as session:
            results = session.run(new_cypher).data()
        
        results_truncated = truncate_content(results)
        
        old_count = len(json.loads(row['EXPECTED_QUERY_RESULT'])) if row['EXPECTED_QUERY_RESULT'] else 0
        new_count = len(results_truncated)
        
        print(f"[{test_id}] {row['QUESTION']}")
        print(f"  Old query:  ...CONTAINS... -> {old_count} rows")
        print(f"  New query:  ...exact match... -> {new_count} rows")
        
        # Show what was removed
        if old_count > new_count:
            old_data = json.loads(row['EXPECTED_QUERY_RESULT'])
            old_labels = set()
            new_labels = set()
            for r in old_data:
                label = r.get('pasal', r.get('label', ''))
                old_labels.add(label)
            for r in results_truncated:
                label = r.get('pasal', r.get('label', ''))
                new_labels.add(label)
            removed = old_labels - new_labels
            print(f"  Removed false positives: {sorted(removed)}")
        
        # Update row
        row['EXPECTED_CYPHER_QUERY'] = new_cypher
        row['EXPECTED_QUERY_RESULT'] = json.dumps(results_truncated, ensure_ascii=False)
        row['FORMATTED_EXPECTED_QUERY_RESULT'] = json.dumps(results_truncated, indent=2, ensure_ascii=False)
        print()

# Write back
fieldnames = ['TEST_ID', 'QUESTION', 'CATEGORY', 'EXPECTED_CYPHER_QUERY', 'EXPECTED_QUERY_RESULT', 'FORMATTED_EXPECTED_QUERY_RESULT']
with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("CSV updated successfully!")
driver.close()
