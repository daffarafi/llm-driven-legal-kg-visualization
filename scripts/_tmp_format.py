"""Fill FORMATTED_EXPECTED_QUERY_RESULT column with pretty-printed JSON."""
import csv
import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

input_path = r'd:\TA\[SFT] Regulation Fine Tuning for Knowledge Graph - QUESTION_TO_CYPHER_QUERY_DATA_TEST.csv'

with open(input_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Loaded {len(rows)} rows")
print(f"Columns: {list(rows[0].keys())}")

filled = 0
for row in rows:
    raw = row.get('EXPECTED_QUERY_RESULT', '').strip()
    if raw:
        try:
            data = json.loads(raw)
            row['FORMATTED_EXPECTED_QUERY_RESULT'] = json.dumps(data, indent=2, ensure_ascii=False)
            filled += 1
        except json.JSONDecodeError as e:
            print(f"  [{row['TEST_ID']}] JSON parse error: {str(e)[:80]}")
            row['FORMATTED_EXPECTED_QUERY_RESULT'] = raw
    else:
        row['FORMATTED_EXPECTED_QUERY_RESULT'] = ''

# Write back
fieldnames = ['TEST_ID', 'QUESTION', 'CATEGORY', 'EXPECTED_CYPHER_QUERY', 'EXPECTED_QUERY_RESULT', 'FORMATTED_EXPECTED_QUERY_RESULT']
with open(input_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone: filled {filled}/{len(rows)} FORMATTED_EXPECTED_QUERY_RESULT cells")
