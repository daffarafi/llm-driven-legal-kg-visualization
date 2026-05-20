"""
POJK 11/2022 KG Extraction Validation Script
Runs all test cases against Neo4j and reports pass/fail per category.
"""
import os, sys, json, csv, re
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(r'd:\TA\llm-driven-legal-kg-visualization')
os.chdir(PROJECT_ROOT)
load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', ''))
)
db = os.getenv('NEO4J_DATABASE', 'neo4j')

# Load test data
TEST_DATA_PATH = PROJECT_ROOT / 'data' / 'pojk_11_2022_kg_test_data.csv'
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    tests = list(reader)

print(f"Loaded {len(tests)} test cases from {TEST_DATA_PATH.name}")
print(f"Categories: {dict(sorted({cat: sum(1 for t in tests if t['CATEGORY']==cat) for cat in set(t['CATEGORY'] for t in tests)}.items()))}")
print()

# Run tests
results = []
with driver.session(database=db) as session:
    for t in tests:
        test_id = t['TEST_ID']
        category = t['CATEGORY']
        description = t['DESCRIPTION']
        cypher = t['CYPHER_QUERY']
        expected_str = t['EXPECTED_VALUES']
        comparison = t['COMPARISON_TYPE']
        
        try:
            # Execute Cypher
            result = session.run(cypher)
            actual = [dict(r) for r in result]
            
            # Parse expected
            expected = json.loads(expected_str)
            
            # Compare
            if comparison == 'CONTAINS_ALL':
                # Check that all expected records exist in actual
                passed = True
                for exp in expected:
                    found = False
                    for act in actual:
                        if all(str(act.get(k, '')).lower().strip() == str(v).lower().strip() 
                               or str(v).lower() in str(act.get(k, '')).lower()
                               for k, v in exp.items()):
                            found = True
                            break
                    if not found:
                        passed = False
                        break
                        
            elif comparison == 'GREATER_THAN':
                # Check actual value is greater than expected
                exp_val = list(expected[0].values())[0]
                act_val = list(actual[0].values())[0] if actual else 0
                passed = act_val > exp_val
                
            elif comparison == 'EXACT':
                passed = len(actual) == len(expected)
                if passed:
                    for exp, act in zip(expected, actual):
                        if not all(str(act.get(k, '')) == str(v) for k, v in exp.items()):
                            passed = False
                            break
            else:
                passed = len(actual) > 0
            
            status = 'PASS' if passed else 'FAIL'
            actual_summary = json.dumps(actual[:3], ensure_ascii=False, default=str)[:200]
            
        except Exception as e:
            status = 'ERROR'
            actual_summary = str(e)[:200]
            actual = []
        
        results.append({
            'test_id': test_id,
            'category': category,
            'description': description,
            'status': status,
            'actual_count': len(actual) if isinstance(actual, list) else 0,
            'actual_summary': actual_summary,
        })
        
        icon = '✅' if status == 'PASS' else ('❌' if status == 'FAIL' else '⚠️')
        print(f"  {icon} [{test_id}] {description}")
        if status != 'PASS':
            print(f"      Actual: {actual_summary}")

driver.close()

# Summary
print(f"\n{'='*70}")
print("VALIDATION SUMMARY — POJK 11/2022 KG Extraction")
print(f"{'='*70}")

categories = sorted(set(r['category'] for r in results))
total_pass = 0
total_tests = 0

for cat in categories:
    cat_results = [r for r in results if r['category'] == cat]
    cat_pass = sum(1 for r in cat_results if r['status'] == 'PASS')
    cat_total = len(cat_results)
    total_pass += cat_pass
    total_tests += cat_total
    pct = (cat_pass / cat_total * 100) if cat_total else 0
    bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
    status_icon = '✅' if pct == 100 else ('⚠️' if pct >= 50 else '❌')
    print(f"  {status_icon} {cat:20s} {cat_pass:2d}/{cat_total:2d} ({pct:5.1f}%) {bar}")

pct_total = (total_pass / total_tests * 100) if total_tests else 0
print(f"\n  {'─'*55}")
print(f"  TOTAL: {total_pass}/{total_tests} ({pct_total:.1f}%)")
print(f"{'='*70}")

# List all failures
failures = [r for r in results if r['status'] != 'PASS']
if failures:
    print(f"\n❌ FAILURES ({len(failures)}):")
    for f in failures:
        print(f"  [{f['test_id']}] {f['description']}")
        print(f"    Status: {f['status']} | Actual: {f['actual_summary'][:100]}")
