"""Fix POJK_11_2022 hierarchy issues:
1. Rename 3 truncated Bab labels
2. Investigate and fix Pasal 15 wrong Bagian connection
"""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(".env")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
)
DB = os.getenv("NEO4J_DATABASE", "neo4j")
DOC = "POJK_11_2022"

def run(q, params=None):
    with driver.session(database=DB) as s:
        return s.run(q, params or {}).data()

def run_single(q, params=None):
    with driver.session(database=DB) as s:
        return s.run(q, params or {}).single()

# ═══════════════════════════════════════════════════════════════
# FIX 1: Rename 3 truncated Bab labels
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("FIX 1: Renaming truncated Bab labels")
print("=" * 70)

renames = [
    ("BAB IV PENERAPAN MANAJEMEN RISIKO", "BAB IV PENERAPAN MANAJEMEN RISIKO PENYELENGGARAAN TI BANK"),
    ("BAB VIII PENGELOLAAN DATA DAN PELINDUNGAN DATA PRIBADI", "BAB VIII PENGELOLAAN DATA DAN PELINDUNGAN DATA PRIBADI DALAM PENYELENGGARAAN TI BANK"),
    ("BAB X PENGENDALIAN DAN AUDIT INTERN", "BAB X PENGENDALIAN DAN AUDIT INTERN DALAM PENYELENGGARAAN TI BANK"),
]

for old_label, new_label in renames:
    result = run_single("""
        MATCH (b:Bab {label: $old, source_document_id: $doc})
        SET b.label = $new
        RETURN count(b) AS updated
    """, {"doc": DOC, "old": old_label, "new": new_label})
    print(f"  [{result['updated']} updated] {old_label}")
    print(f"    -> {new_label}")

# ═══════════════════════════════════════════════════════════════
# INVESTIGATE: Pasal 15 - which "Bagian Kesatu Umum" is it in?
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("INVESTIGATE: Pasal 15 connections")
print("=" * 70)

# Find all Bagian named "Bagian Kesatu Umum"
bagian_umum = run("""
    MATCH (bg:Bagian {source_document_id: $doc})
    WHERE bg.label = 'Bagian Kesatu Umum'
    OPTIONAL MATCH (bab:Bab)-[:MEMUAT]->(bg)
    RETURN bg.label AS bagian, elementId(bg) AS bg_id, bab.label AS parent_bab
""", {"doc": DOC})
print(f"\nAll 'Bagian Kesatu Umum' nodes ({len(bagian_umum)}):")
for b in bagian_umum:
    print(f"  {b['bagian']} (id={b['bg_id']}) -> parent: {b['parent_bab']}")

# Check which Bagian has Pasal 15
p15_parents = run("""
    MATCH (parent)-[:MEMUAT]->(p:Pasal {label: 'Pasal 15', source_document_id: $doc})
    RETURN labels(parent) AS parent_labels, parent.label AS parent_label, elementId(parent) AS parent_id
""", {"doc": DOC})
print(f"\nPasal 15 parent connections:")
for p in p15_parents:
    print(f"  {p['parent_labels']} '{p['parent_label']}' (id={p['parent_id']})")

# Check what Bagian Kesatu Umum in BAB II has
bg_umum_bab2 = run("""
    MATCH (bab:Bab)-[:MEMUAT]->(bg:Bagian {label: 'Bagian Kesatu Umum'})
    WHERE bab.source_document_id = $doc AND bab.label CONTAINS 'BAB II'
    MATCH (bg)-[:MEMUAT]->(p:Pasal)
    RETURN bg.label AS bagian, p.label AS pasal, elementId(bg) AS bg_id
    ORDER BY p.label
""", {"doc": DOC})
print(f"\nBagian Kesatu Umum (BAB II) children:")
for b in bg_umum_bab2:
    print(f"  {b['pasal']} (bg_id={b['bg_id']})")

# Check what Bagian Kesatu Umum in BAB IV has
bg_umum_bab4 = run("""
    MATCH (bab:Bab)-[:MEMUAT]->(bg:Bagian {label: 'Bagian Kesatu Umum'})
    WHERE bab.source_document_id = $doc AND bab.label CONTAINS 'BAB IV'
    MATCH (bg)-[:MEMUAT]->(p:Pasal)
    RETURN bg.label AS bagian, p.label AS pasal, elementId(bg) AS bg_id
    ORDER BY p.label
""", {"doc": DOC})
print(f"\nBagian Kesatu Umum (BAB IV) children:")
for b in bg_umum_bab4:
    print(f"  {b['pasal']} (bg_id={b['bg_id']})")

# ═══════════════════════════════════════════════════════════════
# FIX 2: Remove wrong edge Bagian Kesatu Umum (BAB II) -> Pasal 15
#         Pasal 15 should only be under Bagian Kesatu Umum (BAB IV)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FIX 2: Fixing Pasal 15 wrong Bagian connection")
print("=" * 70)

# Get the Bagian Kesatu Umum under BAB II
bab2_bg = run("""
    MATCH (bab:Bab)-[:MEMUAT]->(bg:Bagian {label: 'Bagian Kesatu Umum'})
    WHERE bab.source_document_id = $doc AND bab.label CONTAINS 'BAB II'
    RETURN elementId(bg) AS bg_id
""", {"doc": DOC})

if bab2_bg:
    bg_id = bab2_bg[0]["bg_id"]
    # Delete wrong edge: Bagian Kesatu Umum (BAB II) -> Pasal 15
    result = run_single("""
        MATCH (bg:Bagian)-[r:MEMUAT]->(p:Pasal {label: 'Pasal 15', source_document_id: $doc})
        WHERE elementId(bg) = $bg_id
        DELETE r
        RETURN count(r) AS deleted
    """, {"doc": DOC, "bg_id": bg_id})
    print(f"  Deleted wrong edge (BAB II/Bagian Kesatu Umum -> Pasal 15): {result['deleted']}")

# Verify Pasal 15 is still in correct Bagian (BAB IV)
verify = run("""
    MATCH (bab:Bab)-[:MEMUAT]->(bg:Bagian)-[:MEMUAT]->(p:Pasal {label: 'Pasal 15', source_document_id: $doc})
    RETURN bab.label AS bab, bg.label AS bagian
""", {"doc": DOC})
print(f"\n  Pasal 15 now connected via:")
for v in verify:
    print(f"    {v['bab']} -> {v['bagian']} -> Pasal 15  OK")

# ═══════════════════════════════════════════════════════════════
# VERIFY: Re-run full check
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERIFY: Post-fix Bab connections")
print("=" * 70)

all_babs = run("""
    MATCH (reg:Regulasi {source_document_id: $doc})-[:MEMUAT]->(b:Bab)
    RETURN b.label AS bab
    ORDER BY b.label
""", {"doc": DOC})
for b in all_babs:
    print(f"  {b['bab']}")

driver.close()
print("\n" + "=" * 70)
print("ALL FIXES APPLIED")
print("=" * 70)
