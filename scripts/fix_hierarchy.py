"""Fix redundant edges + output clean hierarchy Cypher for UU_11_2008."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(".env")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
)
DB = os.getenv("NEO4J_DATABASE", "neo4j")
DOC = "UU_11_2008"

def run(q, params=None):
    with driver.session(database=DB) as s:
        return s.run(q, params or {}).data()

def run_single(q, params=None):
    with driver.session(database=DB) as s:
        return s.run(q, params or {}).single()

# ── STEP 1: Delete redundant BAB IV -> Pasal edges ─────────────
print("=" * 70)
print("STEP 1: Removing redundant MEMUAT edges (BAB IV -> Pasal 13-16)")
print("=" * 70)

bab4 = "BAB IV PENYELENGGARAAN SERTIFIKASI ELEKTRONIK DAN SISTEM ELEKTRONIK"
for pasal in ["Pasal 13", "Pasal 14", "Pasal 15", "Pasal 16"]:
    result = run_single("""
        MATCH (b:Bab {label: $bab, source_document_id: $doc})-[r:MEMUAT]->(p:Pasal {label: $pasal, source_document_id: $doc})
        DELETE r
        RETURN count(r) AS deleted
    """, {"doc": DOC, "bab": bab4, "pasal": pasal})
    print(f"  Deleted BAB IV -[MEMUAT]-> {pasal}: {result['deleted']} edge(s)")

# Verify Bagian path still exists
print("\n  Verifying Bagian paths still intact:")
verify = run("""
    MATCH (b:Bab {label: $bab, source_document_id: $doc})-[:MEMUAT]->(bg:Bagian)-[:MEMUAT]->(p:Pasal)
    RETURN bg.label AS bagian, p.label AS pasal
    ORDER BY bg.label, p.label
""", {"doc": DOC, "bab": bab4})
for v in verify:
    print(f"    BAB IV -> {v['bagian']} -> {v['pasal']} OK")

# ── STEP 2: Output full hierarchy query ─────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Full hierarchy (Regulasi -> Bab -> Bagian? -> Pasal -> Ayat)")
print("=" * 70)

hierarchy = run("""
    MATCH (reg:Regulasi {source_document_id: $doc})
    OPTIONAL MATCH (reg)-[:MEMUAT]->(bab:Bab)
    OPTIONAL MATCH (bab)-[:MEMUAT]->(bagian:Bagian)
    OPTIONAL MATCH (bab)-[:MEMUAT]->(pasal_direct:Pasal)
    OPTIONAL MATCH (bagian)-[:MEMUAT]->(pasal_bagian:Pasal)
    OPTIONAL MATCH (pasal_direct)-[:MEMILIKI_AYAT]->(ayat_direct:Ayat)
    OPTIONAL MATCH (pasal_bagian)-[:MEMILIKI_AYAT]->(ayat_bagian:Ayat)
    RETURN reg.label AS regulasi,
           bab.label AS bab,
           bagian.label AS bagian,
           pasal_direct.label AS pasal_direct,
           pasal_bagian.label AS pasal_bagian,
           ayat_direct.label AS ayat_direct,
           ayat_bagian.label AS ayat_bagian
    ORDER BY bab.label, bagian.label,
             pasal_direct.label, pasal_bagian.label,
             ayat_direct.label, ayat_bagian.label
""", {"doc": DOC})

# Build tree
import re

def pasal_sort_key(label):
    if not label:
        return (0,)
    m = re.search(r'(\d+)', label)
    return (int(m.group(1)),) if m else (0,)

def ayat_sort_key(label):
    if not label:
        return (0, 0)
    nums = re.findall(r'\d+', label)
    return tuple(int(n) for n in nums) if nums else (0, 0)

# Collect unique entries
tree = {}  # bab -> {bagian -> {pasal -> [ayat]}, "_direct": {pasal -> [ayat]}}

for row in hierarchy:
    bab = row["bab"]
    if not bab:
        continue

    if bab not in tree:
        tree[bab] = {"_direct": {}, "_bagian": {}}

    # Direct pasal
    pd = row["pasal_direct"]
    if pd:
        if pd not in tree[bab]["_direct"]:
            tree[bab]["_direct"][pd] = set()
        ad = row["ayat_direct"]
        if ad:
            tree[bab]["_direct"][pd].add(ad)

    # Bagian pasal
    bg = row["bagian"]
    pb = row["pasal_bagian"]
    if bg:
        if bg not in tree[bab]["_bagian"]:
            tree[bab]["_bagian"][bg] = {}
        if pb:
            if pb not in tree[bab]["_bagian"][bg]:
                tree[bab]["_bagian"][bg][pb] = set()
            ab = row["ayat_bagian"]
            if ab:
                tree[bab]["_bagian"][bg][pb].add(ab)

# Print tree
reg_label = hierarchy[0]["regulasi"] if hierarchy else DOC
print(f"\n- (Regulasi) {reg_label}")

bab_roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
             "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13}

def bab_sort_key(bab_label):
    m = re.search(r'BAB\s+([IVXLC]+)', bab_label)
    if m:
        return bab_roman.get(m.group(1), 99)
    return 99

for bab in sorted(tree.keys(), key=bab_sort_key):
    data = tree[bab]
    print(f"  - (Bab) {bab}")

    # Bagian first
    for bg in sorted(data["_bagian"].keys()):
        pasals = data["_bagian"][bg]
        print(f"    - (Bagian) {bg}")
        for pasal in sorted(pasals.keys(), key=pasal_sort_key):
            ayats = pasals[pasal]
            print(f"      - (Pasal) {pasal}")
            for ayat in sorted(ayats, key=ayat_sort_key):
                print(f"        - (Ayat) {ayat}")

    # Direct pasal
    for pasal in sorted(data["_direct"].keys(), key=pasal_sort_key):
        ayats = data["_direct"][pasal]
        print(f"    - (Pasal) {pasal}")
        for ayat in sorted(ayats, key=ayat_sort_key):
            print(f"      - (Ayat) {ayat}")

driver.close()
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
