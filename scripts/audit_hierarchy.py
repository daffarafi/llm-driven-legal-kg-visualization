"""Compare actual Neo4j hierarchy vs expected structure for UU_11_2008."""
import os, json, re
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

# ── EXPECTED STRUCTURE (from user) ──────────────────────────────
EXPECTED = {
    "BAB I KETENTUAN UMUM": {
        "bagian": {},
        "pasal": {
            "Pasal 1": [],
            "Pasal 2": [],
        }
    },
    "BAB II ASAS DAN TUJUAN": {
        "bagian": {},
        "pasal": {
            "Pasal 3": [],
            "Pasal 4": [],
        }
    },
    "BAB III INFORMASI, DOKUMEN, DAN TANDA TANGAN ELEKTRONIK": {
        "bagian": {},
        "pasal": {
            "Pasal 5": ["Pasal 5 ayat (1)", "Pasal 5 ayat (2)", "Pasal 5 ayat (3)", "Pasal 5 ayat (4)"],
            "Pasal 6": [],
            "Pasal 7": [],
            "Pasal 8": ["Pasal 8 ayat (1)", "Pasal 8 ayat (2)", "Pasal 8 ayat (3)", "Pasal 8 ayat (4)"],
            "Pasal 9": [],
            "Pasal 10": ["Pasal 10 ayat (1)", "Pasal 10 ayat (2)"],
            "Pasal 11": ["Pasal 11 ayat (1)", "Pasal 11 ayat (2)"],
            "Pasal 12": ["Pasal 12 ayat (1)", "Pasal 12 ayat (2)", "Pasal 12 ayat (3)"],
        }
    },
    "BAB IV PENYELENGGARAAN SERTIFIKASI ELEKTRONIK DAN SISTEM ELEKTRONIK": {
        "bagian": {
            "Bagian Kesatu Penyelenggaraan Sertifikasi Elektronik": {
                "Pasal 13": ["Pasal 13 ayat (1)", "Pasal 13 ayat (2)", "Pasal 13 ayat (3)", "Pasal 13 ayat (4)", "Pasal 13 ayat (5)", "Pasal 13 ayat (6)"],
                "Pasal 14": [],
            },
            "Bagian Kedua Penyelenggaraan Sistem Elektronik": {
                "Pasal 15": ["Pasal 15 ayat (1)", "Pasal 15 ayat (2)", "Pasal 15 ayat (3)"],
                "Pasal 16": ["Pasal 16 ayat (1)", "Pasal 16 ayat (2)"],
            },
        },
        "pasal": {}  # BAB IV has NO direct pasal, all via Bagian
    },
    "BAB V TRANSAKSI ELEKTRONIK": {
        "bagian": {},
        "pasal": {
            "Pasal 17": ["Pasal 17 ayat (1)", "Pasal 17 ayat (2)", "Pasal 17 ayat (3)"],
            "Pasal 18": ["Pasal 18 ayat (1)", "Pasal 18 ayat (2)", "Pasal 18 ayat (3)", "Pasal 18 ayat (4)", "Pasal 18 ayat (5)"],
            "Pasal 19": [],
            "Pasal 20": ["Pasal 20 ayat (1)", "Pasal 20 ayat (2)"],
            "Pasal 21": ["Pasal 21 ayat (1)", "Pasal 21 ayat (2)", "Pasal 21 ayat (3)", "Pasal 21 ayat (4)", "Pasal 21 ayat (5)"],
            "Pasal 22": ["Pasal 22 ayat (1)", "Pasal 22 ayat (2)"],
        }
    },
    "BAB VI NAMA DOMAIN, HAK KEKAYAAN INTELEKTUAL, DAN PERLINDUNGAN HAK PRIBADI": {
        "bagian": {},
        "pasal": {
            "Pasal 23": ["Pasal 23 ayat (1)", "Pasal 23 ayat (2)", "Pasal 23 ayat (3)"],
            "Pasal 24": ["Pasal 24 ayat (1)", "Pasal 24 ayat (2)", "Pasal 24 ayat (3)", "Pasal 24 ayat (4)"],
            "Pasal 25": [],
            "Pasal 26": ["Pasal 26 ayat (1)", "Pasal 26 ayat (2)"],
        }
    },
    "BAB VII PERBUATAN YANG DILARANG": {
        "bagian": {},
        "pasal": {
            "Pasal 27": ["Pasal 27 ayat (1)", "Pasal 27 ayat (2)", "Pasal 27 ayat (3)", "Pasal 27 ayat (4)"],
            "Pasal 28": ["Pasal 28 ayat (1)", "Pasal 28 ayat (2)"],
            "Pasal 29": [],
            "Pasal 30": ["Pasal 30 ayat (1)", "Pasal 30 ayat (2)", "Pasal 30 ayat (3)"],
            "Pasal 31": ["Pasal 31 ayat (1)", "Pasal 31 ayat (2)", "Pasal 31 ayat (3)", "Pasal 31 ayat (4)"],
            "Pasal 32": ["Pasal 32 ayat (1)", "Pasal 32 ayat (2)", "Pasal 32 ayat (3)"],
            "Pasal 33": [],
            "Pasal 34": ["Pasal 34 ayat (1)", "Pasal 34 ayat (2)"],
            "Pasal 35": [],
            "Pasal 36": [],
            "Pasal 37": [],
        }
    },
    "BAB VIII PENYELESAIAN SENGKETA": {
        "bagian": {},
        "pasal": {
            "Pasal 38": ["Pasal 38 ayat (1)", "Pasal 38 ayat (2)"],
            "Pasal 39": ["Pasal 39 ayat (1)", "Pasal 39 ayat (2)"],
        }
    },
    "BAB IX PERAN PEMERINTAH DAN PERAN MASYARAKAT": {
        "bagian": {},
        "pasal": {
            "Pasal 40": ["Pasal 40 ayat (1)", "Pasal 40 ayat (2)", "Pasal 40 ayat (3)", "Pasal 40 ayat (4)", "Pasal 40 ayat (5)", "Pasal 40 ayat (6)"],
            "Pasal 41": ["Pasal 41 ayat (1)", "Pasal 41 ayat (2)", "Pasal 41 ayat (3)"],
        }
    },
    "BAB X PENYIDIKAN": {
        "bagian": {},
        "pasal": {
            "Pasal 42": [],
            "Pasal 43": ["Pasal 43 ayat (1)", "Pasal 43 ayat (2)", "Pasal 43 ayat (3)", "Pasal 43 ayat (4)", "Pasal 43 ayat (5)", "Pasal 43 ayat (6)", "Pasal 43 ayat (7)", "Pasal 43 ayat (8)"],
            "Pasal 44": [],
        }
    },
    "BAB XI KETENTUAN PIDANA": {
        "bagian": {},
        "pasal": {
            "Pasal 45": ["Pasal 45 ayat (1)", "Pasal 45 ayat (2)", "Pasal 45 ayat (3)"],
            "Pasal 46": ["Pasal 46 ayat (1)", "Pasal 46 ayat (2)", "Pasal 46 ayat (3)"],
            "Pasal 47": [],
            "Pasal 48": ["Pasal 48 ayat (1)", "Pasal 48 ayat (2)", "Pasal 48 ayat (3)"],
            "Pasal 49": [],
            "Pasal 50": [],
            "Pasal 51": ["Pasal 51 ayat (1)", "Pasal 51 ayat (2)"],
            "Pasal 52": ["Pasal 52 ayat (1)", "Pasal 52 ayat (2)", "Pasal 52 ayat (3)", "Pasal 52 ayat (4)"],
        }
    },
    "BAB XII KETENTUAN PERALIHAN": {
        "bagian": {},
        "pasal": {
            "Pasal 53": [],
        }
    },
    "BAB XIII KETENTUAN PENUTUP": {
        "bagian": {},
        "pasal": {
            "Pasal 54": ["Pasal 54 ayat (1)", "Pasal 54 ayat (2)"],
        }
    },
}

# ── FETCH ACTUAL FROM NEO4J ─────────────────────────────────────

print("=" * 70)
print(f"HIERARCHY COMPARISON: {DOC}")
print("Expected vs Actual")
print("=" * 70)

issues = []

# 1. Regulasi -> Bab
actual_babs = {r["bab"] for r in run("""
    MATCH (reg:Regulasi)-[:MEMUAT]->(b:Bab)
    WHERE reg.source_document_id = $doc
    RETURN b.label AS bab
""", {"doc": DOC})}
expected_babs = set(EXPECTED.keys())

missing_babs = expected_babs - actual_babs
extra_babs = actual_babs - expected_babs

print("\n--- REGULASI -> BAB ---")
print(f"Expected: {len(expected_babs)} bab, Actual: {len(actual_babs)} bab")
if missing_babs:
    for b in sorted(missing_babs):
        issues.append(f"MISSING BAB: {b}")
        print(f"  [MISSING] {b}")
if extra_babs:
    for b in sorted(extra_babs):
        issues.append(f"EXTRA BAB: {b}")
        print(f"  [EXTRA]   {b}")
if not missing_babs and not extra_babs:
    print("  All 13 Bab present and connected. OK")

# 2. For each Bab, check Bagian and direct Pasal
print("\n--- BAB -> BAGIAN / PASAL ---")
for bab_label, bab_expected in EXPECTED.items():
    if bab_label not in actual_babs:
        continue

    # Check Bagian
    actual_bagians = {r["bagian"] for r in run("""
        MATCH (b:Bab)-[:MEMUAT]->(bg:Bagian)
        WHERE b.source_document_id = $doc AND b.label = $bab
        RETURN bg.label AS bagian
    """, {"doc": DOC, "bab": bab_label})}
    expected_bagians = set(bab_expected["bagian"].keys())

    missing_bg = expected_bagians - actual_bagians
    extra_bg = actual_bagians - expected_bagians

    if missing_bg or extra_bg:
        print(f"\n  [{bab_label}] Bagian:")
        for b in sorted(missing_bg):
            issues.append(f"MISSING BAGIAN in {bab_label}: {b}")
            print(f"    [MISSING] {b}")
        for b in sorted(extra_bg):
            issues.append(f"EXTRA BAGIAN in {bab_label}: {b}")
            print(f"    [EXTRA]   {b}")

    # Check direct Pasal (Bab -> Pasal)
    actual_direct_pasal = {r["pasal"] for r in run("""
        MATCH (b:Bab)-[:MEMUAT]->(p:Pasal)
        WHERE b.source_document_id = $doc AND b.label = $bab
        RETURN p.label AS pasal
    """, {"doc": DOC, "bab": bab_label})}
    expected_direct_pasal = set(bab_expected["pasal"].keys())

    # For Bab with Bagian, pasal under Bagian should NOT be direct
    expected_bagian_pasal = set()
    for bg_label, bg_pasal in bab_expected["bagian"].items():
        expected_bagian_pasal.update(bg_pasal.keys())

    # Pasal that are directly under Bab but should be under Bagian
    should_not_be_direct = actual_direct_pasal & expected_bagian_pasal
    if should_not_be_direct:
        print(f"\n  [{bab_label}] Pasal WRONGLY direct (should be via Bagian only):")
        for p in sorted(should_not_be_direct, key=lambda x: int(re.search(r'\d+', x).group())):
            issues.append(f"WRONG PATH: {p} is direct under {bab_label} but should be via Bagian only")
            print(f"    [WRONG PATH] {p} -- has direct edge to Bab, should only be via Bagian")

    missing_direct = expected_direct_pasal - actual_direct_pasal
    extra_direct = actual_direct_pasal - expected_direct_pasal - expected_bagian_pasal

    if missing_direct:
        print(f"\n  [{bab_label}] Missing direct Pasal:")
        for p in sorted(missing_direct, key=lambda x: int(re.search(r'\d+', x).group())):
            issues.append(f"MISSING DIRECT PASAL in {bab_label}: {p}")
            print(f"    [MISSING] {p}")

    if extra_direct:
        print(f"\n  [{bab_label}] Extra direct Pasal (not expected):")
        for p in sorted(extra_direct, key=lambda x: int(re.search(r'\d+', x).group())):
            issues.append(f"EXTRA DIRECT PASAL in {bab_label}: {p}")
            print(f"    [EXTRA]   {p}")

    # Check Bagian -> Pasal
    for bg_label, bg_pasal in bab_expected["bagian"].items():
        if bg_label not in actual_bagians:
            continue
        actual_bg_pasal = {r["pasal"] for r in run("""
            MATCH (bg:Bagian)-[:MEMUAT]->(p:Pasal)
            WHERE bg.source_document_id = $doc AND bg.label = $bg
            RETURN p.label AS pasal
        """, {"doc": DOC, "bg": bg_label})}
        expected_bg_pasal_set = set(bg_pasal.keys())

        missing_bgp = expected_bg_pasal_set - actual_bg_pasal
        extra_bgp = actual_bg_pasal - expected_bg_pasal_set

        if missing_bgp or extra_bgp:
            print(f"\n  [{bab_label} -> {bg_label}] Pasal:")
            for p in sorted(missing_bgp):
                issues.append(f"MISSING PASAL in {bg_label}: {p}")
                print(f"    [MISSING] {p}")
            for p in sorted(extra_bgp):
                issues.append(f"EXTRA PASAL in {bg_label}: {p}")
                print(f"    [EXTRA]   {p}")

# 3. Check MEMILIKI_AYAT for every Pasal
print("\n--- PASAL -> AYAT ---")
all_expected_pasal = {}
for bab_label, bab_data in EXPECTED.items():
    for pasal, ayats in bab_data["pasal"].items():
        all_expected_pasal[pasal] = ayats
    for bg_label, bg_pasal in bab_data["bagian"].items():
        for pasal, ayats in bg_pasal.items():
            all_expected_pasal[pasal] = ayats

ayat_issues_found = False
for pasal_label, expected_ayats in sorted(all_expected_pasal.items(), key=lambda x: int(re.search(r'\d+', x[0]).group())):
    actual_ayats = {r["ayat"] for r in run("""
        MATCH (p:Pasal)-[:MEMILIKI_AYAT]->(a:Ayat)
        WHERE p.source_document_id = $doc AND p.label = $pasal
        RETURN a.label AS ayat
    """, {"doc": DOC, "pasal": pasal_label})}
    expected_ayat_set = set(expected_ayats)

    missing_ayat = expected_ayat_set - actual_ayats
    extra_ayat = actual_ayats - expected_ayat_set

    if missing_ayat or extra_ayat:
        ayat_issues_found = True
        print(f"\n  [{pasal_label}]")
        for a in sorted(missing_ayat):
            issues.append(f"MISSING AYAT in {pasal_label}: {a}")
            print(f"    [MISSING] {a}")
        for a in sorted(extra_ayat):
            issues.append(f"EXTRA AYAT in {pasal_label}: {a}")
            print(f"    [EXTRA]   {a}")

if not ayat_issues_found:
    print("  All Pasal -> Ayat connections match expected. OK")

# ── SUMMARY ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"SUMMARY: {len(issues)} issues found")
print("=" * 70)
for i, issue in enumerate(issues, 1):
    print(f"  {i}. {issue}")

if not issues:
    print("  No issues found! Hierarchy matches expected structure perfectly.")

driver.close()
