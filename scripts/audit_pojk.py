"""Compare actual Neo4j hierarchy vs expected structure for POJK_11_2022."""
import os, re
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

# ── EXPECTED STRUCTURE ──────────────────────────────────────────
EXPECTED = {
    "BAB I KETENTUAN UMUM": {
        "bagian": {},
        "pasal": {
            "Pasal 1": [],
        }
    },
    "BAB II TATA KELOLA TI BANK": {
        "bagian": {
            "Bagian Kesatu Umum": {
                "Pasal 2": ["Pasal 2 ayat (1)", "Pasal 2 ayat (2)", "Pasal 2 ayat (3)", "Pasal 2 ayat (4)"],
                "Pasal 3": ["Pasal 3 ayat (1)", "Pasal 3 ayat (2)", "Pasal 3 ayat (3)", "Pasal 3 ayat (4)"],
            },
            "Bagian Kedua Penerapan Tata Kelola TI Bank": {
                "Pasal 4": [],
                "Pasal 5": [],
                "Pasal 6": [],
                "Pasal 7": ["Pasal 7 ayat (1)", "Pasal 7 ayat (2)", "Pasal 7 ayat (3)", "Pasal 7 ayat (4)"],
                "Pasal 8": ["Pasal 8 ayat (1)", "Pasal 8 ayat (2)", "Pasal 8 ayat (3)"],
                "Pasal 9": ["Pasal 9 ayat (1)", "Pasal 9 ayat (2)"],
                "Pasal 10": [],
            },
        },
        "pasal": {}
    },
    "BAB III ARSITEKTUR TI BANK": {
        "bagian": {
            "Bagian Kesatu Penyusunan Arsitektur TI Bank": {
                "Pasal 11": ["Pasal 11 ayat (1)", "Pasal 11 ayat (2)", "Pasal 11 ayat (3)", "Pasal 11 ayat (4)", "Pasal 11 ayat (5)"],
            },
            "Bagian Kedua Penyusunan Rencana Strategis TI Bank": {
                "Pasal 12": ["Pasal 12 ayat (1)", "Pasal 12 ayat (2)", "Pasal 12 ayat (3)"],
                "Pasal 13": ["Pasal 13 ayat (1)", "Pasal 13 ayat (2)", "Pasal 13 ayat (3)"],
                "Pasal 14": ["Pasal 14 ayat (1)", "Pasal 14 ayat (2)"],
            },
        },
        "pasal": {}
    },
    "BAB IV PENERAPAN MANAJEMEN RISIKO PENYELENGGARAAN TI BANK": {
        "bagian": {
            "Bagian Kesatu Umum": {
                "Pasal 15": ["Pasal 15 ayat (1)", "Pasal 15 ayat (2)", "Pasal 15 ayat (3)", "Pasal 15 ayat (4)", "Pasal 15 ayat (5)"],
            },
            "Bagian Kedua Pengamanan Informasi dalam Penyelenggaraan TI Bank": {
                "Pasal 16": ["Pasal 16 ayat (1)", "Pasal 16 ayat (2)", "Pasal 16 ayat (3)", "Pasal 16 ayat (4)"],
                "Pasal 17": ["Pasal 17 ayat (1)", "Pasal 17 ayat (2)"],
                "Pasal 18": ["Pasal 18 ayat (1)", "Pasal 18 ayat (2)", "Pasal 18 ayat (3)", "Pasal 18 ayat (4)", "Pasal 18 ayat (5)"],
                "Pasal 19": [],
                "Pasal 20": ["Pasal 20 ayat (1)", "Pasal 20 ayat (2)"],
            },
        },
        "pasal": {}
    },
    "BAB V KETAHANAN DAN KEAMANAN SIBER BANK": {
        "bagian": {},
        "pasal": {
            "Pasal 21": ["Pasal 21 ayat (1)", "Pasal 21 ayat (2)", "Pasal 21 ayat (3)"],
            "Pasal 22": ["Pasal 22 ayat (1)", "Pasal 22 ayat (2)", "Pasal 22 ayat (3)", "Pasal 22 ayat (4)"],
            "Pasal 23": [],
            "Pasal 24": ["Pasal 24 ayat (1)", "Pasal 24 ayat (2)"],
            "Pasal 25": ["Pasal 25 ayat (1)", "Pasal 25 ayat (2)", "Pasal 25 ayat (3)", "Pasal 25 ayat (4)"],
            "Pasal 26": ["Pasal 26 ayat (1)", "Pasal 26 ayat (2)"],
            "Pasal 27": ["Pasal 27 ayat (1)", "Pasal 27 ayat (2)"],
            "Pasal 28": [],
        }
    },
    "BAB VI PENGGUNAAN PIHAK PENYEDIA JASA TI DALAM PENYELENGGARAAN TI BANK": {
        "bagian": {},
        "pasal": {
            "Pasal 29": ["Pasal 29 ayat (1)", "Pasal 29 ayat (2)", "Pasal 29 ayat (3)"],
            "Pasal 30": ["Pasal 30 ayat (1)", "Pasal 30 ayat (2)", "Pasal 30 ayat (3)", "Pasal 30 ayat (4)", "Pasal 30 ayat (5)"],
            "Pasal 31": [],
            "Pasal 32": ["Pasal 32 ayat (1)", "Pasal 32 ayat (2)", "Pasal 32 ayat (3)", "Pasal 32 ayat (4)"],
            "Pasal 33": ["Pasal 33 ayat (1)", "Pasal 33 ayat (2)"],
            "Pasal 34": [],
        }
    },
    "BAB VII PENEMPATAN SISTEM ELEKTRONIK DAN PEMROSESAN TRANSAKSI BERBASIS TI": {
        "bagian": {
            "Bagian Kesatu Penempatan Sistem Elektronik": {
                "Pasal 35": ["Pasal 35 ayat (1)", "Pasal 35 ayat (2)", "Pasal 35 ayat (3)", "Pasal 35 ayat (4)"],
                "Pasal 36": ["Pasal 36 ayat (1)", "Pasal 36 ayat (2)", "Pasal 36 ayat (3)", "Pasal 36 ayat (4)"],
                "Pasal 37": [],
                "Pasal 38": [],
            },
            "Bagian Kedua Pemrosesan Transaksi Berbasis TI": {
                "Pasal 39": ["Pasal 39 ayat (1)", "Pasal 39 ayat (2)", "Pasal 39 ayat (3)", "Pasal 39 ayat (4)", "Pasal 39 ayat (5)", "Pasal 39 ayat (6)", "Pasal 39 ayat (7)"],
            },
            "Bagian Ketiga Tata Cara Permohonan Izin dan Batas Waktu Pelaksanaan Setelah Memperoleh Izin": {
                "Pasal 40": ["Pasal 40 ayat (1)", "Pasal 40 ayat (2)"],
                "Pasal 41": ["Pasal 41 ayat (1)", "Pasal 41 ayat (2)"],
                "Pasal 42": ["Pasal 42 ayat (1)", "Pasal 42 ayat (2)"],
            },
        },
        "pasal": {}
    },
    "BAB VIII PENGELOLAAN DATA DAN PELINDUNGAN DATA PRIBADI DALAM PENYELENGGARAAN TI BANK": {
        "bagian": {
            "Bagian Kesatu Pengelolaan Data oleh Bank": {
                "Pasal 43": ["Pasal 43 ayat (1)", "Pasal 43 ayat (2)", "Pasal 43 ayat (3)"],
            },
            "Bagian Kedua Pelindungan Data Pribadi oleh Bank": {
                "Pasal 44": ["Pasal 44 ayat (1)", "Pasal 44 ayat (2)"],
                "Pasal 45": ["Pasal 45 ayat (1)", "Pasal 45 ayat (2)"],
                "Pasal 46": ["Pasal 46 ayat (1)", "Pasal 46 ayat (2)"],
                "Pasal 47": [],
            },
        },
        "pasal": {}
    },
    "BAB IX PENYEDIAAN JASA TI OLEH BANK": {
        "bagian": {},
        "pasal": {
            "Pasal 48": ["Pasal 48 ayat (1)", "Pasal 48 ayat (2)", "Pasal 48 ayat (3)", "Pasal 48 ayat (4)"],
            "Pasal 49": ["Pasal 49 ayat (1)", "Pasal 49 ayat (2)"],
            "Pasal 50": ["Pasal 50 ayat (1)", "Pasal 50 ayat (2)"],
            "Pasal 51": ["Pasal 51 ayat (1)", "Pasal 51 ayat (2)"],
            "Pasal 52": [],
        }
    },
    "BAB X PENGENDALIAN DAN AUDIT INTERN DALAM PENYELENGGARAAN TI BANK": {
        "bagian": {
            "Bagian Kesatu Pengendalian Intern Bank dalam Penyelenggaraan TI": {
                "Pasal 53": ["Pasal 53 ayat (1)", "Pasal 53 ayat (2)", "Pasal 53 ayat (3)", "Pasal 53 ayat (4)"],
            },
            "Bagian Kedua Audit Intern dalam Penyelenggaraan TI": {
                "Pasal 54": ["Pasal 54 ayat (1)", "Pasal 54 ayat (2)", "Pasal 54 ayat (3)", "Pasal 54 ayat (4)"],
                "Pasal 55": ["Pasal 55 ayat (1)", "Pasal 55 ayat (2)", "Pasal 55 ayat (3)"],
                "Pasal 56": ["Pasal 56 ayat (1)", "Pasal 56 ayat (2)"],
                "Pasal 57": [],
            },
        },
        "pasal": {}
    },
    "BAB XI PELAPORAN": {
        "bagian": {
            "Bagian Kesatu Laporan Penyelenggaraan TI": {
                "Pasal 58": ["Pasal 58 ayat (1)", "Pasal 58 ayat (2)", "Pasal 58 ayat (3)", "Pasal 58 ayat (4)"],
                "Pasal 59": [],
            },
            "Bagian Kedua Laporan Insidentil": {
                "Pasal 60": ["Pasal 60 ayat (1)", "Pasal 60 ayat (2)", "Pasal 60 ayat (3)", "Pasal 60 ayat (4)", "Pasal 60 ayat (5)"],
            },
            "Bagian Ketiga Laporan Realisasi Penyelenggaraan TI Bank": {
                "Pasal 61": ["Pasal 61 ayat (1)", "Pasal 61 ayat (2)"],
            },
            "Bagian Keempat Tata Cara Penyampaian Laporan": {
                "Pasal 62": ["Pasal 62 ayat (1)", "Pasal 62 ayat (2)", "Pasal 62 ayat (3)"],
                "Pasal 63": ["Pasal 63 ayat (1)", "Pasal 63 ayat (2)"],
                "Pasal 64": [],
                "Pasal 65": [],
            },
        },
        "pasal": {}
    },
    "BAB XII PENILAIAN TINGKAT MATURITAS DIGITAL BANK": {
        "bagian": {},
        "pasal": {
            "Pasal 66": ["Pasal 66 ayat (1)", "Pasal 66 ayat (2)", "Pasal 66 ayat (3)", "Pasal 66 ayat (4)", "Pasal 66 ayat (5)", "Pasal 66 ayat (6)"],
        }
    },
    "BAB XIII KETENTUAN PERALIHAN": {
        "bagian": {},
        "pasal": {
            "Pasal 67": [],
            "Pasal 68": [],
            "Pasal 69": [],
        }
    },
    "BAB XIV KETENTUAN PENUTUP": {
        "bagian": {},
        "pasal": {
            "Pasal 70": [],
            "Pasal 71": [],
            "Pasal 72": [],
            "Pasal 73": [],
        }
    },
}

# ── AUDIT ───────────────────────────────────────────────────────

print("=" * 70)
print(f"HIERARCHY AUDIT: {DOC}")
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

print(f"\n--- REGULASI -> BAB ---")
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
    print("  All Bab present and connected. OK")

# 2. For each Bab, check Bagian and direct Pasal
print(f"\n--- BAB -> BAGIAN / PASAL ---")
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
        print(f"\n  [{bab_label}] Bagian issues:")
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

    # Pasal under Bagian should NOT be direct
    expected_bagian_pasal = set()
    for bg_label, bg_pasal in bab_expected["bagian"].items():
        expected_bagian_pasal.update(bg_pasal.keys())

    should_not_be_direct = actual_direct_pasal & expected_bagian_pasal
    if should_not_be_direct:
        print(f"\n  [{bab_label}] Pasal WRONGLY direct (should be via Bagian):")
        for p in sorted(should_not_be_direct, key=lambda x: int(re.search(r'\d+', x).group())):
            issues.append(f"WRONG PATH: {p} direct under {bab_label}, should be via Bagian")
            print(f"    [WRONG PATH] {p}")

    missing_direct = expected_direct_pasal - actual_direct_pasal
    extra_direct = actual_direct_pasal - expected_direct_pasal - expected_bagian_pasal

    if missing_direct:
        print(f"\n  [{bab_label}] Missing direct Pasal:")
        for p in sorted(missing_direct, key=lambda x: int(re.search(r'\d+', x).group())):
            issues.append(f"MISSING DIRECT PASAL in {bab_label}: {p}")
            print(f"    [MISSING] {p}")
    if extra_direct:
        print(f"\n  [{bab_label}] Extra direct Pasal:")
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
            print(f"\n  [{bg_label}] Pasal issues:")
            for p in sorted(missing_bgp, key=lambda x: int(re.search(r'\d+', x).group())):
                issues.append(f"MISSING PASAL in {bg_label}: {p}")
                print(f"    [MISSING] {p}")
            for p in sorted(extra_bgp, key=lambda x: int(re.search(r'\d+', x).group())):
                issues.append(f"EXTRA PASAL in {bg_label}: {p}")
                print(f"    [EXTRA]   {p}")

# 3. Check MEMILIKI_AYAT
print(f"\n--- PASAL -> AYAT ---")
all_expected_pasal = {}
for bab_label, bab_data in EXPECTED.items():
    for pasal, ayats in bab_data["pasal"].items():
        all_expected_pasal[pasal] = ayats
    for bg_label, bg_pasal in bab_data["bagian"].items():
        for pasal, ayats in bg_pasal.items():
            all_expected_pasal[pasal] = ayats

ayat_issues_found = False
for pasal_label in sorted(all_expected_pasal.keys(), key=lambda x: int(re.search(r'\d+', x).group())):
    expected_ayats = all_expected_pasal[pasal_label]
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
    print("  No issues! Hierarchy matches expected structure perfectly.")

driver.close()
