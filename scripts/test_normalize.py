import json, sys, os
sys.path.insert(0, '.')
from pipeline.extract.structure_parser import normalize_structural_headers

# Test with actual noisy lines from all documents
test_cases = [
    # Should be normalized (structural headers)
    ("BABVIII ...", "BAB VIII"),
    ("REPUBLIK INDONESIA BAB VIII", "BAB VIII"),
    ("BAB IX. . .", "BAB IX"),
    ("UALIK INDONES BAB Ix", "BAB IX"),
    ("BABxII ...", "BAB XII"),
    ("REPUBLIK INDONESIA BAB xII", "BAB XII"),
    ("BABII ...", "BAB II"),
    ("BABI.", "BAB I"),
    ("BABV...", "BAB V"),
    ("BABVI...", "BAB VI"),
    ("BABVII ...", "BAB VII"),
    ("BABX...", "BAB X"),
    ("REPUBLIK INDONES BAB VI", "BAB VI"),
    ("REPUELIK INDONE5IA BAB XV", "BAB XV"),
    ("REPUBTIK TNDONESIA BAB X", "BAB X"),
    ("REFUELIK TNDONESIA BAB IX", "BAB IX"),
    ("REPUBtlK INDONESIA BAB XII", "BAB XII"),
    
    # Should NOT be normalized (content lines)
    ("yang demikian pesat telah menyebabkan perubahan kegiatan", None),
    ("penetrasi pengguna internet di Indonesia disebabkan oleh", None),
    ("pada bab ini, menjadi panduan untuk mengidentifikasi", None),
    ("sub bab III.A, dimana memiliki struktur", None),
    ("Pada Sub Bab II.A, telah dijelaskan bahwa", None),
    ("BAB VII", None),  # Already clean, keep as-is
    ("BAB I", None),    # Already clean
]

print(f"{'Input':<55} {'Expected':<15} {'Got':<15} {'OK?'}")
print("-" * 100)

passed = 0
failed = 0
for inp, expected in test_cases:
    result = normalize_structural_headers(inp).strip()
    if expected is None:
        expected = inp.strip()  # Should stay unchanged
    ok = result == expected
    status = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
    else:
        passed += 1
    print(f"{inp:<55} {expected:<15} {result:<15} {status}")

print(f"\n{passed} passed, {failed} failed")
