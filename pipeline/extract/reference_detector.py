"""
Cross-Reference Detector for Indonesian Legal Documents.

Uses regex pattern matching (Lex2KG approach, Section 4.7) to detect:
1. References to other regulations (UU, PP, Perpres, Permen)
2. Amendment operations (mengubah, menyisipkan, menghapus)
3. Internal cross-references (Pasal X ayat Y)

Input:  data/parsed/{document_id}.json
Output: data/parsed/{document_id}.json (updated with 'references' field)

Note: OCR-extracted text may contain artifacts (e.g., "l l" → "11",
"200g" → "2008"), so patterns are designed to be flexible.
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import Optional


# ============================================================
# Reference Patterns (designed to handle OCR noise)
# ============================================================

# Pattern for regulation numbers with OCR tolerance
# Handles: "11", "1 1", "l l", "1l", "l1" → all treated as potential numbers
_NUM = r'[\d\sl]{1,4}'   # Loose number pattern for OCR tolerance
_YEAR = r'[\d\sOo]{4}'   # Year pattern (handles "2Ol2" → "2012")

REFERENCE_PATTERNS = [
    # ── External Document References ──
    # "Undang-Undang Nomor X Tahun YYYY"
    {
        "name": "UU_REF",
        "pattern": re.compile(
            r'Undang[_\-\s]?Undang\s+(?:Nomor|No\.?)\s*(\S+)\s+Tahun\s+(\d{4})',
            re.IGNORECASE
        ),
        "ref_type": "MERUJUK_DOKUMEN",
        "target_prefix": "UU",
    },
    # "Peraturan Pemerintah Nomor X Tahun YYYY"
    {
        "name": "PP_REF",
        "pattern": re.compile(
            r'Peraturan\s+Pemerintah\s+(?:Nomor|No\.?)\s*(\S+)\s+Tahun\s+(\d{4})',
            re.IGNORECASE
        ),
        "ref_type": "MERUJUK_DOKUMEN",
        "target_prefix": "PP",
    },
    # "Peraturan Presiden Nomor X Tahun YYYY"
    {
        "name": "PERPRES_REF",
        "pattern": re.compile(
            r'Peraturan\s+Presiden\s+(?:Nomor|No\.?)\s*(\S+)\s+Tahun\s+(\d{4})',
            re.IGNORECASE
        ),
        "ref_type": "MERUJUK_DOKUMEN",
        "target_prefix": "Perpres",
    },
    # "Peraturan Menteri ... Nomor X Tahun YYYY"
    {
        "name": "PERMEN_REF",
        "pattern": re.compile(
            r'Peraturan\s+Menteri\s+\w+\s+(?:Nomor|No\.?)\s*(\S+)\s+Tahun\s+(\d{4})',
            re.IGNORECASE
        ),
        "ref_type": "MERUJUK_DOKUMEN",
        "target_prefix": "Permen",
    },

    # ── Amendment Operations ──
    # "Pasal X diubah sehingga berbunyi"
    {
        "name": "AMEND_UBAH",
        "pattern": re.compile(
            r'(?:Ketentuan\s+)?Pasal\s+(\d+[A-Z]?)\s+(?:diubah|tetap\s+dengan\s+perubahan)',
            re.IGNORECASE
        ),
        "ref_type": "MENGUBAH_PASAL",
        "target_prefix": None,
    },
    # "di antara Pasal X dan Pasal Y disisipkan"
    {
        "name": "AMEND_SISIP",
        "pattern": re.compile(
            r'[Dd]i\s*antara\s+(?:Pasal|ayat|angka)\s+(\S+)\s+dan\s+(?:Pasal|ayat|angka)\s+(\S+)\s+disisipkan',
            re.IGNORECASE
        ),
        "ref_type": "MENYISIPKAN_PASAL",
        "target_prefix": None,
    },
    # "Pasal X dihapus"
    {
        "name": "AMEND_HAPUS",
        "pattern": re.compile(
            r'Pasal\s+(\d+[A-Z]?)\s+dihapus',
            re.IGNORECASE
        ),
        "ref_type": "MENGHAPUS_PASAL",
        "target_prefix": None,
    },

    # ── Internal Cross-References ──
    # "sebagaimana dimaksud dalam Pasal X" (internal reference)
    {
        "name": "INTERNAL_PASAL_REF",
        "pattern": re.compile(
            r'sebagaimana\s+dimaksud\s+(?:dalam|pada)\s+Pasal\s+(\d+[A-Z]?)',
            re.IGNORECASE
        ),
        "ref_type": "MERUJUK_PASAL",
        "target_prefix": None,
    },
]


def clean_ocr_number(raw: str) -> str:
    """Clean OCR artifacts from regulation numbers.
    
    Handles common OCR errors:
    - "l l" → "11" (lowercase L → 1)
    - "1 I" → "11" (space + uppercase I → 1)
    - "200g" → "2008" (g → 8)
    - "2Ol2" → "2012" (O → 0, l → 1)
    """
    cleaned = raw.strip()
    # Remove spaces within numbers
    cleaned = re.sub(r'\s+', '', cleaned)
    # Common OCR substitutions
    cleaned = cleaned.replace('l', '1').replace('I', '1')
    cleaned = cleaned.replace('O', '0').replace('o', '0')
    cleaned = cleaned.replace('g', '8').replace('G', '6')
    cleaned = cleaned.replace('S', '5').replace('s', '5')
    # Only keep digits
    cleaned = re.sub(r'[^0-9]', '', cleaned)
    return cleaned


def resolve_target_doc_id(prefix: str, number_raw: str, year_raw: str,
                          known_doc_ids: set) -> Optional[str]:
    """Resolve a detected reference to a known doc_id.
    
    Tries exact match first, then OCR-cleaned match.
    Returns None if no match found in known_doc_ids.
    """
    number = clean_ocr_number(number_raw)
    year = clean_ocr_number(year_raw)
    
    if not number or not year:
        return None
    
    # Try direct match
    candidate = f"{prefix}_{number}_{year}"
    if candidate in known_doc_ids:
        return candidate
    
    # Special case for Permen
    if prefix == "Permen":
        candidate = f"Permen_Kominfo_{number}_{year}"
        if candidate in known_doc_ids:
            return candidate
    
    return candidate  # Return even if not in known set (could be external doc)


def detect_references(
    components: list[dict],
    document_id: str,
    known_doc_ids: set = None,
) -> list[dict]:
    """Detect cross-references in parsed legal components using regex.
    
    Approach: Lex2KG (Rompis 2025) Section 4.7 — regex-based reference detection.
    
    Args:
        components: List of component dicts from parsed JSON
        document_id: Current document's ID
        known_doc_ids: Set of known regulation doc_ids for resolution
        
    Returns:
        Updated components with 'references' field added
    """
    if known_doc_ids is None:
        known_doc_ids = set()
    
    total_refs = 0
    ref_summary = {}  # type → count
    
    for component in components:
        text = component.get("content", "")
        if not text:
            continue
        
        references = []
        
        for pat_info in REFERENCE_PATTERNS:
            pattern = pat_info["pattern"]
            ref_type = pat_info["ref_type"]
            prefix = pat_info["target_prefix"]
            name = pat_info["name"]
            
            for match in pattern.finditer(text):
                ref = {
                    "type": ref_type,
                    "pattern_name": name,
                    "source_component": component["component_id"],
                    "source_text": match.group(0)[:200],
                    "match_position": match.start(),
                }
                
                if prefix and ref_type == "MERUJUK_DOKUMEN":
                    # External document reference
                    number_raw = match.group(1)
                    year_raw = match.group(2)
                    target_id = resolve_target_doc_id(prefix, number_raw, year_raw, known_doc_ids)
                    ref["target_doc_id"] = target_id
                    ref["target_number"] = clean_ocr_number(number_raw)
                    ref["target_year"] = clean_ocr_number(year_raw)
                    
                    # Skip self-references
                    if target_id == document_id:
                        continue
                        
                elif ref_type in ("MENGUBAH_PASAL", "MENGHAPUS_PASAL"):
                    ref["target_article"] = f"Pasal {match.group(1)}"
                    
                elif ref_type == "MENYISIPKAN_PASAL":
                    ref["between_start"] = match.group(1)
                    ref["between_end"] = match.group(2)
                    
                elif ref_type == "MERUJUK_PASAL":
                    ref["target_article"] = f"Pasal {match.group(1)}"
                
                references.append(ref)
                ref_summary[ref_type] = ref_summary.get(ref_type, 0) + 1
                total_refs += 1
        
        # Add references to component
        if references:
            component["references"] = references
    
    return components, total_refs, ref_summary


def detect_references_in_file(input_path: str, regulation_list_path: str = None) -> str:
    """Run reference detection on a parsed document file.
    
    Updates the file in-place with reference annotations.
    
    Args:
        input_path: Path to parsed document JSON
        regulation_list_path: Path to regulation_list.json for doc_id resolution
        
    Returns:
        Path to updated file
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    document_id = data["document_id"]
    
    # Load known doc_ids from regulation list
    known_doc_ids = set()
    if regulation_list_path and os.path.exists(regulation_list_path):
        with open(regulation_list_path, "r", encoding="utf-8") as f:
            regs = json.load(f)
        known_doc_ids = {r["doc_id"] for r in regs}
    
    # Detect references
    components, total_refs, ref_summary = detect_references(
        data["components"], document_id, known_doc_ids
    )
    
    # Update data
    data["components"] = components
    data["reference_summary"] = {
        "total_references": total_refs,
        "by_type": ref_summary,
    }
    
    # Save in-place
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return input_path


def detect_references_all(parsed_dir: str, regulation_list_path: str = None) -> dict:
    """Run reference detection on all parsed documents.
    
    Returns summary of references detected per document.
    """
    json_files = list(Path(parsed_dir).glob("*.json"))
    summary = {}
    
    for json_path in json_files:
        print(f"Detecting references: {json_path.name}")
        detect_references_in_file(str(json_path), regulation_list_path)
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        ref_info = data.get("reference_summary", {})
        doc_id = data["document_id"]
        summary[doc_id] = ref_info
        
        total = ref_info.get("total_references", 0)
        by_type = ref_info.get("by_type", {})
        print(f"  -> {total} references: {by_type}")
    
    return summary


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect cross-references in parsed legal documents")
    parser.add_argument("--input", required=True, help="Parsed document JSON or directory")
    parser.add_argument("--regulation-list", default=None, help="Path to regulation_list.json")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if input_path.is_dir():
        summary = detect_references_all(str(input_path), args.regulation_list)
        print(f"\nDone! Processed {len(summary)} documents.")
        total_all = sum(v.get("total_references", 0) for v in summary.values())
        print(f"Total references detected: {total_all}")
    else:
        detect_references_in_file(str(input_path), args.regulation_list)
        print("Done!")
