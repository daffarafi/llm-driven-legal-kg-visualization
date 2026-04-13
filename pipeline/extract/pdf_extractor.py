"""
PDF Text Extractor for Indonesian Legal Documents.

Extracts text from PDF files using PyMuPDF (for digital PDFs) with
PaddleOCR fallback (for scanned PDFs). Outputs structured JSON per document.

Input:  data/raw/*.pdf
Output: data/extracted/{document_id}.json
"""

import fitz  # PyMuPDF
import json
import os
import re
import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""
    page_number: int
    selectable_text: str       # text from PyMuPDF (digital)
    ocr_text: Optional[str]    # text from PaddleOCR (scanned), None if digital
    is_scanned: bool
    clean_text: str = ""       # cleaned final text


@dataclass
class ExtractedDocument:
    """Full extracted document with metadata."""
    document_id: str           # e.g. "UU_11_2008"
    uu_number: str             # e.g. "11/2008"
    title: str                 # e.g. "Undang-Undang tentang ITE"
    year: int
    source_url: str
    total_pages: int
    pages: list = field(default_factory=list)  # list of PageContent dicts


# ============================================================
# Header/footer patterns to remove (common in Indonesian legal PDFs)
# ============================================================
HEADER_FOOTER_PATTERNS = [
    r"^-?\s*\d+\s*-?\s*$",                          # standalone page numbers: "- 1 -", "2"
    r"^www\.hukumonline\.com$",                       # website headers
    r"^Salinan$",                                     # "Salinan" header
    r"^PRESIDEN\s+REPUBLIK\s+INDONESIA$",             # president header
    r"^LEMBARAN\s+NEGARA\s+REPUBLIK\s+INDONESIA$",   # gazette header  
    r"^\s*\.{3,}\s*$",                                # dots separator
    r"^TAMBAHAN\s+LEMBARAN\s+NEGARA",                 # supplement gazette
    r"^PENJELASAN\s*(ATAS)?$",                         # explanation header
]

HEADER_FOOTER_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in HEADER_FOOTER_PATTERNS]

# Inline patterns that appear mid-text due to page breaks
INLINE_CLEANUP_PATTERNS = [
    # "Setiap . . . PRESIDEN REPUBLIK INDONESIA (2) Setiap" → remove the header part
    (re.compile(r"\.\s*\.\s*\.?\s*PRESIDEN\s+REPUBLIK\s+INDONESIA\s*", re.IGNORECASE), ""),
    # Standalone dots that indicate page break: ". . ."
    (re.compile(r"(?<!\w)\.\s+\.\s+\.(?!\w)"), ""),
    # Page numbers appearing inline: "... - 12 -"
    (re.compile(r"\s*-\s*\d+\s*-\s*"), " "),
]


def remove_headers_footers(text: str) -> str:
    """Remove common headers and footers from legal PDF text."""
    # First pass: remove inline page-break artifacts
    for pattern, replacement in INLINE_CLEANUP_PATTERNS:
        text = pattern.sub(replacement, text)
    
    # Second pass: remove standalone header/footer lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        is_header_footer = any(pat.match(stripped) for pat in HEADER_FOOTER_RE)
        if not is_header_footer:
            cleaned.append(line)
    return "\n".join(cleaned)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace/newlines into single ones."""
    # Collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def clean_page_text(page: PageContent) -> str:
    """Pick the best text source and clean it."""
    raw = page.ocr_text if page.is_scanned else page.selectable_text
    if not raw:
        return ""
    raw = remove_headers_footers(raw)
    raw = normalize_whitespace(raw)
    return raw


def parse_uu_metadata_from_filename(pdf_path: str) -> dict:
    """Extract regulation metadata from filename.
    
    Expected filename patterns:
      - "UU Nomor 11 Tahun 2008.pdf"
      - "PP Nomor 71 Tahun 2019.pdf"
      - "Perpres Nomor 95 Tahun 2018.pdf"
      - "Permen Kominfo Nomor 5 Tahun 2017.pdf"
      - "UU_11_2008_ITE.pdf"
    """
    basename = Path(pdf_path).stem
    
    # Pattern map: regex → doc_id prefix
    PATTERNS = [
        # "UU Nomor X Tahun YYYY"
        (r"(?:UU|Undang.Undang)\s*(?:Nomor|No\.?)\s*(\d+)\s*(?:Tahun)\s*(\d{4})", "UU"),
        # "PP Nomor X Tahun YYYY"
        (r"PP\s*(?:Nomor|No\.?)\s*(\d+)\s*(?:Tahun)\s*(\d{4})", "PP"),
        # "Perpres Nomor X Tahun YYYY"
        (r"Perpres\s*(?:Nomor|No\.?)\s*(\d+)\s*(?:Tahun)\s*(\d{4})", "Perpres"),
        # "Permen Kominfo Nomor X Tahun YYYY" (or other ministries)
        (r"Permen\s*\w*\s*(?:Nomor|No\.?)\s*(\d+)\s*(?:Tahun)\s*(\d{4})", "Permen_Kominfo"),
        # Underscore/dash format: "UU_11_2008", "PP-71-2019"
        (r"(UU|PP|Perpres)[_\-](\d+)[_\-](\d{4})", None),  # special handling
    ]
    
    for pattern, prefix in PATTERNS:
        match = re.search(pattern, basename, re.IGNORECASE)
        if match:
            if prefix is None:
                # Underscore/dash format with type in group 1
                reg_type = match.group(1)
                nomor = match.group(2)
                tahun = match.group(3)
                doc_id = f"{reg_type}_{nomor}_{tahun}"
            else:
                nomor = match.group(1)
                tahun = match.group(2)
                doc_id = f"{prefix}_{nomor}_{tahun}"
            
            return {
                "document_id": doc_id,
                "uu_number": f"{nomor}/{tahun}",
                "title": basename,
                "year": int(tahun),
                "source_url": "",
            }
    
    # Fallback: use sanitized filename
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", basename)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return {
        "document_id": sanitized,
        "uu_number": "unknown",
        "title": basename,
        "year": 0,
        "source_url": "",
    }


def extract_pdf(pdf_path: str, scanned_threshold: int = 50) -> ExtractedDocument:
    """Extract text from a legal PDF document.
    
    Uses PyMuPDF for digital text extraction. Falls back to PaddleOCR
    if a page has fewer than `scanned_threshold` characters (likely scanned).
    
    Args:
        pdf_path: Path to the PDF file
        scanned_threshold: Min chars to consider a page as digital
        
    Returns:
        ExtractedDocument with all pages extracted and cleaned
    """
    doc = fitz.open(pdf_path)
    
    # Lazy-load PaddleOCR only if needed
    ocr = None
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        selectable = page.get_text("text")
        
        # Heuristic: if text < threshold chars, likely scanned
        is_scanned = len(selectable.strip()) < scanned_threshold
        ocr_text = None
        
        if is_scanned:
            if ocr is None:
                try:
                    from paddleocr import PaddleOCR
                    ocr = PaddleOCR(use_angle_cls=True, lang="id", show_log=False)
                except ImportError:
                    print("  [WARN] PaddleOCR not installed. Skipping OCR for scanned pages.")
                    ocr = "unavailable"
            
            if ocr != "unavailable":
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                result = ocr.ocr(img_bytes)
                if result and result[0]:
                    ocr_text = "\n".join([line[1][0] for line in result[0]])
        
        page_content = PageContent(
            page_number=page_num + 1,
            selectable_text=selectable,
            ocr_text=ocr_text,
            is_scanned=is_scanned,
        )
        page_content.clean_text = clean_page_text(page_content)
        pages.append(page_content)
    
    doc.close()
    
    # Build metadata
    metadata = parse_uu_metadata_from_filename(pdf_path)
    
    # Try to extract title from first page content
    first_page_text = pages[0].clean_text if pages else ""
    title_match = re.search(
        r"UNDANG-UNDANG\s+REPUBLIK\s+INDONESIA\s+.*?TENTANG\s+(.+?)(?:\n|$)",
        first_page_text, re.IGNORECASE | re.DOTALL
    )
    if title_match:
        metadata["title"] = f"Undang-Undang Nomor {metadata['uu_number']} tentang {title_match.group(1).strip()}"
    
    return ExtractedDocument(
        **metadata,
        total_pages=len(pages),
        pages=[asdict(p) for p in pages],
    )


def save_extracted_document(doc: ExtractedDocument, output_dir: str) -> str:
    """Save extracted document to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{doc.document_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(doc), f, ensure_ascii=False, indent=2)
    return output_path


def extract_all_pdfs(input_dir: str, output_dir: str, scanned_threshold: int = 50) -> list[str]:
    """Extract all PDFs in a directory.
    
    Returns list of output file paths.
    """
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return []
    
    output_paths = []
    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")
        doc = extract_pdf(str(pdf_path), scanned_threshold)
        out = save_extracted_document(doc, output_dir)
        print(f"  → {out} ({doc.total_pages} pages)")
        output_paths.append(out)
    
    return output_paths


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from legal PDF documents")
    parser.add_argument("--input", required=True, help="Input directory with PDF files")
    parser.add_argument("--output", required=True, help="Output directory for JSON files")
    parser.add_argument("--scanned-threshold", type=int, default=50, help="Min chars for digital page")
    args = parser.parse_args()
    
    paths = extract_all_pdfs(args.input, args.output, args.scanned_threshold)
    print(f"\nDone! Extracted {len(paths)} documents.")
