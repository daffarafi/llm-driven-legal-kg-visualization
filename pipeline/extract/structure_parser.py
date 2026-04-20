"""
Document Structure Parser for Indonesian Legal Documents.

Parses extracted text into hierarchical legal components using regex-based
detection (adapted from Lex2KG approach). Detects: BAB, BAGIAN, PARAGRAF,
PASAL, AYAT, HURUF, ANGKA.

Input:  data/extracted/{document_id}.json
Output: data/parsed/{document_id}.json
"""

import json
import re
import os
import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class LegalComponent:
    """A structural component of a legal document."""
    component_id: str           # e.g. "UU_11_2008__BAB_I"
    component_type: str         # BAB | BAGIAN | PARAGRAF | PASAL | AYAT | HURUF | ANGKA
    number: str                 # e.g. "I", "Kesatu", "6", "(1)"
    title: Optional[str]        # e.g. "Ketentuan Umum" (only BAB/BAGIAN have titles)
    content: str                # full text content of this component
    page_range: list = field(default_factory=list)   # [start_page, end_page]
    parent_id: Optional[str] = None
    children: list = field(default_factory=list)     # list of child component IDs
    is_penjelasan: bool = False                      # True if part of "Penjelasan" section


# ============================================================
# Regex patterns for detecting legal components (from Lex2KG)
# ============================================================

# Hierarchy levels (higher number = deeper nesting)
HIERARCHY = {
    "BAB": 1,
    "BAGIAN": 2,
    "PARAGRAF": 3,
    "PASAL": 4,
    "AYAT": 5,
    "HURUF": 6,
    "ANGKA": 7,
}

# Regex patterns - applied line by line
PATTERNS = {
    "BAB": re.compile(
        r"^\s*BAB\s+([IVXLCDM]+)\s*$",
        re.IGNORECASE
    ),
    "BAGIAN": re.compile(
        r"^\s*Bagian\s+(Ke(?:satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|dua\s+belas|tiga\s+belas|empat\s+belas|lima\s+belas))\s*$",
        re.IGNORECASE
    ),
    "PARAGRAF": re.compile(
        r"^\s*Paragraf\s+(\d+)\s*$",
        re.IGNORECASE
    ),
    "PASAL": re.compile(
        r"^\s*Pasal\s+(\d+[A-Z]?)\s*$",
        re.IGNORECASE
    ),
    "AYAT": re.compile(
        r"^\s*\((\d+)\)\s+(.+)",
    ),
    "HURUF": re.compile(
        r"^\s*([a-z])\.\s+(.+)",
    ),
    "ANGKA": re.compile(
        r"^\s*(\d+)\.\s+(.+)",
    ),
}

# Title line that follows BAB/BAGIAN header
TITLE_LINE_RE = re.compile(r"^[A-Z\s]+$")

# Penjelasan section separator — appears in Indonesian legal documents
# between the main body (Batang Tubuh) and the explanation section.
# Matches patterns like:
#   "PENJELASAN"  (standalone line)
#   "PENJEI,ASAN"  (common OCR error: L→I, E→,)
#   "PENJELASAN ATAS" (on same line)
PENJELASAN_RE = re.compile(
    r"^\s*PENJ[EI.,]*L?[AE.,]*S[AE.,]*N\s*$",
    re.IGNORECASE
)


def _make_component_id(document_id: str, comp_type: str, number: str, parent_id: Optional[str] = None) -> str:
    """Generate a unique component ID."""
    safe_number = re.sub(r"[^a-zA-Z0-9]", "", str(number))
    base = f"{document_id}__{comp_type}_{safe_number}"
    return base


def merge_pages_to_text(pages: list[dict]) -> tuple[str, dict]:
    """Merge all pages into a single text with page tracking.
    
    Returns:
        text: Combined text from all pages
        line_to_page: Dict mapping line number to page number
    """
    lines = []
    line_to_page = {}
    
    for page in pages:
        text = page.get("clean_text", "") or page.get("selectable_text", "")
        page_lines = text.split("\n")
        for line in page_lines:
            line_num = len(lines)
            line_to_page[line_num] = page["page_number"]
            lines.append(line)
    
    return "\n".join(lines), line_to_page


def parse_document_structure(extracted_doc: dict) -> list[LegalComponent]:
    """Parse extracted document text into hierarchical legal components.
    
    Algorithm (adapted from Lex2KG):
    1. Merge all pages into single text with page tracking
    2. Iterate line by line, match regex patterns
    3. When match found, close previous component, open new one
    4. Track parent-child via stack (BAB > BAGIAN > PASAL > AYAT > HURUF)
    
    Args:
        extracted_doc: Dict from ExtractedDocument JSON
        
    Returns:
        List of LegalComponent objects
    """
    document_id = extracted_doc["document_id"]
    pages = extracted_doc["pages"]
    
    full_text, line_to_page = merge_pages_to_text(pages)
    lines = full_text.split("\n")
    
    # Detect Penjelasan section start line
    penjelasan_start_line = None
    for idx, line in enumerate(lines):
        if PENJELASAN_RE.match(line.strip()):
            # Verify it's the real Penjelasan separator by checking
            # the next few lines for "ATAS" or "UNDANG-UNDANG"
            lookahead = " ".join(
                lines[idx+1:idx+4]
            ).upper() if idx + 1 < len(lines) else ""
            if "ATAS" in lookahead or "UNDANG" in lookahead or "PERATURAN" in lookahead:
                penjelasan_start_line = idx
                break
    
    if penjelasan_start_line is not None:
        print(f"[INFO] Penjelasan detected at line {penjelasan_start_line} "
              f"(page {line_to_page.get(penjelasan_start_line, '?')})")
    
    components = []
    component_map = {}  # id -> LegalComponent
    
    # Stack tracks current hierarchy: [(component_type, component_id, hierarchy_level)]
    stack = []
    
    # Current component being built
    current_content_lines = []
    current_component = None
    current_start_page = 1
    in_penjelasan = False  # Flag: are we past the Penjelasan separator?
    
    def close_current_component():
        """Save accumulated content to the current component."""
        nonlocal current_component, current_content_lines
        if current_component is not None:
            current_component.content = "\n".join(current_content_lines).strip()
            current_content_lines = []
    
    def find_parent(comp_type: str) -> Optional[str]:
        """Find the appropriate parent for a new component type."""
        level = HIERARCHY.get(comp_type, 99)
        # Walk stack backwards to find first component with lower hierarchy level
        for i in range(len(stack) - 1, -1, -1):
            s_type, s_id, s_level = stack[i]
            if s_level < level:
                return s_id
        return None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        page_num = line_to_page.get(i, 1)
        
        # Check if we've entered the Penjelasan section
        if penjelasan_start_line is not None and i >= penjelasan_start_line:
            in_penjelasan = True
        
        matched = False
        
        # Check each pattern (in hierarchy order: BAB first, then BAGIAN, etc.)
        for comp_type in ["BAB", "BAGIAN", "PARAGRAF", "PASAL", "AYAT", "HURUF", "ANGKA"]:
            pattern = PATTERNS[comp_type]
            match = pattern.match(stripped)
            
            if match:
                # Close previous component
                close_current_component()
                
                number = match.group(1)
                level = HIERARCHY[comp_type]
                
                # For AYAT/HURUF/ANGKA, the match might include inline content
                inline_content = ""
                if comp_type in ("AYAT", "HURUF", "ANGKA") and match.lastindex >= 2:
                    inline_content = match.group(2)
                
                # Pop stack entries that are at same or deeper level
                while stack and stack[-1][2] >= level:
                    stack.pop()
                
                # Find parent
                parent_id = find_parent(comp_type)
                
                # Create new component
                comp_id = _make_component_id(document_id, comp_type, number)
                
                # Handle duplicate IDs (e.g. multiple HURUF "a" in different Ayats)
                if comp_id in component_map:
                    # Make unique by appending parent context
                    if parent_id:
                        parent_suffix = parent_id.split("__")[-1]
                        comp_id = f"{comp_id}__{parent_suffix}"
                
                # Check for title on next line(s) for BAB/BAGIAN
                title = None
                if comp_type in ("BAB", "BAGIAN"):
                    # Look ahead for title line(s)
                    title_lines = []
                    j = i + 1
                    while j < len(lines) and j < i + 3:
                        next_stripped = lines[j].strip()
                        if next_stripped and not any(p.match(next_stripped) for p in PATTERNS.values()):
                            title_lines.append(next_stripped)
                            j += 1
                        else:
                            break
                    if title_lines:
                        title = " ".join(title_lines)
                        i = j - 1  # Skip title lines
                
                component = LegalComponent(
                    component_id=comp_id,
                    component_type=comp_type,
                    number=number,
                    title=title,
                    content="",
                    page_range=[page_num, page_num],
                    parent_id=parent_id,
                    children=[],
                    is_penjelasan=in_penjelasan,
                )
                
                # Register as child of parent
                if parent_id and parent_id in component_map:
                    component_map[parent_id].children.append(comp_id)
                
                components.append(component)
                component_map[comp_id] = component
                current_component = component
                current_start_page = page_num
                current_content_lines = []
                
                if inline_content:
                    current_content_lines.append(inline_content)
                
                # Push to stack
                stack.append((comp_type, comp_id, level))
                
                matched = True
                break
        
        if not matched and stripped:
            # Regular content line - append to current component
            current_content_lines.append(stripped)
            if current_component:
                current_component.page_range[1] = page_num
        
        i += 1
    
    # Close last component
    close_current_component()
    
    return components


def save_parsed_document(document_id: str, components: list[LegalComponent], output_dir: str) -> str:
    """Save parsed components to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{document_id}.json")
    
    data = {
        "document_id": document_id,
        "total_components": len(components),
        "component_types": {},
        "components": [asdict(c) for c in components],
    }
    
    # Count by type
    for c in components:
        data["component_types"][c.component_type] = data["component_types"].get(c.component_type, 0) + 1
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return output_path


def parse_all_documents(input_dir: str, output_dir: str) -> list[str]:
    """Parse all extracted documents in a directory."""
    json_files = list(Path(input_dir).glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return []
    
    output_paths = []
    for json_path in json_files:
        print(f"Parsing: {json_path.name}")
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        
        components = parse_document_structure(doc)
        out = save_parsed_document(doc["document_id"], components, output_dir)
        
        # Print summary
        type_counts = {}
        for c in components:
            type_counts[c.component_type] = type_counts.get(c.component_type, 0) + 1
        print(f"  → {out}")
        print(f"  → {len(components)} components: {type_counts}")
        output_paths.append(out)
    
    return output_paths


def print_component_tree(components: list[LegalComponent], max_depth: int = 3):
    """Print a visual tree of the document hierarchy."""
    # Build lookup
    by_id = {c.component_id: c for c in components}
    roots = [c for c in components if c.parent_id is None]
    
    def _print(comp, indent=0, depth=0):
        if depth > max_depth:
            return
        prefix = "  " * indent + ("├── " if indent > 0 else "")
        label = f"[{comp.component_type}] {comp.number}"
        if comp.title:
            label += f" - {comp.title}"
        content_preview = comp.content[:60].replace("\n", " ") if comp.content else ""
        if content_preview:
            label += f" | {content_preview}..."
        print(f"{prefix}{label}")
        
        for child_id in comp.children:
            if child_id in by_id:
                _print(by_id[child_id], indent + 1, depth + 1)
    
    for root in roots:
        _print(root)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse legal document structure")
    parser.add_argument("--input", required=True, help="Input directory with extracted JSON files")
    parser.add_argument("--output", required=True, help="Output directory for parsed JSON files")
    args = parser.parse_args()
    
    paths = parse_all_documents(args.input, args.output)
    print(f"\nDone! Parsed {len(paths)} documents.")
