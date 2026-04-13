"""
LLM-based Knowledge Graph Extractor for Indonesian Legal Documents.

Uses Google Gemini to extract nodes (entities) and edges (relations) from
legal text chunks, following the ontology defined in the plan.

Input:  data/chunks/{document_id}_chunks.json
Output: data/triples/{document_id}_triples.json
"""

import json
import os
import re
import time
import hashlib
import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from tqdm import tqdm


@dataclass
class ExtractedNode:
    """A node/entity extracted from legal text."""
    id: str
    type: str           # UndangUndang | Pasal | EntitasHukum | PerbuatanHukum | Sanksi | KonsepHukum
    label: str          # e.g. "Pasal 27", "Pencemaran Nama Baik"
    content: str        # description or original text
    provenance: dict = field(default_factory=dict)


@dataclass
class ExtractedEdge:
    """A relationship between two nodes."""
    source_id: str
    target_id: str
    type: str           # MENGATUR | MENETAPKAN_SANKSI | MERUJUK | MEMUAT | BERLAKU_UNTUK
    provenance: dict = field(default_factory=dict)


# ============================================================
# System prompt with ontology definition
# ============================================================

SYSTEM_PROMPT = """You are a Knowledge Graph extractor for Indonesian legal documents.
Given a chunk of legal text, extract entities (nodes) and relationships (edges) according to the ontology below.

## Valid Node Types

| Type | Description | Example Labels |
|------|-------------|----------------|
| UndangUndang | The regulation being processed. Create exactly ONE per document. | "Undang-Undang tentang Informasi dan Transaksi Elektronik" |
| Bab | A chapter (Bab) within the regulation. | "BAB VII PERBUATAN YANG DILARANG" |
| Pasal | An article (Pasal) within the regulation. | "Pasal 27", "Pasal 45" |
| Ayat | A sub-article (Ayat) within a Pasal. Only create if the text explicitly refers to a specific ayat. | "Pasal 27 ayat (1)", "Pasal 45 ayat (3)" |
| EntitasHukum | A legal subject or object — a person, institution, or legal body explicitly named as a party in a provision. | "Setiap Orang", "Penyelenggara Sistem Elektronik", "Pemerintah" |
| PerbuatanHukum | A specific legal act that is regulated, prohibited, or required. Must be a concrete action, not an abstract concept. | "mendistribusikan informasi yang memiliki muatan penghinaan", "mengakses Komputer dan/atau Sistem Elektronik milik Orang lain" |
| Sanksi | A penalty or sanction stated in the text. Always include the FULL penalty detail in the label. | "pidana penjara paling lama 6 tahun dan/atau denda paling banyak Rp1.000.000.000,00" |
| KonsepHukum | A legal concept or term that is **formally defined** in the text (e.g., in Pasal 1 definitions section). Do NOT create KonsepHukum for general terms that are merely mentioned. | "Informasi Elektronik", "Dokumen Elektronik", "Tanda Tangan Elektronik" |

## Valid Relation Types

| Type | Direction | Description |
|------|-----------|-------------|
| MEMUAT | UndangUndang → Bab, Bab → Pasal | Hierarchical containment |
| MEMILIKI_AYAT | Pasal → Ayat | Pasal contains Ayat |
| MENGATUR | Pasal/Ayat → PerbuatanHukum | An article regulates an act |
| MENETAPKAN_SANKSI | Pasal/Ayat → Sanksi | An article establishes a sanction |
| BERLAKU_UNTUK | Pasal/Ayat → EntitasHukum | A provision applies to a legal entity |
| MERUJUK | Pasal → Pasal | Explicit cross-reference to another article (only when the text says "sebagaimana dimaksud dalam Pasal X") |
| MENDEFINISIKAN | Pasal → KonsepHukum | An article formally defines a concept (typically in Pasal 1) |

## Critical Rules

### Deduplication
1. Every node MUST have a unique `id`, `type`, and `label`.
2. Use the format `{Type}_{short_label}` for ids (e.g., `Pasal_27`, `Sanksi_pidana_penjara_6_tahun`).
3. If the same entity appears multiple times in the text, reuse the SAME id — do NOT create duplicates.
4. Create only ONE `UndangUndang` node for the document being processed. Other laws referenced in the text (e.g., UUD 1945, UU Telekomunikasi) should NOT get their own UndangUndang node; instead, mention them in the `content` field of the MERUJUK edge or the referencing Pasal.

### Quality over Quantity
5. Prefer fewer, high-quality nodes over many low-quality ones.
6. Only create `KonsepHukum` for terms that are **explicitly defined** with a definition in the text (e.g., "Yang dimaksud dengan X adalah ..."). General legal terms that are merely mentioned should NOT become KonsepHukum nodes.
7. `PerbuatanHukum` must be a **specific, concrete action** (e.g., "mendistribusikan konten bermuatan penghinaan"), not a vague description.
8. Every `Sanksi` node must contain the FULL penalty text including duration and/or fine amount.

### Hierarchy
9. Maintain strict hierarchy: UndangUndang → Bab → Pasal → Ayat.
10. Every Pasal should be connected to its parent Bab via MEMUAT if the Bab is known from the text.
11. If a Pasal has multiple ayat, create Ayat nodes and connect them via MEMILIKI_AYAT.

### Relationships
12. Each Pasal/Ayat that regulates an action MUST have a MENGATUR edge.
13. Each Pasal/Ayat that specifies a sanction MUST have both MENGATUR (to the prohibited act) and MENETAPKAN_SANKSI (to the penalty).
14. MERUJUK edges should only be created for **explicit cross-references** (e.g., "sebagaimana dimaksud dalam Pasal 27").

## Output Format

Output MUST be valid JSON:
```json
{
  "nodes": [
    {"id": "Pasal_27", "type": "Pasal", "label": "Pasal 27", "content": "brief description or original text excerpt"}
  ],
  "edges": [
    {"source": "Pasal_27", "target": "PerbuatanHukum_distribusi_konten_ilegal", "type": "MENGATUR"}
  ]
}
```

## Example

**Input text:**
"Pasal 45
(1) Setiap Orang yang dengan sengaja dan tanpa hak mendistribusikan dan/atau mentransmisikan dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan yang melanggar kesusilaan sebagaimana dimaksud dalam Pasal 27 ayat (1) dipidana dengan pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)."

**Expected output:**
```json
{
  "nodes": [
    {"id": "Pasal_45", "type": "Pasal", "label": "Pasal 45", "content": "Ketentuan pidana untuk pelanggaran Pasal 27"},
    {"id": "Ayat_45_1", "type": "Ayat", "label": "Pasal 45 ayat (1)", "content": "Sanksi pidana untuk distribusi konten melanggar kesusilaan"},
    {"id": "EntitasHukum_Setiap_Orang", "type": "EntitasHukum", "label": "Setiap Orang", "content": "Subjek hukum umum"},
    {"id": "PerbuatanHukum_distribusi_konten_asusila", "type": "PerbuatanHukum", "label": "mendistribusikan dan/atau mentransmisikan Informasi Elektronik yang memiliki muatan melanggar kesusilaan", "content": ""},
    {"id": "Sanksi_penjara_6_tahun_denda_1M", "type": "Sanksi", "label": "pidana penjara paling lama 6 tahun dan/atau denda paling banyak Rp1.000.000.000,00", "content": ""},
    {"id": "Pasal_27", "type": "Pasal", "label": "Pasal 27", "content": ""},
    {"id": "Ayat_27_1", "type": "Ayat", "label": "Pasal 27 ayat (1)", "content": ""}
  ],
  "edges": [
    {"source": "Pasal_45", "target": "Ayat_45_1", "type": "MEMILIKI_AYAT"},
    {"source": "Ayat_45_1", "target": "PerbuatanHukum_distribusi_konten_asusila", "type": "MENGATUR"},
    {"source": "Ayat_45_1", "target": "Sanksi_penjara_6_tahun_denda_1M", "type": "MENETAPKAN_SANKSI"},
    {"source": "Ayat_45_1", "target": "EntitasHukum_Setiap_Orang", "type": "BERLAKU_UNTUK"},
    {"source": "Ayat_45_1", "target": "Ayat_27_1", "type": "MERUJUK"},
    {"source": "Pasal_27", "target": "Ayat_27_1", "type": "MEMILIKI_AYAT"}
  ]
}
```
"""


USER_PROMPT_TEMPLATE = """Extract all entities and relationships from the following Indonesian legal text.
Document ID: {document_id}

<LEGAL_TEXT>
{chunk_text}
</LEGAL_TEXT>"""


def generate_unique_id(label: str, node_type: str) -> str:
    """Generate a deterministic unique ID for a node."""
    # Sanitize label for ID
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", label.strip()[:50])
    clean = re.sub(r"_+", "_", clean).strip("_")
    return f"{node_type}_{clean}"


def extract_triples_from_chunk(
    chunk_text: str,
    chunk_metadata: dict,
    model: genai.GenerativeModel,
) -> tuple[list[ExtractedNode], list[ExtractedEdge]]:
    """Extract nodes and edges from a single chunk using LLM.
    
    Args:
        chunk_text: The text content of the chunk
        chunk_metadata: Metadata dict with document_id, chunk_id, page_range
        model: Gemini GenerativeModel instance
        
    Returns:
        Tuple of (nodes, edges)
    """
    prompt = USER_PROMPT_TEMPLATE.format(
        document_id=chunk_metadata.get("document_id", ""),
        chunk_text=chunk_text,
    )
    
    response = model.generate_content(
        [SYSTEM_PROMPT, prompt],
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
        }
    )
    
    try:
        raw = json.loads(response.text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        text = response.text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            raw = json.loads(json_match.group())
        else:
            return [], []
    
    nodes = []
    edges = []
    
    provenance = {
        "source_document_id": chunk_metadata.get("document_id", ""),
        "source_chunk_id": chunk_metadata.get("chunk_id", ""),
        "source_pages": chunk_metadata.get("page_range", []),
        "extraction_model": "gemini-2.5-flash",
    }
    
    for n in raw.get("nodes", []):
        node_id = n.get("id", generate_unique_id(n.get("label", ""), n.get("type", "")))
        nodes.append(ExtractedNode(
            id=node_id,
            type=n.get("type", "KonsepHukum"),
            label=n.get("label", ""),
            content=n.get("content", ""),
            provenance=provenance.copy(),
        ))
    
    for e in raw.get("edges", []):
        edges.append(ExtractedEdge(
            source_id=e.get("source", ""),
            target_id=e.get("target", ""),
            type=e.get("type", "MERUJUK"),
            provenance=provenance.copy(),
        ))
    
    return nodes, edges


def extract_triples_from_batch(
    chunks: list[dict],
    model: genai.GenerativeModel,
    batch_size: int = 5,
) -> tuple[list[ExtractedNode], list[ExtractedEdge]]:
    """Extract triples from a batch of chunks, concatenated.
    
    Args:
        chunks: List of chunk dicts from chunks JSON
        model: Gemini GenerativeModel instance
        batch_size: Number of chunks to process per LLM call
        
    Returns:
        Tuple of (all_nodes, all_edges)
    """
    batch_text = "\n\n---CHUNK---\n\n".join([c["text"] for c in chunks[:batch_size]])
    
    metadata = {
        "document_id": chunks[0].get("document_id", ""),
        "chunk_id": ",".join([c.get("chunk_id", "") for c in chunks[:batch_size]]),
        "page_range": list(set(p for c in chunks[:batch_size] for p in c.get("page_range", []))),
    }
    
    return extract_triples_from_chunk(batch_text, metadata, model)


def extract_all_triples(
    chunks_path: str,
    output_dir: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    batch_size: int = 5,
    max_retries: int = 3,
    delay_between_calls: float = 1.0,
) -> str:
    """Extract triples from all chunks in a document.
    
    Args:
        chunks_path: Path to chunks JSON file
        output_dir: Output directory for triples JSON
        api_key: Gemini API key
        model_name: Gemini model to use
        batch_size: Chunks per LLM call
        max_retries: Max retry attempts on failure
        delay_between_calls: Seconds to wait between API calls
        
    Returns:
        Path to output triples JSON file
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    document_id = chunks_data["document_id"]
    chunks = chunks_data["chunks"]
    
    all_nodes = []
    all_edges = []
    errors = []
    
    # Process in batches
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(chunks), batch_size), total=total_batches, desc=f"Extracting {document_id}"):
        batch = chunks[i:i + batch_size]
        
        for attempt in range(max_retries):
            try:
                nodes, edges = extract_triples_from_batch(batch, model, batch_size)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
                break
            except Exception as e:
                error_msg = f"Batch {i//batch_size}: attempt {attempt+1} failed: {str(e)}"
                if attempt == max_retries - 1:
                    errors.append(error_msg)
                    print(f"  [ERROR] {error_msg}")
                else:
                    wait = (attempt + 1) * 2
                    print(f"  [RETRY] {error_msg} — waiting {wait}s")
                    time.sleep(wait)
        
        # Rate limiting
        time.sleep(delay_between_calls)
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{document_id}_triples.json")
    
    output_data = {
        "document_id": document_id,
        "total_nodes": len(all_nodes),
        "total_edges": len(all_edges),
        "errors": errors,
        "nodes": [asdict(n) for n in all_nodes],
        "edges": [asdict(e) for e in all_edges],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_path


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Extract KG triples from chunks using LLM")
    parser.add_argument("--input", required=True, help="Input chunks JSON file")
    parser.add_argument("--output", required=True, help="Output directory for triples")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--batch-size", type=int, default=5, help="Chunks per batch")
    args = parser.parse_args()
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment")
        exit(1)
    
    path = extract_all_triples(args.input, args.output, api_key, args.model, args.batch_size)
    print(f"Done! Triples saved to: {path}")
