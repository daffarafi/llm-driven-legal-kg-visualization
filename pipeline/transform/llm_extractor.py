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

SYSTEM_PROMPT = """Anda adalah ekstraktor Knowledge Graph hukum Indonesia.
Dari teks hukum yang diberikan, ekstrak entitas dan relasi sesuai ontologi berikut:

### Node Types yang Valid:
- UndangUndang: Peraturan perundang-undangan (contoh: "UU No. 11 Tahun 2008")
- Bab: Bab dari UU (contoh: "Bab VII")
- Pasal: Pasal dari UU (contoh: "Pasal 27")
- EntitasHukum: Subjek/objek hukum — orang, badan, institusi (contoh: "Penyelenggara Sistem Elektronik", "Pemerintah")
- PerbuatanHukum: Tindakan yang diatur/dilarang (contoh: "mendistribusikan konten ilegal", "akses ilegal")
- Sanksi: Hukuman yang ditetapkan (contoh: "pidana penjara paling lama 6 tahun", "denda paling banyak Rp1.000.000.000")
- KonsepHukum: Konsep abstrak atau definisi (contoh: "Informasi Elektronik", "Transaksi Elektronik", "Tanda Tangan Elektronik")

### Relation Types yang Valid:
- MENGATUR: Pasal mengatur suatu perbuatan hukum (Pasal → PerbuatanHukum)
- MENETAPKAN_SANKSI: Pasal menetapkan sanksi (Pasal → Sanksi)
- BERLAKU_UNTUK: Ketentuan berlaku untuk entitas tertentu (Pasal → EntitasHukum)
- MERUJUK: Referensi silang ke pasal/UU lain (Pasal → Pasal)
- MEMUAT: Hierarki (UU → Bab, Bab → Pasal)
- MENDEFINISIKAN: Pasal mendefinisikan konsep (Pasal → KonsepHukum)

### Aturan Penting:
1. Setiap node HARUS punya id unik, type, dan label
2. id harus deskriptif: gunakan format "TYPE_label_singkat" (contoh: "Pasal_27", "PerbuatanHukum_akses_ilegal")
3. Untuk Sanksi, sertakan detail lengkap di label (misal "pidana penjara paling lama 12 tahun dan/atau denda paling banyak Rp12.000.000.000")
4. Untuk MERUJUK, hanya jika ada referensi eksplisit ke pasal/UU lain
5. Jangan membuat node duplikat — jika entitas yang sama muncul, gunakan id yang sama

Output HARUS dalam JSON format:
{"nodes": [{"id": "...", "type": "...", "label": "...", "content": "..."}],
 "edges": [{"source": "...", "target": "...", "type": "..."}]}
"""


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
    prompt = f"Ekstrak entitas dan relasi dari teks hukum berikut:\n\n{chunk_text}"
    
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
