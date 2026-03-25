"""
Text Chunker for Legal Documents.

Splits parsed legal components into overlapping chunks suitable for LLM
processing. Follows Rompis (2025): 400-800 tokens per chunk, 100 token overlap.

Input:  data/parsed/{document_id}.json
Output: data/chunks/{document_id}_chunks.json
"""

import json
import os
import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class Chunk:
    """A text chunk ready for LLM processing."""
    chunk_id: str               # e.g. "UU_11_2008__chunk_001"
    document_id: str
    text: str
    token_count: int
    page_range: list = field(default_factory=list)    # pages covered
    parent_component_id: str = ""                      # nearest legal component
    parent_component_type: str = ""                    # BAB, PASAL, etc.
    chunk_index: int = 0                               # ordinal in document


def _count_tokens(text: str, encoder=None) -> int:
    """Count tokens using tiktoken or fallback to word-based estimate."""
    if encoder is not None:
        return len(encoder.encode(text))
    # Fallback: rough estimate (~1.3 tokens per word for Indonesian)
    return int(len(text.split()) * 1.3)


def _get_encoder(encoding_name: str = "cl100k_base"):
    """Try to load tiktoken encoder, return None if unavailable."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except ImportError:
        print("[WARN] tiktoken not installed. Using word-based token estimation.")
        return None


def create_chunks(
    components: list[dict],
    document_id: str,
    min_tokens: int = 400,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split legal components into overlapping chunks.
    
    Rules (from Rompis 2025):
    - Chunk size: 400-800 tokens
    - Overlap: 100 tokens between consecutive chunks
    - Chunks may span pages within one document
    - Chunks must NOT span different documents
    - Each chunk tracks its nearest parent legal component
    
    Strategy:
    1. Collect text from leaf components (PASAL, AYAT, HURUF level)
    2. If a component's content fits within max_tokens, make it one chunk
    3. If too large, split with overlap
    4. If too small, merge with next component(s)
    
    Args:
        components: List of component dicts from parsed JSON
        document_id: Document ID
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk  
        overlap_tokens: Number of overlapping tokens between chunks
        encoding_name: tiktoken encoding to use
    
    Returns:
        List of Chunk objects
    """
    encoder = _get_encoder(encoding_name)
    chunks = []
    chunk_index = 0
    
    # Collect text segments with metadata
    segments = _collect_segments(components)
    
    if not segments:
        return chunks
    
    # Build chunks using sliding window with overlap
    buffer_text = ""
    buffer_pages = set()
    buffer_component_id = ""
    buffer_component_type = ""
    
    for seg in segments:
        candidate = (buffer_text + "\n\n" + seg["text"]).strip() if buffer_text else seg["text"]
        candidate_tokens = _count_tokens(candidate, encoder)
        
        if candidate_tokens <= max_tokens:
            # Still fits — accumulate
            buffer_text = candidate
            buffer_pages.update(seg.get("pages", []))
            if not buffer_component_id:
                buffer_component_id = seg.get("component_id", "")
                buffer_component_type = seg.get("component_type", "")
        else:
            # Would exceed max — flush current buffer as chunk
            if buffer_text:
                buffer_tokens = _count_tokens(buffer_text, encoder)
                if buffer_tokens >= min_tokens:
                    chunks.append(Chunk(
                        chunk_id=f"{document_id}__chunk_{chunk_index:03d}",
                        document_id=document_id,
                        text=buffer_text.strip(),
                        token_count=buffer_tokens,
                        page_range=sorted(buffer_pages),
                        parent_component_id=buffer_component_id,
                        parent_component_type=buffer_component_type,
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
                    
                    # Create overlap: take last overlap_tokens worth of text
                    overlap_text = _get_tail_tokens(buffer_text, overlap_tokens, encoder)
                    buffer_text = overlap_text + "\n\n" + seg["text"]
                else:
                    # Too small to flush — keep accumulating  
                    buffer_text = candidate
                    
                buffer_pages = set(seg.get("pages", []))
                buffer_component_id = seg.get("component_id", "")
                buffer_component_type = seg.get("component_type", "")
            else:
                # Empty buffer, but segment alone is too large — split it
                sub_chunks = _split_large_text(
                    seg["text"], max_tokens, overlap_tokens, encoder
                )
                for sub_text in sub_chunks:
                    chunks.append(Chunk(
                        chunk_id=f"{document_id}__chunk_{chunk_index:03d}",
                        document_id=document_id,
                        text=sub_text.strip(),
                        token_count=_count_tokens(sub_text, encoder),
                        page_range=sorted(seg.get("pages", [])),
                        parent_component_id=seg.get("component_id", ""),
                        parent_component_type=seg.get("component_type", ""),
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
    
    # Flush remaining buffer
    if buffer_text.strip():
        buffer_tokens = _count_tokens(buffer_text, encoder)
        chunks.append(Chunk(
            chunk_id=f"{document_id}__chunk_{chunk_index:03d}",
            document_id=document_id,
            text=buffer_text.strip(),
            token_count=buffer_tokens,
            page_range=sorted(buffer_pages),
            parent_component_id=buffer_component_id,
            parent_component_type=buffer_component_type,
            chunk_index=chunk_index,
        ))
    
    return chunks


def _collect_segments(components: list[dict]) -> list[dict]:
    """Collect text segments from leaf-level components.
    
    Priority order: PASAL content first, then AYAT, HURUF, etc.
    Components with children delegate content to children.
    """
    segments = []
    
    # Build children set for quick lookup
    has_children = set()
    for c in components:
        if c.get("children"):
            has_children.add(c["component_id"])
    
    for c in components:
        content = c.get("content", "").strip()
        if not content:
            continue
        
        # Build context header for this component
        header = _build_component_header(c)
        full_text = f"{header}\n{content}" if header else content
        
        segments.append({
            "text": full_text,
            "component_id": c["component_id"],
            "component_type": c["component_type"],
            "pages": c.get("page_range", []),
        })
    
    return segments


def _build_component_header(component: dict) -> str:
    """Build a header string for context (e.g. 'BAB I - Ketentuan Umum')."""
    comp_type = component.get("component_type", "")
    number = component.get("number", "")
    title = component.get("title", "")
    
    if comp_type in ("BAB", "BAGIAN", "PARAGRAF"):
        header = f"{comp_type} {number}"
        if title:
            header += f" - {title}"
        return header
    elif comp_type == "PASAL":
        return f"Pasal {number}"
    elif comp_type == "AYAT":
        return f"Ayat ({number})"
    
    return ""


def _get_tail_tokens(text: str, num_tokens: int, encoder=None) -> str:
    """Get the last ~num_tokens of text for overlap."""
    if encoder:
        tokens = encoder.encode(text)
        if len(tokens) <= num_tokens:
            return text
        tail_tokens = tokens[-num_tokens:]
        return encoder.decode(tail_tokens)
    else:
        words = text.split()
        est_words = int(num_tokens / 1.3)
        return " ".join(words[-est_words:])


def _split_large_text(text: str, max_tokens: int, overlap_tokens: int, encoder=None) -> list[str]:
    """Split a large text into chunks with overlap."""
    if encoder:
        tokens = encoder.encode(text)
        step = max_tokens - overlap_tokens
        sub_chunks = []
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + max_tokens]
            sub_chunks.append(encoder.decode(chunk_tokens))
        return sub_chunks
    else:
        # Word-based fallback
        words = text.split()
        est_max_words = int(max_tokens / 1.3)
        est_overlap_words = int(overlap_tokens / 1.3)
        step = est_max_words - est_overlap_words
        sub_chunks = []
        for i in range(0, len(words), step):
            sub_chunks.append(" ".join(words[i:i + est_max_words]))
        return sub_chunks


def save_chunks(document_id: str, chunks: list[Chunk], output_dir: str) -> str:
    """Save chunks to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{document_id}_chunks.json")
    
    data = {
        "document_id": document_id,
        "total_chunks": len(chunks),
        "chunks": [asdict(c) for c in chunks],
    }
    
    # Add stats
    if chunks:
        token_counts = [c.token_count for c in chunks]
        data["stats"] = {
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
        }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return output_path


def chunk_all_documents(input_dir: str, output_dir: str, min_tokens: int = 400,
                        max_tokens: int = 800, overlap_tokens: int = 100) -> list[str]:
    """Chunk all parsed documents in a directory."""
    json_files = list(Path(input_dir).glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return []
    
    output_paths = []
    for json_path in json_files:
        print(f"Chunking: {json_path.name}")
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        
        chunks = create_chunks(
            doc["components"], doc["document_id"],
            min_tokens=min_tokens, max_tokens=max_tokens, overlap_tokens=overlap_tokens,
        )
        out = save_chunks(doc["document_id"], chunks, output_dir)
        
        # Print stats
        if chunks:
            tokens = [c.token_count for c in chunks]
            print(f"  → {out}")
            print(f"  → {len(chunks)} chunks (tokens: min={min(tokens)}, max={max(tokens)}, avg={sum(tokens)//len(tokens)})")
        output_paths.append(out)
    
    return output_paths


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk legal documents for LLM processing")
    parser.add_argument("--input", required=True, help="Input directory with parsed JSON files")
    parser.add_argument("--output", required=True, help="Output directory for chunk JSON files")
    parser.add_argument("--min-tokens", type=int, default=400, help="Minimum tokens per chunk")
    parser.add_argument("--max-tokens", type=int, default=800, help="Maximum tokens per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap tokens between chunks")
    args = parser.parse_args()
    
    paths = chunk_all_documents(args.input, args.output, args.min_tokens, args.max_tokens, args.overlap)
    print(f"\nDone! Chunked {len(paths)} documents.")
