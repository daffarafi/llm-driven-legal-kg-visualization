"""
Entity Deduplicator for Knowledge Graph triples.

Merges duplicate entities using label normalization and optional embedding
cosine similarity. Consolidates provenance from merged nodes.

Input:  data/validated/{document_id}_triples.json
Output: data/deduped/{document_id}_triples.json
"""

import json
import os
import re
import argparse
from collections import defaultdict
from pathlib import Path


def normalize_label(label: str) -> str:
    """Normalize entity label for comparison.
    
    - Lowercase
    - Strip whitespace
    - Remove extra spaces
    - Remove common prefixes
    """
    normalized = label.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    # Remove leading article-like words
    normalized = re.sub(r"^(pasal|ayat|bab|bagian)\s+", "", normalized)
    return normalized


def deduplicate_entities(
    nodes: list[dict],
    edges: list[dict],
    similarity_threshold: float = 0.85,
    use_embeddings: bool = False,
    embeddings: dict = None,
) -> tuple[list[dict], list[dict], dict]:
    """Deduplicate nodes by label normalization and optional embedding similarity.
    
    Process:
    1. Normalize all labels: lowercase, strip, canonical form
    2. Group by (normalized_label, type) → merge exact matches
    3. (Optional) For remaining, check cosine similarity of embeddings 
    4. Merge: keep longest label, combine provenance
    5. Update edge references to point to merged IDs
    
    Args:
        nodes: List of node dicts
        edges: List of edge dicts
        similarity_threshold: Cosine similarity threshold for merge
        use_embeddings: Whether to use embedding-based dedup
        embeddings: Dict mapping node_id -> embedding vector
        
    Returns:
        Tuple of (deduped_nodes, updated_edges, merge_map)
    """
    # Step 1: Group by (normalized_label, type)
    groups = defaultdict(list)
    for node in nodes:
        key = (normalize_label(node["label"]), node["type"])
        groups[key].append(node)
    
    # Step 2: Merge each group
    deduped_nodes = []
    merge_map = {}  # old_id -> canonical_id
    
    for key, group_nodes in groups.items():
        if len(group_nodes) == 1:
            deduped_nodes.append(group_nodes[0])
            continue
        
        # Pick canonical: longest label, most content, prefer uppercase (standard legal format)
        canonical = max(group_nodes, key=lambda n: (
            len(n.get("label") or ""),
            n.get("label", "").isupper(),  # prefer uppercase labels (e.g., "BAB VI" over "Bab VI")
            len(n.get("content") or ""),
        ))
        
        # Merge provenance from all duplicates
        all_sources = set()
        all_chunks = set()
        all_pages = set()
        
        for node in group_nodes:
            prov = node.get("provenance", {})
            if prov.get("source_document_id"):
                all_sources.add(prov["source_document_id"])
            chunk_ids = prov.get("source_chunk_id", "")
            if chunk_ids:
                all_chunks.update(chunk_ids.split(","))
            for p in prov.get("source_pages", []):
                all_pages.add(p)
            
            # Map old ID to canonical ID
            if node["id"] != canonical["id"]:
                merge_map[node["id"]] = canonical["id"]
        
        # Update canonical provenance
        canonical["provenance"] = {
            "source_document_id": ",".join(sorted(all_sources)) if all_sources else "",
            "source_chunk_id": ",".join(sorted(all_chunks)) if all_chunks else "",
            "source_pages": sorted(all_pages),
            "extraction_model": canonical.get("provenance", {}).get("extraction_model", ""),
            "merged_from": len(group_nodes),
        }
        
        deduped_nodes.append(canonical)
    
    # Step 3 (optional): Embedding-based similarity dedup
    if use_embeddings and embeddings:
        deduped_nodes, extra_merges = _embedding_dedup(
            deduped_nodes, embeddings, similarity_threshold
        )
        merge_map.update(extra_merges)
    
    # Step 4: Update edge references
    updated_edges = []
    seen_edges = set()
    
    for edge in edges:
        source = edge.get("source_id", "") or edge.get("source", "")
        target = edge.get("target_id", "") or edge.get("target", "")
        
        # Apply merge map
        source = merge_map.get(source, source)
        target = merge_map.get(target, target)
        
        # Skip self-loops created by merging
        if source == target:
            continue
        
        # Skip duplicate edges
        edge_key = (source, target, edge.get("type", ""))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        
        updated_edge = dict(edge)
        updated_edge["source_id"] = source
        updated_edge["target_id"] = target
        updated_edges.append(updated_edge)
    
    return deduped_nodes, updated_edges, merge_map


def _embedding_dedup(nodes, embeddings, threshold):
    """Deduplicate using cosine similarity of embeddings."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        return nodes, {}
    
    # Filter nodes that have embeddings
    indexed_nodes = [(i, n) for i, n in enumerate(nodes) if n["id"] in embeddings]
    if len(indexed_nodes) < 2:
        return nodes, {}
    
    ids = [n["id"] for _, n in indexed_nodes]
    vecs = np.array([embeddings[nid] for nid in ids])
    sim_matrix = cosine_similarity(vecs)
    
    merge_map = {}
    merged_indices = set()
    
    for i in range(len(ids)):
        if i in merged_indices:
            continue
        for j in range(i + 1, len(ids)):
            if j in merged_indices:
                continue
            # Only merge same-type nodes
            if indexed_nodes[i][1]["type"] != indexed_nodes[j][1]["type"]:
                continue
            if sim_matrix[i, j] >= threshold:
                merge_map[ids[j]] = ids[i]
                merged_indices.add(j)
    
    # Remove merged nodes
    remaining = [n for i, (_, n) in enumerate(indexed_nodes) if i not in merged_indices]
    # Add back nodes without embeddings
    for n in nodes:
        if n["id"] not in embeddings:
            remaining.append(n)
    
    return remaining, merge_map


def deduplicate_triples_file(input_path: str, output_dir: str, similarity_threshold: float = 0.85, prompt_id: str = None) -> str:
    """Deduplicate a validated triples file.
    
    Args:
        input_path: Path to validated triples JSON
        output_dir: Output directory
        similarity_threshold: Cosine similarity threshold
        prompt_id: Prompt identifier for output filename
        
    Returns:
        Path to deduplicated output file
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    document_id = data["document_id"]
    
    deduped_nodes, updated_edges, merge_map = deduplicate_entities(
        data["nodes"], data["edges"], similarity_threshold
    )
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{document_id}_{prompt_id}_triples.json" if prompt_id else f"{document_id}_triples.json"
    output_path = os.path.join(output_dir, filename)
    
    output_data = {
        "document_id": document_id,
        "total_nodes": len(deduped_nodes),
        "total_edges": len(updated_edges),
        "nodes_before": data["total_nodes"],
        "edges_before": data["total_edges"],
        "nodes_merged": len(merge_map),
        "nodes": deduped_nodes,
        "edges": updated_edges,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_path


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate KG entities")
    parser.add_argument("--input", required=True, help="Input validated triples JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")
    args = parser.parse_args()
    
    path = deduplicate_triples_file(args.input, args.output, args.threshold)
    print(f"Deduplicated triples saved to: {path}")
