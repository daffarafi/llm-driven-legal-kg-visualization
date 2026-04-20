"""
Validator for LLM-extracted Knowledge Graph triples.

Checks that extracted nodes and edges conform to the defined ontology
(valid types, non-empty labels, edge references valid nodes).

Input:  data/triples/{document_id}_triples.json
Output: data/validated/{document_id}_triples.json + error log
"""

import json
import os
import argparse
from dataclasses import asdict
from pathlib import Path


# ============================================================
# Load schema from config (single source of truth)
# ============================================================

def _load_schema():
    """Load KG schema from config file."""
    schema_path = Path(__file__).parent.parent.parent / "config" / "kg_schema.json"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback if config not found
    return None

_schema = _load_schema()

if _schema:
    VALID_NODE_TYPES = set(_schema["node_types"].keys())
    VALID_EDGE_TYPES = set(_schema["edge_types"].keys())
    EDGE_CONSTRAINTS = {
        etype: (set(edef["allowed_sources"]), set(edef["allowed_targets"]))
        for etype, edef in _schema["edge_types"].items()
        if "allowed_sources" in edef and "allowed_targets" in edef
    }
else:
    # Hardcoded fallback
    VALID_NODE_TYPES = {
        "Regulasi", "Bab", "Bagian", "Pasal", "Ayat",
        "EntitasHukum", "PerbuatanHukum", "Sanksi", "KonsepHukum",
    }
    VALID_EDGE_TYPES = {
        "MEMUAT", "MEMILIKI_AYAT", "MENGATUR", "MENETAPKAN_SANKSI", "BERLAKU_UNTUK",
        "MERUJUK", "MENDEFINISIKAN",
    }
    EDGE_CONSTRAINTS = {
        "MENGATUR": ({"Pasal", "Ayat"}, {"PerbuatanHukum"}),
        "MENETAPKAN_SANKSI": ({"Pasal", "Ayat"}, {"Sanksi"}),
        "BERLAKU_UNTUK": ({"Pasal", "Ayat", "Regulasi"}, {"EntitasHukum"}),
        "MERUJUK": ({"Pasal", "Ayat"}, {"Pasal", "Ayat", "Regulasi"}),
        "MEMUAT": ({"Regulasi", "Bab", "Bagian"}, {"Bab", "Bagian", "Pasal"}),
        "MEMILIKI_AYAT": ({"Pasal"}, {"Ayat"}),
        "MENDEFINISIKAN": ({"Pasal", "Ayat"}, {"KonsepHukum"}),
    }


def validate_extraction(
    nodes: list[dict],
    edges: list[dict],
    strict: bool = False,
) -> tuple[list[dict], list[dict], list[str]]:
    """Validate extracted nodes and edges against the ontology.
    
    Args:
        nodes: List of node dicts
        edges: List of edge dicts
        strict: If True, also enforce edge type constraints
        
    Returns:
        Tuple of (valid_nodes, valid_edges, error_messages)
    """
    valid_nodes = []
    valid_edges = []
    errors = []
    node_ids = {}  # id -> node type
    
    # Validate nodes
    for n in nodes:
        node_type = n.get("type", "")
        label = n.get("label", "")
        node_id = n.get("id", "")
        
        if not node_id:
            errors.append(f"Node missing id: {label}")
            continue
        
        if node_type not in VALID_NODE_TYPES:
            errors.append(f"Invalid node type '{node_type}' for node '{label}' (id={node_id})")
            continue
        
        if not label or len(label.strip()) < 2:
            errors.append(f"Empty or too-short label for node id={node_id}")
            continue
        
        valid_nodes.append(n)
        node_ids[node_id] = node_type
    
    # Validate edges
    for e in edges:
        edge_type = e.get("type", "")
        source_id = e.get("source_id", "") or e.get("source", "")
        target_id = e.get("target_id", "") or e.get("target", "")
        
        if edge_type not in VALID_EDGE_TYPES:
            errors.append(f"Invalid edge type '{edge_type}': {source_id} -> {target_id}")
            continue
        
        if source_id not in node_ids:
            errors.append(f"Edge source not found: {source_id} -[{edge_type}]-> {target_id}")
            continue
        
        if target_id not in node_ids:
            errors.append(f"Edge target not found: {source_id} -[{edge_type}]-> {target_id}")
            continue
        
        # Optional: check edge type constraints
        if strict and edge_type in EDGE_CONSTRAINTS:
            allowed_sources, allowed_targets = EDGE_CONSTRAINTS[edge_type]
            source_type = node_ids[source_id]
            target_type = node_ids[target_id]
            
            if source_type not in allowed_sources:
                errors.append(f"Edge {edge_type}: source type '{source_type}' not allowed (expected {allowed_sources})")
                continue
            if target_type not in allowed_targets:
                errors.append(f"Edge {edge_type}: target type '{target_type}' not allowed (expected {allowed_targets})")
                continue
        
        # Normalize edge to use source_id/target_id keys
        valid_edge = dict(e)
        valid_edge["source_id"] = source_id
        valid_edge["target_id"] = target_id
        valid_edges.append(valid_edge)
    
    return valid_nodes, valid_edges, errors


def validate_triples_file(input_path: str, output_dir: str, strict: bool = False, prompt_id: str = None) -> str:
    """Validate a triples JSON file and save validated output.
    
    Args:
        input_path: Path to triples JSON
        output_dir: Output directory
        strict: Whether to enforce edge constraints
        prompt_id: Prompt identifier for output filename
        
    Returns:
        Path to validated output file
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    document_id = data["document_id"]
    valid_nodes, valid_edges, errors = validate_extraction(
        data["nodes"], data["edges"], strict
    )
    
    # Save validated triples
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{document_id}_{prompt_id}_triples.json" if prompt_id else f"{document_id}_triples.json"
    output_path = os.path.join(output_dir, filename)
    
    output_data = {
        "document_id": document_id,
        "total_nodes": len(valid_nodes),
        "total_edges": len(valid_edges),
        "removed_nodes": len(data["nodes"]) - len(valid_nodes),
        "removed_edges": len(data["edges"]) - len(valid_edges),
        "nodes": valid_nodes,
        "edges": valid_edges,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Save errors log
    if errors:
        error_filename = f"{document_id}_{prompt_id}_errors.log" if prompt_id else f"{document_id}_errors.log"
        error_path = os.path.join(output_dir, error_filename)
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"Validation Errors for {document_id}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total errors: {len(errors)}\n\n")
            for i, err in enumerate(errors, 1):
                f.write(f"{i}. {err}\n")
    
    return output_path


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate extracted KG triples")
    parser.add_argument("--input", required=True, help="Input triples JSON file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--strict", action="store_true", help="Enforce edge constraints")
    args = parser.parse_args()
    
    path = validate_triples_file(args.input, args.output, args.strict)
    print(f"Validated triples saved to: {path}")
