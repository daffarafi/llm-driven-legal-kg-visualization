"""
Embedding Generator for Knowledge Graph nodes.

Generates vector embeddings for KG nodes using Google's text-embedding-004 model.
Embeddings are used for semantic search in Neo4j vector index.

Input:  data/deduped/{document_id}_triples.json
Output: data/embedded/{document_id}_triples.json (with embedding vectors)
"""

import json
import os
import time
import argparse
from pathlib import Path

import google.generativeai as genai
from tqdm import tqdm


def generate_embeddings(
    nodes: list[dict],
    api_key: str,
    model: str = "text-embedding-004",
    batch_size: int = 100,
    delay_between_calls: float = 0.5,
) -> dict[str, list[float]]:
    """Generate embedding vectors for KG nodes.
    
    Each node is embedded using: "{label}: {content[:500]}"
    
    Args:
        nodes: List of node dicts
        api_key: Gemini API key
        model: Embedding model name
        batch_size: Max items per API call
        delay_between_calls: Rate limiting delay
        
    Returns:
        Dict mapping node_id -> embedding vector
    """
    genai.configure(api_key=api_key)
    
    embeddings = {}
    texts_to_embed = []
    ids = []
    
    # Prepare texts
    for node in nodes:
        label = node.get("label", "")
        content = node.get("content", "")[:500]
        embed_text = f"{label}: {content}" if content else label
        texts_to_embed.append(embed_text)
        ids.append(node["id"])
    
    # Batch embed
    total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts_to_embed), batch_size), total=total_batches, desc="Generating embeddings"):
        batch_texts = texts_to_embed[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        try:
            result = genai.embed_content(
                model=f"models/{model}",
                content=batch_texts,
            )
            
            for j, emb in enumerate(result["embedding"]):
                embeddings[batch_ids[j]] = emb
                
        except Exception as e:
            print(f"  [ERROR] Batch {i//batch_size}: {e}")
            # Generate zero vectors as fallback
            for bid in batch_ids:
                embeddings[bid] = [0.0] * 768
        
        time.sleep(delay_between_calls)
    
    return embeddings


def embed_triples_file(
    input_path: str,
    output_dir: str,
    api_key: str,
    model: str = "gemini-embedding-001",
) -> str:
    """Generate embeddings for all nodes in a triples file.
    
    Args:
        input_path: Path to deduplicated triples JSON
        output_dir: Output directory
        api_key: Gemini API key
        model: Embedding model name
        
    Returns:
        Path to output file with embeddings
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    document_id = data["document_id"]
    nodes = data["nodes"]
    
    # Generate embeddings
    embeddings = generate_embeddings(nodes, api_key, model)
    
    # Add embeddings to nodes
    for node in nodes:
        node["embedding"] = embeddings.get(node["id"], [0.0] * 768)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{document_id}_triples.json")
    
    output_data = {
        "document_id": document_id,
        "total_nodes": len(nodes),
        "total_edges": len(data["edges"]),
        "embedding_model": model,
        "embedding_dimensions": 768,
        "nodes": nodes,
        "edges": data["edges"],
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
    
    parser = argparse.ArgumentParser(description="Generate embeddings for KG nodes")
    parser.add_argument("--input", required=True, help="Input deduped triples JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="text-embedding-004", help="Embedding model")
    args = parser.parse_args()
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found")
        exit(1)
    
    path = embed_triples_file(args.input, args.output, api_key, args.model)
    print(f"Embeddings saved to: {path}")
