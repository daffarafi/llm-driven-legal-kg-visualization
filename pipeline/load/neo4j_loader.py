"""
Neo4j Loader for Knowledge Graph ingestion.

Loads embedded nodes and edges into Neo4j, creates vector and full-text
indexes for search.

Input:  data/embedded/{document_id}_triples.json
Output: Neo4j database populated with KG
"""

import json
import os
import argparse
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm


class Neo4jLoader:
    """Manages Neo4j connection and KG loading."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        print(f"Connected to Neo4j at {uri}")
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Delete all nodes and relationships (use with caution!)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")
    
    def create_constraints(self):
        """Create uniqueness constraints."""
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    print(f"  Constraint warning: {e}")
        print("Constraints created.")
    
    def create_indexes(self):
        """Create vector and full-text indexes for search."""
        with self.driver.session() as session:
            # Vector index for semantic search
            try:
                session.run("""
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (n:Entity) ON (n.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 3072,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                print("  Vector index created.")
            except Exception as e:
                print(f"  Vector index warning: {e}")
            
            # Full-text index for keyword search
            try:
                session.run("""
                    CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
                    FOR (n:Entity) ON EACH [n.label, n.content]
                """)
                print("  Full-text index created.")
            except Exception as e:
                print(f"  Full-text index warning: {e}")
        
        print("Indexes created.")
    
    def load_nodes(self, nodes: list[dict], show_progress: bool = True):
        """Load nodes into Neo4j. Each node gets both :Entity and :{type} labels."""
        iterator = tqdm(nodes, desc="Loading nodes") if show_progress else nodes
        
        with self.driver.session() as session:
            for node in iterator:
                node_type = node.get("type", "Entity")
                embedding = node.get("embedding", [])
                provenance = node.get("provenance", {})
                
                # Use MERGE to avoid duplicates
                cypher = f"""
                    MERGE (n:Entity:{node_type} {{id: $id}})
                    SET n.label = $label,
                        n.content = $content,
                        n.node_type = $node_type,
                        n.source_document_id = $source_doc,
                        n.source_pages = $source_pages,
                        n.extraction_model = $model,
                        n.created_at = datetime()
                """
                
                params = {
                    "id": node["id"],
                    "label": node.get("label", ""),
                    "content": node.get("content", ""),
                    "node_type": node_type,
                    "source_doc": provenance.get("source_document_id", ""),
                    "source_pages": provenance.get("source_pages", []),
                    "model": provenance.get("extraction_model", ""),
                }
                
                # Add embedding if present
                if embedding and any(v != 0.0 for v in embedding):
                    cypher += ",\n                        n.embedding = $embedding"
                    params["embedding"] = embedding
                
                session.run(cypher, **params)
    
    def load_edges(self, edges: list[dict], show_progress: bool = True):
        """Load edges/relationships into Neo4j."""
        iterator = tqdm(edges, desc="Loading edges") if show_progress else edges
        
        with self.driver.session() as session:
            for edge in iterator:
                source_id = edge.get("source_id", "") or edge.get("source", "")
                target_id = edge.get("target_id", "") or edge.get("target", "")
                edge_type = edge.get("type", "RELATED_TO")
                provenance = edge.get("provenance", {})
                
                # Dynamic relationship type
                cypher = f"""
                    MATCH (a:Entity {{id: $source_id}})
                    MATCH (b:Entity {{id: $target_id}})
                    MERGE (a)-[r:{edge_type}]->(b)
                    SET r.source_document_id = $source_doc,
                        r.created_at = datetime()
                """
                
                try:
                    session.run(
                        cypher,
                        source_id=source_id,
                        target_id=target_id,
                        source_doc=provenance.get("source_document_id", ""),
                    )
                except Exception as e:
                    if show_progress:
                        tqdm.write(f"  [WARN] Edge {source_id} -[{edge_type}]-> {target_id}: {e}")
    
    def load_amendment_kg(self, amendments: list[dict]):
        """Load amendment relationships between UU nodes.
        
        Types: MENGAMANDEMEN, MENGAMANDEMEN_SEBAGIAN, MENCABUT, MENCABUT_SEBAGIAN
        """
        with self.driver.session() as session:
            for amend in amendments:
                cypher = f"""
                    MERGE (a:Entity:UndangUndang {{uu_number: $source_uu}})
                    MERGE (b:Entity:UndangUndang {{uu_number: $target_uu}})
                    MERGE (a)-[:{amend['relation_type']}]->(b)
                """
                session.run(
                    cypher,
                    source_uu=amend["source"],
                    target_uu=amend["target"],
                )
        print(f"Loaded {len(amendments)} amendment relationships.")
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            
            # Count by label
            labels = session.run("""
                MATCH (n) 
                WITH labels(n) AS lbls 
                UNWIND lbls AS lbl 
                WITH lbl WHERE lbl <> 'Entity'
                RETURN lbl, count(*) AS cnt 
                ORDER BY cnt DESC
            """)
            label_counts = {r["lbl"]: r["cnt"] for r in labels}
            
            # Count by relationship type
            rels = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) AS t, count(r) AS cnt 
                ORDER BY cnt DESC
            """)
            rel_counts = {r["t"]: r["cnt"] for r in rels}
            
        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "node_labels": label_counts,
            "relationship_types": rel_counts,
        }
    
    def test_vector_search(self, query_text: str, api_key: str, top_k: int = 5) -> list[dict]:
        """Test semantic search using vector index."""
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Generate query embedding
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query_text,
        )
        query_embedding = result["embedding"]
        
        with self.driver.session() as session:
            results = session.run("""
                CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $embedding)
                YIELD node, score
                RETURN node.id AS id, node.label AS label, node.node_type AS type, score
                ORDER BY score DESC
            """, top_k=top_k, embedding=query_embedding)
            
            return [dict(r) for r in results]


def load_from_file(
    input_path: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    clear_first: bool = False,
) -> dict:
    """Load embedded triples from JSON file into Neo4j.
    
    Returns database stats after loading.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    loader = Neo4jLoader(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        if clear_first:
            loader.clear_database()
        
        loader.create_constraints()
        loader.create_indexes()
        
        print(f"\nLoading {len(data['nodes'])} nodes...")
        loader.load_nodes(data["nodes"])
        
        print(f"\nLoading {len(data['edges'])} edges...")
        loader.load_edges(data["edges"])
        
        stats = loader.get_stats()
        print(f"\n=== Database Stats ===")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total edges: {stats['total_edges']}")
        print(f"Labels: {stats['node_labels']}")
        print(f"Relations: {stats['relationship_types']}")
        
        return stats
    finally:
        loader.close()


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Load KG into Neo4j")
    parser.add_argument("--input", required=True, help="Input embedded triples JSON")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument("--clear", action="store_true", help="Clear database before loading")
    args = parser.parse_args()
    
    load_from_file(args.input, args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.clear)
