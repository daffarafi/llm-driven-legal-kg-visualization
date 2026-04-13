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
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.driver.verify_connectivity()
        print(f"Connected to Neo4j at {uri} (database: {database})")
    
    def close(self):
        self.driver.close()
    
    def _session(self):
        """Get a session targeting the configured database."""
        return self.driver.session(database=self.database)
    
    def clear_database(self):
        """Delete all nodes and relationships (use with caution!)."""
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")
    
    def create_constraints(self):
        """Create uniqueness constraints."""
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        ]
        with self._session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    print(f"  Constraint warning: {e}")
        print("Constraints created.")
    
    def create_indexes(self):
        """Create vector and full-text indexes for search."""
        with self._session() as session:
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
        
        with self._session() as session:
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
        
        with self._session() as session:
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
        with self._session() as session:
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
    
    def load_regex_references(self, parsed_doc_path: str):
        """Load regex-detected cross-references into Neo4j.
        
        Reads the 'references' field from parsed components and creates
        MERUJUK_DOKUMEN and MERUJUK_PASAL edges.
        """
        with open(parsed_doc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        document_id = data["document_id"]
        ref_count = 0
        
        with self._session() as session:
            for component in data.get("components", []):
                for ref in component.get("references", []):
                    ref_type = ref.get("type", "")
                    source_comp_id = ref.get("source_component", "")
                    
                    if ref_type == "MERUJUK_DOKUMEN":
                        target_doc_id = ref.get("target_doc_id", "")
                        if not target_doc_id:
                            continue
                        # Create edge from source component to target document
                        cypher = """
                            MATCH (src:Entity {source_document_id: $source_doc_id})
                            WHERE src.label CONTAINS $source_label
                            WITH src LIMIT 1
                            MERGE (target:Entity:Peraturan {id: $target_doc_id})
                            MERGE (src)-[r:MERUJUK_DOKUMEN]->(target)
                            SET r.source_text = $source_text,
                                r.detection_method = 'regex',
                                r.created_at = datetime()
                        """
                        # Extract component label from ID
                        parts = source_comp_id.split("__")
                        comp_label = parts[-1].replace("_", " ") if parts else ""
                        
                        try:
                            session.run(cypher,
                                source_doc_id=document_id,
                                source_label=comp_label,
                                target_doc_id=target_doc_id,
                                source_text=ref.get("source_text", "")[:200],
                            )
                            ref_count += 1
                        except Exception as e:
                            pass  # Non-critical
                    
                    elif ref_type in ("MENGUBAH_PASAL", "MENGHAPUS_PASAL", "MENYISIPKAN_PASAL"):
                        # Amendment operations — link to target articles
                        target_article = ref.get("target_article", "")
                        if not target_article:
                            continue
                        
                        cypher = f"""
                            MATCH (amender:Entity {{source_document_id: $amender_doc}})
                            WHERE amender.label CONTAINS $amender_label
                            WITH amender LIMIT 1
                            MATCH (target:Entity)
                            WHERE target.label CONTAINS $target_label
                            AND target.source_document_id <> $amender_doc
                            WITH amender, target LIMIT 1
                            MERGE (amender)-[r:{ref_type}]->(target)
                            SET r.source_text = $source_text,
                                r.detection_method = 'regex',
                                r.created_at = datetime()
                        """
                        parts = source_comp_id.split("__")
                        comp_label = parts[-1].replace("_", " ") if parts else ""
                        
                        try:
                            session.run(cypher,
                                amender_doc=document_id,
                                amender_label=comp_label,
                                target_label=target_article,
                                source_text=ref.get("source_text", "")[:200],
                            )
                            ref_count += 1
                        except Exception as e:
                            pass
        
        return ref_count
    
    def load_versi_pasal(self, regulation_list_path: str):
        """Create VersiPasal nodes for tracking amendment versions.
        
        Lex2KG concept: Each amended article gets a VersiPasal node that tracks
        the original version and the amended version.
        
        For each article in amended_articles:
        - Creates VersiPasal node (version_original, version_amended)
        - Links: (VersiPasal_original) -[:DIAMANDEMEN_MENJADI]-> (VersiPasal_amended)
        - Links: (Pasal_original) -[:MEMILIKI_VERSI]-> (VersiPasal_original)
        - Links: (Pasal_amended) -[:MEMILIKI_VERSI]-> (VersiPasal_amended)
        """
        with open(regulation_list_path, "r", encoding="utf-8") as f:
            regulations = json.load(f)
        
        versi_count = 0
        
        with self._session() as session:
            for reg in regulations:
                amender_doc_id = reg["doc_id"]
                
                # Only process docs with amended_articles
                for art in reg.get("amended_articles", []):
                    article = art["article"]  # e.g., "Pasal 27"
                    action = art["action"]    # MENGUBAH, MENYISIPKAN, MENGHAPUS
                    description = art.get("description", "")
                    
                    # Find the target doc being amended
                    target_doc_ids = [
                        r["target_doc_id"]
                        for r in reg.get("relations", [])
                        if r["type"] == "MENGAMANDEMEN"
                    ]
                    
                    if not target_doc_ids:
                        continue
                    
                    target_doc_id = target_doc_ids[0]
                    
                    if action == "MENGUBAH":
                        # Create two VersiPasal nodes
                        versi_original_id = f"VersiPasal_{target_doc_id}__{article.replace(' ', '_')}_v1"
                        versi_amended_id = f"VersiPasal_{amender_doc_id}__{article.replace(' ', '_')}_v2"
                        
                        cypher = """
                            MERGE (v1:Entity:VersiPasal {id: $v1_id})
                            SET v1.label = $v1_label,
                                v1.node_type = 'VersiPasal',
                                v1.version = 1,
                                v1.source_document_id = $target_doc,
                                v1.status = 'diamandemen',
                                v1.content = $description,
                                v1.created_at = datetime()
                            
                            MERGE (v2:Entity:VersiPasal {id: $v2_id})
                            SET v2.label = $v2_label,
                                v2.node_type = 'VersiPasal',
                                v2.version = 2,
                                v2.source_document_id = $amender_doc,
                                v2.status = 'berlaku',
                                v2.content = $description,
                                v2.created_at = datetime()
                            
                            MERGE (v1)-[:DIAMANDEMEN_MENJADI]->(v2)
                        """
                        session.run(cypher,
                            v1_id=versi_original_id,
                            v1_label=f"{article} (versi {target_doc_id})",
                            v2_id=versi_amended_id,
                            v2_label=f"{article} (versi {amender_doc_id})",
                            target_doc=target_doc_id,
                            amender_doc=amender_doc_id,
                            description=description,
                        )
                        
                        # Link VersiPasal to actual Pasal nodes if they exist
                        link_cypher = """
                            OPTIONAL MATCH (p1:Entity)
                            WHERE p1.label CONTAINS $article AND p1.source_document_id = $target_doc
                            WITH p1
                            WHERE p1 IS NOT NULL
                            MATCH (v1:VersiPasal {id: $v1_id})
                            MERGE (p1)-[:MEMILIKI_VERSI]->(v1)
                        """
                        session.run(link_cypher,
                            article=article,
                            target_doc=target_doc_id,
                            v1_id=versi_original_id,
                        )
                        
                        link_cypher2 = """
                            OPTIONAL MATCH (p2:Entity)
                            WHERE p2.label CONTAINS $article AND p2.source_document_id = $amender_doc
                            WITH p2
                            WHERE p2 IS NOT NULL
                            MATCH (v2:VersiPasal {id: $v2_id})
                            MERGE (p2)-[:MEMILIKI_VERSI]->(v2)
                        """
                        session.run(link_cypher2,
                            article=article,
                            amender_doc=amender_doc_id,
                            v2_id=versi_amended_id,
                        )
                        
                        versi_count += 1
                    
                    elif action == "MENYISIPKAN":
                        # Inserted articles only have v1 (new)
                        versi_id = f"VersiPasal_{amender_doc_id}__{article.replace(' ', '_')}_v1"
                        cypher = """
                            MERGE (v:Entity:VersiPasal {id: $v_id})
                            SET v.label = $label,
                                v.node_type = 'VersiPasal',
                                v.version = 1,
                                v.source_document_id = $amender_doc,
                                v.status = 'baru (disisipkan)',
                                v.content = $description,
                                v.created_at = datetime()
                        """
                        session.run(cypher,
                            v_id=versi_id,
                            label=f"{article} (disisipkan oleh {amender_doc_id})",
                            amender_doc=amender_doc_id,
                            description=description,
                        )
                        versi_count += 1
        
        print(f"Created {versi_count} VersiPasal nodes.")
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._session() as session:
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
        
        with self._session() as session:
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
