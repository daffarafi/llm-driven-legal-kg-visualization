"""Neo4j database service — all Cypher queries go through here."""

from neo4j import GraphDatabase
from app.config import settings


class Neo4jService:
    """Singleton-style Neo4j driver wrapper."""

    _driver = None

    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            cls._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
        return cls._driver

    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @classmethod
    def get_stats(cls) -> dict:
        """Get KG overview statistics."""
        with cls.get_driver().session() as s:
            # Node counts by label
            node_counts = s.run("""
                MATCH (n)
                WITH labels(n) AS lbls, count(n) AS cnt
                UNWIND lbls AS lbl
                RETURN lbl AS label, sum(cnt) AS count
                ORDER BY count DESC
            """).data()

            # Edge counts by type
            edge_counts = s.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS label, count(r) AS count
                ORDER BY count DESC
            """).data()

            # Totals
            totals = s.run("""
                MATCH (n) WITH count(n) AS nodes
                MATCH ()-[r]->() WITH nodes, count(r) AS edges
                RETURN nodes, edges
            """).single()

        return {
            "total_nodes": totals["nodes"],
            "total_edges": totals["edges"],
            "node_types": node_counts,
            "edge_types": edge_counts,
        }

    # ------------------------------------------------------------------
    # Graph retrieval
    # ------------------------------------------------------------------

    @classmethod
    def get_graph(
        cls,
        node_types: list[str] | None = None,
        relation_types: list[str] | None = None,
        limit: int = 100,
    ) -> dict:
        """Get a subgraph with optional filters."""
        where_clauses = []
        params: dict = {"limit": limit}

        if node_types:
            # Filter by node labels
            label_checks = " OR ".join(f"n:{t}" for t in node_types)
            where_clauses.append(f"({label_checks})")

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        query = f"""
            MATCH (n)
            {where}
            WITH n LIMIT $limit
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN collect(DISTINCT {{
                id: elementId(n),
                labels: labels(n),
                label: n.label,
                content: substring(coalesce(n.content, ''), 0, 200)
            }}) AS source_nodes,
            collect(DISTINCT {{
                id: elementId(m),
                labels: labels(m),
                label: m.label,
                content: substring(coalesce(m.content, ''), 0, 200)
            }}) AS target_nodes,
            collect(DISTINCT {{
                source: elementId(n),
                target: elementId(m),
                type: type(r)
            }}) AS edges
        """

        with cls.get_driver().session() as s:
            result = s.run(query, params).single()

        # Merge source and target nodes, deduplicate
        nodes_map = {}
        for n in (result["source_nodes"] or []) + (result["target_nodes"] or []):
            if n and n.get("id"):
                nodes_map[n["id"]] = n

        # Filter edges
        edges = []
        for e in result["edges"] or []:
            if e and e.get("source") and e.get("target"):
                if relation_types is None or e["type"] in relation_types:
                    edges.append(e)

        return {
            "nodes": list(nodes_map.values()),
            "edges": edges,
        }

    # ------------------------------------------------------------------
    # Node detail
    # ------------------------------------------------------------------

    @classmethod
    def get_node(cls, node_id: str) -> dict | None:
        """Get single node with all properties and relations."""
        query = """
            MATCH (n) WHERE elementId(n) = $id
            OPTIONAL MATCH (n)-[r_out]->(m_out)
            OPTIONAL MATCH (m_in)-[r_in]->(n)
            RETURN n,
                collect(DISTINCT {
                    type: type(r_out),
                    direction: 'outgoing',
                    target_id: elementId(m_out),
                    target_label: m_out.label,
                    target_type: labels(m_out)
                }) AS outgoing,
                collect(DISTINCT {
                    type: type(r_in),
                    direction: 'incoming',
                    source_id: elementId(m_in),
                    source_label: m_in.label,
                    source_type: labels(m_in)
                }) AS incoming
        """
        with cls.get_driver().session() as s:
            result = s.run(query, {"id": node_id}).single()

        if not result or not result["n"]:
            return None

        node = result["n"]
        props = dict(node.items())

        return {
            "id": node_id,
            "labels": list(node.labels),
            "properties": props,
            "outgoing": [r for r in result["outgoing"] if r.get("target_id")],
            "incoming": [r for r in result["incoming"] if r.get("source_id")],
        }

    # ------------------------------------------------------------------
    # Subgraph from node
    # ------------------------------------------------------------------

    @classmethod
    def get_node_subgraph(cls, node_id: str, depth: int = 1) -> dict:
        """Get subgraph around a node up to given depth."""
        query = """
            MATCH (start) WHERE elementId(start) = $id
            CALL apoc.path.subgraphAll(start, {maxLevel: $depth})
            YIELD nodes, relationships
            UNWIND nodes AS n
            WITH collect(DISTINCT {
                id: elementId(n),
                labels: labels(n),
                label: n.label,
                content: substring(coalesce(n.content, ''), 0, 200)
            }) AS nodeList, relationships
            UNWIND relationships AS r
            RETURN nodeList AS nodes,
                collect(DISTINCT {
                    source: elementId(startNode(r)),
                    target: elementId(endNode(r)),
                    type: type(r)
                }) AS edges
        """

        # Fallback if APOC not available
        fallback_query = """
            MATCH path = (start)-[*1..{depth}]-(connected)
            WHERE elementId(start) = $id
            WITH nodes(path) AS pathNodes, relationships(path) AS pathRels
            UNWIND pathNodes AS n
            WITH collect(DISTINCT {{
                id: elementId(n),
                labels: labels(n),
                label: n.label,
                content: substring(coalesce(n.content, ''), 0, 200)
            }}) AS nodes, pathRels
            UNWIND pathRels AS r
            RETURN nodes,
                collect(DISTINCT {{
                    source: elementId(startNode(r)),
                    target: elementId(endNode(r)),
                    type: type(r)
                }}) AS edges
        """.format(depth=min(depth, 3))

        with cls.get_driver().session() as s:
            try:
                result = s.run(query, {"id": node_id, "depth": min(depth, 3)}).single()
            except Exception:
                result = s.run(fallback_query, {"id": node_id}).single()

        if not result:
            return {"nodes": [], "edges": []}

        return {
            "nodes": result["nodes"] or [],
            "edges": result["edges"] or [],
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @classmethod
    def search(cls, query: str, mode: str = "keyword", limit: int = 20) -> list[dict]:
        """Search nodes by keyword or semantic similarity."""
        if mode == "semantic":
            return cls._semantic_search(query, limit)
        return cls._keyword_search(query, limit)

    @classmethod
    def _keyword_search(cls, query: str, limit: int) -> list[dict]:
        """Full-text search on label and content properties."""
        cypher = """
            MATCH (n)
            WHERE toLower(n.label) CONTAINS toLower($query)
               OR toLower(coalesce(n.content, '')) CONTAINS toLower($query)
            RETURN elementId(n) AS id,
                   labels(n) AS labels,
                   n.label AS label,
                   substring(coalesce(n.content, ''), 0, 200) AS content
            LIMIT $limit
        """
        with cls.get_driver().session() as s:
            return s.run(cypher, {"query": query, "limit": limit}).data()

    @classmethod
    def _semantic_search(cls, query: str, limit: int) -> list[dict]:
        """Vector similarity search using embeddings."""
        # TODO: implement when vector index is ready
        # For now, fallback to keyword
        return cls._keyword_search(query, limit)

    # ------------------------------------------------------------------
    # Document
    # ------------------------------------------------------------------

    @classmethod
    def get_document(cls, doc_id: str) -> dict | None:
        """Get document with its hierarchical structure."""
        query = """
            MATCH (u:UndangUndang)
            WHERE elementId(u) = $id OR u.label CONTAINS $id
            OPTIONAL MATCH (u)-[:MEMUAT]->(bab:Bab)
            OPTIONAL MATCH (bab)-[:MEMUAT|MEMILIKI_PASAL]->(pasal:Pasal)
            OPTIONAL MATCH (pasal)-[:MEMILIKI_AYAT]->(ayat:Ayat)
            RETURN u,
                collect(DISTINCT {
                    id: elementId(bab),
                    label: bab.label,
                    content: bab.content
                }) AS bab_list,
                collect(DISTINCT {
                    id: elementId(pasal),
                    label: pasal.label,
                    content: pasal.content,
                    bab: bab.label
                }) AS pasal_list,
                collect(DISTINCT {
                    id: elementId(ayat),
                    label: ayat.label,
                    content: ayat.content,
                    pasal: pasal.label
                }) AS ayat_list
        """
        with cls.get_driver().session() as s:
            result = s.run(query, {"id": doc_id}).single()

        if not result or not result["u"]:
            return None

        doc = dict(result["u"].items())
        doc["id"] = doc_id

        return {
            "document": doc,
            "bab": [b for b in result["bab_list"] if b.get("id")],
            "pasal": [p for p in result["pasal_list"] if p.get("id")],
            "ayat": [a for a in result["ayat_list"] if a.get("id")],
        }

    # ------------------------------------------------------------------
    # Execute raw Cypher (for QA pipeline)
    # ------------------------------------------------------------------

    @classmethod
    def execute_cypher(cls, cypher: str) -> list[dict]:
        """Execute a Cypher query and return results."""
        with cls.get_driver().session() as s:
            try:
                return s.run(cypher).data()
            except Exception as e:
                return [{"error": str(e)}]
