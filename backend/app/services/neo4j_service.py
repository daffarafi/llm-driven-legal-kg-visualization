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
    def get_session(cls):
        """Get a session targeting the configured database."""
        return cls.get_driver().session(database=settings.NEO4J_DATABASE)

    @classmethod
    def close(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @classmethod
    def get_stats(cls, doc_id: str | None = None) -> dict:
        """Get KG overview statistics, optionally filtered by document."""
        doc_filter = "WHERE n.source_document_id = $doc_id" if doc_id else ""
        edge_doc_filter = "WHERE a.source_document_id = $doc_id" if doc_id else ""
        params = {"doc_id": doc_id} if doc_id else {}

        with cls.get_session() as s:
            # Node counts by label
            node_counts = s.run(f"""
                MATCH (n)
                {doc_filter}
                WITH labels(n) AS lbls, count(n) AS cnt
                UNWIND lbls AS lbl
                RETURN lbl AS label, sum(cnt) AS count
                ORDER BY count DESC
            """, params).data()

            # Edge counts by type
            edge_counts = s.run(f"""
                MATCH (a)-[r]->()
                {edge_doc_filter}
                RETURN type(r) AS label, count(r) AS count
                ORDER BY count DESC
            """, params).data()

            # Totals
            totals = s.run(f"""
                MATCH (n) {doc_filter} WITH count(n) AS nodes
                OPTIONAL MATCH (a)-[r]->() {'WHERE a.source_document_id = $doc_id' if doc_id else ''}
                WITH nodes, count(r) AS edges
                RETURN nodes, edges
            """, params).single()

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
        doc_ids: list[str] | None = None,
        limit: int = 2000,
    ) -> dict:
        """Get a subgraph with optional filters.
        
        Fetches nodes first, and then retrieves edges between them.
        This avoids slow, memory-intensive nested collect(DISTINCT ...)
        operations in Neo4j, making it significantly faster and safer
        for larger limits.
        """
        where_clauses = []
        params: dict = {"limit": limit}

        if node_types:
            label_checks = " OR ".join(f"n:{t}" for t in node_types)
            where_clauses.append(f"({label_checks})")

        if doc_ids:
            where_clauses.append("n.source_document_id IN $doc_ids")
            params["doc_ids"] = doc_ids

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Step 1: Query nodes matching criteria
        nodes_query = f"""
            MATCH (n)
            {where}
            RETURN elementId(n) AS id,
                   labels(n) AS labels,
                   n.label AS label,
                   coalesce(n.node_type, head(labels(n))) AS node_type,
                   n.source_document_id AS source_document_id,
                   substring(coalesce(n.content, ''), 0, 200) AS content
            LIMIT $limit
        """

        with cls.get_session() as s:
            nodes_records = s.run(nodes_query, params).data()

        if not nodes_records:
            return {"nodes": [], "edges": []}

        # Build nodes list and collect IDs
        nodes = []
        node_ids = []
        for r in nodes_records:
            nid = r["id"]
            node_ids.append(nid)
            nodes.append({
                "id": nid,
                "labels": r["labels"],
                "label": r["label"],
                "node_type": r["node_type"],
                "source_document_id": r["source_document_id"],
                "content": r["content"],
            })

        # Step 2: Query edges connecting the retrieved nodes
        edges_query = """
            MATCH (n)-[r]->(m)
            WHERE elementId(n) IN $node_ids AND elementId(m) IN $node_ids
            RETURN elementId(n) AS source,
                   elementId(m) AS target,
                   type(r) AS type
        """

        with cls.get_session() as s:
            edges_records = s.run(edges_query, {"node_ids": node_ids}).data()

        # Build and optionally filter edges by relation type
        edges = []
        for e in edges_records:
            if relation_types is None or e["type"] in relation_types:
                edges.append(e)

        return {
            "nodes": nodes,
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
                    target_type: labels(m_out),
                    target_source_document_id: m_out.source_document_id
                }) AS outgoing,
                collect(DISTINCT {
                    type: type(r_in),
                    direction: 'incoming',
                    source_id: elementId(m_in),
                    source_label: m_in.label,
                    source_type: labels(m_in),
                    source_source_document_id: m_in.source_document_id
                }) AS incoming
        """
        with cls.get_session() as s:
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

        with cls.get_session() as s:
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
                   n.source_document_id AS source_document_id,
                   substring(coalesce(n.content, ''), 0, 200) AS content
            LIMIT $limit
        """
        with cls.get_session() as s:
            return s.run(cypher, {"query": query, "limit": limit}).data()

    @classmethod
    def batch_keyword_search(cls, queries: list[dict]) -> list[dict]:
        """Search nodes using multiple keywords/phrases in a single batched query."""
        if not queries:
            return []
        
        # Subquery CALL (q) requires Neo4j 5.x.
        # We use a LIMIT 50 in the subquery to ensure enough matching candidate nodes are retrieved.
        cypher = """
            UNWIND $queries AS q
            CALL (q) {
                MATCH (n)
                WHERE toLower(n.label) CONTAINS toLower(q.term)
                   OR toLower(coalesce(n.content, '')) CONTAINS toLower(q.term)
                RETURN elementId(n) AS id,
                       labels(n) AS labels,
                       n.label AS label,
                       n.source_document_id AS source_document_id,
                       substring(coalesce(n.content, ''), 0, 200) AS content,
                       toLower(n.label) CONTAINS toLower(q.term) AS matched_in_label
                LIMIT 50
            }
            RETURN id, labels, label, source_document_id, content,
                   collect(q.term) AS matched_terms,
                   collect(matched_in_label) AS matched_in_labels
        """
        with cls.get_session() as s:
            return s.run(cypher, {"queries": queries}).data()

    @classmethod
    def _semantic_search(cls, query: str, limit: int) -> list[dict]:
        """Vector similarity search using embeddings."""
        # TODO: implement when vector index is ready
        # For now, fallback to keyword
        return cls._keyword_search(query, limit)

    # ------------------------------------------------------------------
    # Document
    # ------------------------------------------------------------------

    @staticmethod
    def _roman_to_int(roman: str) -> int:
        """Convert Roman numeral string to integer for sorting."""
        vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        result = 0
        for i, c in enumerate(roman):
            if i + 1 < len(roman) and vals.get(c, 0) < vals.get(roman[i + 1], 0):
                result -= vals.get(c, 0)
            else:
                result += vals.get(c, 0)
        return result

    @classmethod
    def _sort_by_roman(cls, items: list[dict], label_key: str = "label") -> list[dict]:
        """Sort items by Roman numeral extracted from BAB label."""
        import re
        def sort_key(item):
            label = item.get(label_key, "") or ""
            m = re.match(r'BAB\s+([IVXLC]+)', label)
            return cls._roman_to_int(m.group(1)) if m else 999
        return sorted(items, key=sort_key)

    @staticmethod
    def _sort_by_pasal_num(items: list[dict], label_key: str = "label") -> list[dict]:
        """Sort items by Pasal number extracted from label."""
        import re
        def sort_key(item):
            label = item.get(label_key, "") or ""
            m = re.search(r'Pasal\s+(\d+)', label)
            num = int(m.group(1)) if m else 999
            # Handle sub-labels like '45A', '45B'
            suffix = re.search(r'Pasal\s+\d+([A-Z])', label)
            suffix_val = ord(suffix.group(1)) - ord('A') + 1 if suffix else 0
            # Handle ayat numbers like 'ayat (2a)', 'ayat (7a)'
            ayat = re.search(r'ayat\s*\((\d+)([a-z]?)\)', label)
            ayat_num = int(ayat.group(1)) * 10 + (ord(ayat.group(2)) - ord('a') + 1 if ayat.group(2) else 0) if ayat else 0
            return (num, suffix_val, ayat_num)
        return sorted(items, key=sort_key)

    @classmethod
    def _sort_by_bagian(cls, items: list[dict], label_key: str = "label") -> list[dict]:
        """Sort items by Indonesian ordinal number in Bagian label."""
        import re
        ordinals = {
            "kesatu": 1, "kedua": 2, "ketiga": 3, "keempat": 4, "kelima": 5,
            "keenam": 6, "ketujuh": 7, "kedelapan": 8, "kesembilan": 9, "kesepuluh": 10,
            "kesebelas": 11, "keduabelas": 12, "ketigabelas": 13, "keempatbelas": 14,
            "kelimabelas": 15, "keenambelas": 16, "ketujuhbelas": 17, "kedelapanbelas": 18,
            "kesembilanbelas": 19, "kedua belas": 12, "ketiga belas": 13, "keempat belas": 14,
            "kelima belas": 15, "keenam belas": 16, "ketujuh belas": 17, "kedelapan belas": 18,
            "kesembilan belas": 19
        }
        def sort_key(item):
            label = (item.get(label_key, "") or "").lower()
            # E.g. "bagian kesatu umum" -> extract words after "bagian"
            m = re.search(r'bagian\s+([a-z\s]+?)(?:\s+|$)', label)
            if m:
                word = m.group(1).strip()
                for ord_word, val in ordinals.items():
                    if word.startswith(ord_word):
                        return val
            return 999
        return sorted(items, key=sort_key)

    @classmethod
    def get_document(cls, doc_id: str) -> dict | None:
        """Get document with its hierarchical structure.
        
        Uses Regulasi nodes with hierarchy: Regulasi → Bab → Bagian → Pasal → Ayat.
        """
        with cls.get_session() as s:
            # Get Regulasi node
            reg_result = s.run("""
                MATCH (r:Regulasi)
                WHERE r.id = $id OR r.source_document_id = $id
                   OR elementId(r) = $id
                   OR toLower(r.label) CONTAINS toLower($id)
                RETURN r
            """, {"id": doc_id}).single()

            if not reg_result or not reg_result["r"]:
                return None

            reg_node = reg_result["r"]
            source_doc = reg_node.get("source_document_id", doc_id)

            # Get BABs
            bab_list = s.run("""
                MATCH (r:Regulasi)-[:MEMUAT]->(b:Bab)
                WHERE r.source_document_id = $d
                RETURN elementId(b) AS id, b.label AS label, b.content AS content
            """, {"d": source_doc}).data()

            # Get Bagian with parent BAB
            bagian_list = s.run("""
                MATCH (b:Bab {source_document_id: $d})-[:MEMUAT]->(bg:Bagian)
                RETURN elementId(bg) AS id, bg.label AS label, bg.content AS content,
                       b.label AS bab
            """, {"d": source_doc}).data()

            # Get Pasal — direct under BAB (no Bagian)
            pasal_direct = s.run("""
                MATCH (b:Bab {source_document_id: $d})-[:MEMUAT]->(p:Pasal)
                WHERE NOT EXISTS { (b)-[:MEMUAT]->(:Bagian)-[:MEMUAT]->(p) }
                RETURN elementId(p) AS id, p.label AS label, p.content AS content,
                       b.label AS bab, null AS bagian
            """, {"d": source_doc}).data()

            # Get Pasal — under Bagian
            pasal_bagian = s.run("""
                MATCH (bg:Bagian {source_document_id: $d})-[:MEMUAT]->(p:Pasal)
                OPTIONAL MATCH (b:Bab {source_document_id: $d})-[:MEMUAT]->(bg)
                RETURN elementId(p) AS id, p.label AS label, p.content AS content,
                       b.label AS bab, bg.label AS bagian
            """, {"d": source_doc}).data()

            # Get Pasal — directly under Regulasi (no BAB, e.g. UU_19_2016)
            pasal_regulasi = s.run("""
                MATCH (r:Regulasi {source_document_id: $d})-[:MEMUAT]->(p:Pasal)
                RETURN elementId(p) AS id, p.label AS label, p.content AS content,
                       null AS bab, null AS bagian
            """, {"d": source_doc}).data()

            # Merge pasal lists, deduplicate by id
            pasal_map = {}
            for p in pasal_direct + pasal_bagian + pasal_regulasi:
                if p.get("id") and p["id"] not in pasal_map:
                    pasal_map[p["id"]] = p
            pasal_list = list(pasal_map.values())

            # Get Ayat
            ayat_list = s.run("""
                MATCH (p:Pasal {source_document_id: $d})-[:MEMILIKI_AYAT]->(a:Ayat)
                RETURN elementId(a) AS id, a.label AS label, a.content AS content,
                       p.label AS pasal
            """, {"d": source_doc}).data()

        # Sort everything properly
        bab_list = cls._sort_by_roman(bab_list)
        bagian_list = cls._sort_by_bagian(bagian_list)
        pasal_list = cls._sort_by_pasal_num(pasal_list)
        ayat_list = cls._sort_by_pasal_num(ayat_list)

        doc = dict(reg_node.items())
        doc["id"] = doc_id
        return {
            "document": doc,
            "bab": [b for b in bab_list if b.get("id")],
            "bagian": [bg for bg in bagian_list if bg.get("id")],
            "pasal": pasal_list,
            "ayat": ayat_list,
        }

    @classmethod
    def get_regulations(cls) -> list[dict]:
        """Return all Regulasi nodes with metadata and entity counts."""
        cypher = """
            MATCH (r:Regulasi)
            OPTIONAL MATCH (e {source_document_id: r.source_document_id})
              WHERE e:PerbuatanHukum OR e:EntitasHukum OR e:KonsepHukum OR e:Sanksi
            WITH r, count(DISTINCT e) AS entity_count
            RETURN coalesce(r.doc_id, r.source_document_id, r.id) AS doc_id,
                   r.label                                         AS label,
                   coalesce(r.short_name, r.label)                 AS short_name,
                   r.source_document_id                            AS source_document_id,
                   coalesce(r.jenis, r.regulation_type,
                       CASE
                           WHEN r.source_document_id STARTS WITH 'UU'   THEN 'UU'
                           WHEN r.source_document_id STARTS WITH 'POJK' THEN 'POJK'
                           WHEN r.source_document_id STARTS WITH 'PP'   THEN 'PP'
                           ELSE 'Lainnya'
                       END)                                        AS regulation_type,
                   r.year                                          AS year,
                   r.status                                        AS status,
                   entity_count
            ORDER BY r.label
        """
        with cls.get_session() as s:
            return s.run(cypher).data()


    # ------------------------------------------------------------------
    # Execute raw Cypher (for QA pipeline)
    # ------------------------------------------------------------------

    @classmethod
    def execute_cypher(cls, cypher: str) -> list[dict]:
        """Execute a Cypher query and return results."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"=== EXECUTE CYPHER ===")
        logger.info(f"Query ({len(cypher)} chars): {cypher[:300]}")
        with cls.get_session() as s:
            try:
                results = s.run(cypher).data()
                logger.info(f"Results: {len(results)} rows")
                if results:
                    logger.info(f"First row keys: {list(results[0].keys())}")
                    logger.info(f"First row: {str(results[0])[:200]}")
                return results
            except Exception as e:
                logger.error(f"Cypher execution error: {e}")
                return [{"error": str(e)}]
