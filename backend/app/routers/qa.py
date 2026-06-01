"""QA API router — hybrid pipeline: question → keyword search → context → answer."""

import re
from fastapi import APIRouter
from app.models.schemas import QARequest, QAResponse, QAProcessStep
from app.services.neo4j_service import Neo4jService
from app.services.llm_service import LLMService

router = APIRouter()


def get_indonesian_root(word: str) -> str:
    w = word
    for suffix in ['nya', 'lah', 'kah']:
        if w.endswith(suffix):
            w = w[:-len(suffix)]
            
    for suffix in ['kan', 'an', 'i']:
        if w.endswith(suffix):
            w = w[:-len(suffix)]
            break
            
    if w.startswith('di'):
        w = w[2:]
    elif w.startswith('ter'):
        w = w[3:]
    elif w.startswith('se'):
        w = w[2:]
    elif w.startswith('ke') and w != 'ke':
        w = w[2:]
    elif w.startswith('me'):
        if w.startswith('meng') or w.startswith('meny') or w.startswith('memb') or w.startswith('mend') or w.startswith('ment') or w.startswith('menc') or w.startswith('menj'):
            if w.startswith('meng'):
                w = w[4:]
            elif w.startswith('meny'):
                w = 's' + w[4:]
            elif w.startswith('memb'):
                w = w[3:]
            elif w.startswith('men'):
                w = w[3:]
        else:
            w = w[2:]
    elif w.startswith('pe'):
        if w.startswith('peng') or w.startswith('peny') or w.startswith('pemb') or w.startswith('pend') or w.startswith('pent') or w.startswith('penc') or w.startswith('penj'):
            if w.startswith('peng'):
                w = w[4:]
            elif w.startswith('peny'):
                w = 's' + w[4:]
            elif w.startswith('pemb'):
                w = w[3:]
            elif w.startswith('pen'):
                w = w[3:]
        else:
            w = w[2:]
            
    return w



def _format_kg_context(records: list[dict]) -> str:
    """Format Neo4j results into readable text."""
    if not records:
        return "(tidak ada hasil)"

    # Smart Numerical Sorting: detect if records contain Pasal or Bab references and sort them naturally
    sort_key_name = None
    first_rec = records[0]
    for k in ["pasal", "ayat", "label", "sumber", "tujuan", "pasal_sanksi", "pasal_larangan"]:
        if k in first_rec and isinstance(first_rec[k], str) and any(kw in first_rec[k] for kw in ["Pasal", "Bab", "BAB"]):
            sort_key_name = k
            break

    if sort_key_name:
        import re
        roman_vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        def roman_to_int(roman: str) -> int:
            res = 0
            for idx, char in enumerate(roman):
                if idx + 1 < len(roman) and roman_vals.get(char, 0) < roman_vals.get(roman[idx + 1], 0):
                    res -= roman_vals.get(char, 0)
                else:
                    res += roman_vals.get(char, 0)
            return res

        def extract_sort_key(item):
            val = str(item.get(sort_key_name, "") or "")
            # Check Bab
            m_bab = re.search(r'(?i)BAB\s+([IVXLC]+)', val)
            if m_bab:
                return (0, roman_to_int(m_bab.group(1).upper()), 0, 0)
            # Check Pasal
            m_pasal = re.search(r'(?i)Pasal\s+(\d+)', val)
            if m_pasal:
                num = int(m_pasal.group(1))
                # Suffix e.g. 45A -> A
                m_suffix = re.search(r'(?i)Pasal\s+\d+([A-Z])', val)
                suffix_val = ord(m_suffix.group(1).upper()) - ord('A') + 1 if m_suffix else 0
                # Ayat e.g. ayat (2)
                m_ayat = re.search(r'(?i)ayat\s*\((\d+)', val)
                ayat_num = int(m_ayat.group(1)) if m_ayat else 0
                return (1, num, suffix_val, ayat_num)
            return (2, 0, 0, 0)

        try:
            records = sorted(records, key=extract_sort_key)
        except Exception:
            pass

    lines = []
    seen_large_strings = []
    for i, r in enumerate(records[:300], 1):
        parts = []
        for k, v in r.items():
            if v is not None and k != "error":
                val = str(v)
                if len(val) > 50:
                    is_redundant = False
                    val_clean = "".join(val.split()).lower()
                    for seen in seen_large_strings:
                        seen_clean = "".join(seen.split()).lower()
                        if val_clean in seen_clean or seen_clean in val_clean:
                            is_redundant = True
                            break
                    if is_redundant:
                        val = "(sudah termuat di atas)"
                    else:
                        seen_large_strings.append(val)
                parts.append(f"{k}: {val}")
        lines.append(f"{i}. " + " | ".join(parts))
    return "\n".join(lines)


def _extract_references(text: str) -> list[str]:
    """Extract legal references from answer text."""
    refs = set()
    for m in re.finditer(r"Pasal\s+\d+(?:\s+ayat\s+\(\d+\))?", text):
        refs.add(m.group())
    for m in re.finditer(r"Bab\s+[IVXLCDM\d]+", text, re.IGNORECASE):
        # Normalise to uppercase like "BAB VII" to match database node labels
        ref_upper = m.group().upper()
        # Clean any double spaces if any
        ref_upper = re.sub(r'\s+', ' ', ref_upper)
        refs.add(ref_upper)
    for m in re.finditer(r"UU\s+(?:No\.\s*)?\d+(?:\s+Tahun\s+\d+)?", text):
        refs.add(m.group())
    return sorted(refs)


def _search_kg_by_keywords(question: str, doc_ids: list[str] | None = None) -> list[dict]:
    """Search KG using multiple keyword and phrase strategies to find relevant nodes."""
    stopwords = {
        'apa', 'apakah', 'adakah', 'siapa', 'bagaimana', 'mengapa', 'dimana', 'kapan', 'berapa', 
        'yang', 'di', 'dan', 'atau', 'itu', 'ini', 'adalah', 'menurut', 'dalam', 'untuk', 'dari', 
        'ke', 'pada', 'dengan', 'oleh', 'tentang', 'saja', 'ketentuan', 'persyaratan', 'syarat', 
        'minimum', 'maksimum', 'aturan', 'hukum', 'regulasi', 'pasal', 'ayat', 'bab', 'terkait',
        'ada', 'bisa', 'boleh', 'jika', 'kalau', 'seperti', 'sebagaimana', 'saya', 'secara'
    }
    
    # Common generic legal verbs and nouns
    generic_words = {
        'sistem', 'elektronik', 'penyelenggara', 'penyelenggaraan', 'informasi', 'dokumen', 
        'transaksi', 'teknologi', 'terhadap', 'tentang', 'melalui', 'penggunaan', 'pemanfaatan',
        'kegiatan', 'pelaksanaan', 'berdasarkan', 'sistem elektronik', 'lakukan', 'melakukan',
        'dilakukan', 'laku', 'merasa', 'rasa', 'dapat', 'oleh', 'hukum'
    }
    
    # 1. Strategy 1: Extract Pasal matches (e.g. "Pasal 27")
    pasal_matches = re.findall(r'(?i)pasal\s+\d+', question)
    pasal_terms = [p.capitalize() for p in pasal_matches]
    
    # Clean question
    cleaned_q = re.sub(r'[^\w\s]', ' ', question.lower())
    words = [w.strip() for w in cleaned_q.split() if w.strip()]
    
    # 2. Extract single-word keywords (excluding stopwords)
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    non_generic_keywords = [w for w in keywords if w not in generic_words]
    generic_keywords = [w for w in keywords if w in generic_words]
    
    non_generic_keywords = sorted(non_generic_keywords, key=len, reverse=True)
    generic_keywords = sorted(generic_keywords, key=len, reverse=True)
    
    # Extract roots of non-generic keywords
    roots = []
    for k in non_generic_keywords:
        root = get_indonesian_root(k)
        if len(root) > 2 and root != k and root not in stopwords and root not in generic_words:
            roots.append(root)
    roots = sorted(roots, key=len, reverse=True)
    
    # 3. Extract 2-word and 3-word phrases
    phrases_2 = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i+1]
        if (w1 in stopwords and w2 in stopwords) or len(w1) <= 2 or len(w2) <= 2:
            continue
        phrases_2.append(f"{w1} {w2}")
        
    phrases_3 = []
    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i+1], words[i+2]
        if (w1 in stopwords and w2 in stopwords and w3 in stopwords) or len(w1) <= 2 or len(w3) <= 2:
            continue
        phrases_3.append(f"{w1} {w2} {w3}")
        
    # Prioritize phrases: those containing non-generic keywords first
    def contains_non_generic(phrase):
        phrase_words = phrase.split()
        return any(w in non_generic_keywords or w in roots for w in phrase_words)
        
    non_generic_phrases_3 = sorted([p for p in phrases_3 if contains_non_generic(p)], key=len, reverse=True)
    non_generic_phrases_2 = sorted([p for p in phrases_2 if contains_non_generic(p)], key=len, reverse=True)
    
    # Assign weights to search terms
    weights = {}
    for p in pasal_terms:
        weights[p] = 10.0
    for p in non_generic_phrases_3:
        weights[p] = 5.0
    for p in non_generic_phrases_2:
        weights[p] = 4.5
    for k in non_generic_keywords:
        weights[k] = 4.0
    for r in roots:
        weights[r] = 3.5
    for k in generic_keywords:
        weights[k] = 1.0
        g_root = get_indonesian_root(k)
        if len(g_root) > 2 and g_root not in weights:
            weights[g_root] = 0.8
            
    # Combine search terms in logical order
    ordered_terms = []
    for p in pasal_terms:
        if p not in ordered_terms:
            ordered_terms.append(p)
    for k in non_generic_keywords:
        if k not in ordered_terms:
            ordered_terms.append(k)
    for r in roots:
        if r not in ordered_terms:
            ordered_terms.append(r)
    for p in non_generic_phrases_3:
        if p not in ordered_terms:
            ordered_terms.append(p)
    for p in non_generic_phrases_2:
        if p not in ordered_terms:
            ordered_terms.append(p)
    for k in generic_keywords:
        if k not in ordered_terms:
            ordered_terms.append(k)
    for k in weights:
        if k not in ordered_terms:
            ordered_terms.append(k)
            
    # Collect search queries (take top 15 terms to be queried)
    batch_queries = [{"term": t} for t in ordered_terms[:15]]
    if not batch_queries:
        return []
        
    results = Neo4jService.batch_keyword_search(batch_queries)
    
    # Score and Rank nodes
    scored_nodes = []
    for node in results:
        if doc_ids and node.get("source_document_id") and node["source_document_id"] not in doc_ids:
            continue
            
        score = 0.0
        matched_terms = node.get("matched_terms", [])
        matched_in_labels = node.get("matched_in_labels", [])
        
        for term, in_label in zip(matched_terms, matched_in_labels):
            term_weight = weights.get(term, 1.0)
            factor = 2.0 if in_label else 1.0
            score += term_weight * factor
            
        # Structure type boost (Pasal/Ayat are main structural elements)
        labels = node.get("labels", [])
        if "Pasal" in labels:
            score += 0.5
        elif "Ayat" in labels:
            score += 0.3
            
        scored_nodes.append((score, node))
        
    scored_nodes = sorted(scored_nodes, key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_nodes[:15]]



def _build_graph_from_cypher(cypher_query: str, cypher_results: list[dict]) -> dict:
    """Build a mini-graph from Cypher query results.

    Re-runs a modified query that returns elementId, labels, and label
    for every node mentioned in the original results so the frontend
    can render them correctly.
    """
    graph_nodes: dict[str, dict] = {}
    graph_edges: list[dict] = []

    # Extract node labels and document IDs from results
    node_labels: list[str] = []
    valid_doc_ids = set()
    for row in cypher_results:
        for k, v in row.items():
            if isinstance(v, str) and v.strip():
                v_clean = v.strip()
                node_labels.append(v_clean)
                if k in ("regulasi_id", "source_document_id"):
                    valid_doc_ids.add(v_clean)
                elif re.match(r'^[A-Z0-9]{2,10}_\d+_\d+$', v_clean):
                    valid_doc_ids.add(v_clean)

    if not node_labels:
        return {"nodes": [], "edges": []}

    # Normalize labels and find all matching nodes in a single batch query
    labels_lower = list(set(lbl.lower().strip() for lbl in node_labels if lbl and lbl.strip()))

    with Neo4jService.get_session() as s:
        if valid_doc_ids:
            nodes_res = s.run("""
                MATCH (n)
                WHERE toLower(n.label) IN $labels_lower
                  AND n.source_document_id IN $doc_ids
                RETURN elementId(n) AS id,
                       labels(n) AS labels,
                       n.label AS label,
                       n.source_document_id AS source_document_id
            """, {"labels_lower": labels_lower, "doc_ids": list(valid_doc_ids)})
        else:
            nodes_res = s.run("""
                MATCH (n)
                WHERE toLower(n.label) IN $labels_lower
                RETURN elementId(n) AS id,
                       labels(n) AS labels,
                       n.label AS label,
                       n.source_document_id AS source_document_id
            """, {"labels_lower": labels_lower})

        for r in nodes_res:
            nid = r["id"]
            if nid and nid not in graph_nodes:
                graph_nodes[nid] = {
                    "id": nid,
                    "labels": r["labels"],
                    "label": r["label"],
                    "source_document_id": r["source_document_id"],
                }

    # Get relationships between found nodes
    node_ids = list(graph_nodes.keys())
    if len(node_ids) >= 2:
        try:
            id_list = "[" + ",".join(f'"{nid}"' for nid in node_ids) + "]"
            rel_query = f"""
                MATCH (a)-[r]->(b)
                WHERE elementId(a) IN {id_list}
                  AND elementId(b) IN {id_list}
                RETURN elementId(a) AS source,
                       elementId(b) AS target,
                       type(r) AS type
            """
            rels = Neo4jService.execute_cypher(rel_query)
            for rel in rels:
                if rel.get("source") and rel.get("target") and not rel.get("error"):
                    graph_edges.append({
                        "source": rel["source"],
                        "target": rel["target"],
                        "type": rel.get("type", ""),
                    })
        except Exception:
            pass  # Relationship query is optional

    return {
        "nodes": list(graph_nodes.values()),
        "edges": graph_edges,
    }


def _enrich_with_relations(nodes: list[dict]) -> tuple[list[dict], dict]:
    """Get relationships for found nodes to build richer context + mini graph."""
    enriched = []
    graph_nodes = {}  # id -> {id, labels, label}
    graph_edges = []

    # Add found nodes to graph
    for node in nodes:
        nid = node.get("id", "")
        if nid:
            graph_nodes[nid] = {
                "id": nid,
                "labels": node.get("labels", []),
                "label": node.get("label", ""),
                "source_document_id": node.get("source_document_id") or node.get("properties", {}).get("source_document_id", ""),
            }

    # Extract up to 15 node IDs to fetch in a single batch query
    node_ids = [n.get("id") for n in nodes[:15] if n.get("id")]
    if not node_ids:
        return enriched, {"nodes": list(graph_nodes.values()), "edges": graph_edges}

    details_map = {}
    with Neo4jService.get_session() as s:
        res = s.run("""
            MATCH (n) WHERE elementId(n) IN $ids
            OPTIONAL MATCH (n)-[r_out]->(m_out)
            OPTIONAL MATCH (m_in)-[r_in]->(n)
            RETURN elementId(n) AS node_id,
                n,
                collect(DISTINCT {
                    type: type(r_out),
                    target_id: elementId(m_out),
                    target_label: m_out.label,
                    target_type: labels(m_out),
                    target_source_document_id: m_out.source_document_id
                }) AS outgoing,
                collect(DISTINCT {
                    type: type(r_in),
                    source_id: elementId(m_in),
                    source_label: m_in.label,
                    source_type: labels(m_in),
                    source_source_document_id: m_in.source_document_id
                }) AS incoming
        """, {"ids": node_ids})
        for r in res:
            nid = r["node_id"]
            node = r["n"]
            details_map[nid] = {
                "id": nid,
                "labels": list(node.labels),
                "properties": dict(node.items()),
                "outgoing": [o for o in r["outgoing"] if o.get("target_id")],
                "incoming": [i for i in r["incoming"] if i.get("source_id")],
            }

    for node in nodes[:15]:
        node_id = node.get("id", "")
        if not node_id or node_id not in details_map:
            enriched.append(node)
            continue

        detail = details_map[node_id]
        entry = {
            "label": node.get("label", ""),
            "type": ", ".join(detail.get("labels", [])),
            "content": str(detail.get("properties", {}).get("content", ""))[:10000],
        }

        # Add outgoing relations
        out_rels = detail.get("outgoing", [])
        if out_rels:
            rel_strs = []
            for r in out_rels[:8]:
                rel_strs.append(f"{r.get('type', '')} → {r.get('target_label', '')}")
                # Add relation target to graph
                tid = r.get("target_id", "")
                if tid:
                    if tid not in graph_nodes:
                        graph_nodes[tid] = {
                            "id": tid,
                            "labels": r.get("target_type", []),
                            "label": r.get("target_label", ""),
                            "source_document_id": r.get("target_source_document_id", ""),
                        }
                    graph_edges.append({
                        "source": node_id,
                        "target": tid,
                        "type": r.get("type", ""),
                    })
            entry["relasi_keluar"] = "; ".join(rel_strs)

        # Add incoming relations
        in_rels = detail.get("incoming", [])
        if in_rels:
            rel_strs = []
            dasar_hukum = []
            for r in in_rels[:8]:
                src_lbl = r.get('source_label', '')
                rel_type = r.get('type', '')
                rel_strs.append(f"{src_lbl} → {rel_type}")
                if src_lbl and (src_lbl.startswith("Pasal") or src_lbl.startswith("Ayat") or src_lbl.lower().startswith("bab")):
                    dasar_hukum.append(src_lbl)
                sid = r.get("source_id", "")
                if sid:
                    if sid not in graph_nodes:
                        graph_nodes[sid] = {
                            "id": sid,
                            "labels": r.get("source_type", []),
                            "label": r.get("source_label", ""),
                            "source_document_id": r.get("source_source_document_id", ""),
                        }
                    graph_edges.append({
                        "source": sid,
                        "target": node_id,
                        "type": r.get("type", ""),
                    })
            entry["relasi_masuk"] = "; ".join(rel_strs)
            if dasar_hukum:
                entry["dasar_hukum"] = ", ".join(sorted(list(set(dasar_hukum))))

        enriched.append(entry)

    graph = {
        "nodes": list(graph_nodes.values()),
        "edges": graph_edges,
    }
    return enriched, graph


def _connect_to_regulasi(mini_graph: dict) -> dict:
    """For each node in the mini_graph, query its path up to its Regulasi node,
    and add the parent nodes (Bab, Bagian, Regulasi) and their edges to the graph.
    """
    if not mini_graph or not mini_graph.get("nodes"):
        return mini_graph

    graph_nodes = {n["id"]: n for n in mini_graph["nodes"]}
    graph_edges = list(mini_graph.get("edges", []))
    existing_edges = set(f"{e['source']}->{e['target']}" for e in graph_edges)

    # Find structural nodes (Pasal, Ayat, Bagian, Bab) to trace upwards
    target_node_ids = []
    for node in mini_graph["nodes"]:
        labels = node.get("labels", [])
        if any(l in labels for l in ["Pasal", "Ayat", "Bagian", "Bab"]):
            target_node_ids.append(node["id"])

    if not target_node_ids:
        return mini_graph

    # Query paths up to Regulasi in a single batch neo4j query
    with Neo4jService.get_session() as s:
        result = s.run("""
            MATCH path = (r:Regulasi)-[:MEMUAT|MEMILIKI_AYAT*1..4]->(n)
            WHERE elementId(n) IN $nids
            RETURN path
        """, {"nids": target_node_ids})
        for record in result:
            path = record["path"]
            if not path:
                continue
            # Add all nodes in path
            for node in path.nodes:
                node_id = node.element_id
                if node_id not in graph_nodes:
                    labels = list(node.labels)
                    label = node.get("label") or node.get("name") or node_id
                    graph_nodes[node_id] = {
                        "id": node_id,
                        "labels": labels,
                        "label": label,
                        "source_document_id": node.get("source_document_id", ""),
                    }
            # Add all relationships in path
            for rel in path.relationships:
                src_id = rel.start_node.element_id
                tgt_id = rel.end_node.element_id
                edge_key = f"{src_id}->{tgt_id}"
                if edge_key not in existing_edges:
                    existing_edges.add(edge_key)
                    graph_edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "type": rel.type,
                    })

    return {
        "nodes": list(graph_nodes.values()),
        "edges": graph_edges,
    }

    return {
        "nodes": list(graph_nodes.values()),
        "edges": graph_edges,
    }


@router.post("/qa", response_model=QAResponse)
async def ask_question(request: QARequest):
    """Hybrid QA pipeline: question → Cypher → keyword search → context → answer."""
    import time
    import logging
    logger = logging.getLogger("uvicorn")

    start_total_time = time.perf_counter()
    steps = []
    question = request.question
    doc_ids = request.doc_ids
    mini_graph = {"nodes": [], "edges": []}
    cypher_query = ""
    cypher_results_raw = []

    msg = f"=== QA Pipeline Started for question: '{question}' ==="
    logger.info(msg)
    print(msg, flush=True)

    # Step 1: Understand question
    t1_start = time.perf_counter()
    # Step 1 parsing is instant, but timed for completeness
    t1_end = time.perf_counter()
    duration_step1 = t1_end - t1_start
    msg = f"Step 1: Understand question - took {duration_step1:.4f} seconds"
    logger.info(msg)
    print(msg, flush=True)
    steps.append(QAProcessStep(
        step=1, label="Memahami pertanyaan",
        detail=f'Pertanyaan: "{question}"',
        duration=round(duration_step1, 4),
    ))

    # Step 2: Generate Cypher query via LLM
    t2_start = time.perf_counter()
    cypher_result = await LLMService.generate_cypher(question, doc_ids=doc_ids)
    cypher_query = cypher_result.get("cypher", "")
    t2_end = time.perf_counter()
    duration_step2 = t2_end - t2_start
    msg = f"Step 2: Generate Cypher query - took {duration_step2:.4f} seconds"
    logger.info(msg)
    print(msg, flush=True)

    if cypher_result["status"] == "ok" and cypher_query:
        steps.append(QAProcessStep(
            step=2, label="Generate Cypher query",
            detail=cypher_query,
            status="done",
            data={"type": "cypher", "cypher": cypher_query},
            duration=round(duration_step2, 4),
        ))
    else:
        error_msg = cypher_result.get("error", "Gagal membuat Cypher query")
        steps.append(QAProcessStep(
            step=2, label="Generate Cypher query",
            detail=f"Error: {error_msg}",
            status="error",
            data={"type": "cypher", "cypher": "", "error": error_msg},
            duration=round(duration_step2, 4),
        ))

    # Step 3: Execute Cypher on Neo4j
    t3_start = time.perf_counter()
    if cypher_query:
        try:
            exec_results = Neo4jService.execute_cypher(cypher_query)
            t3_end = time.perf_counter()
            duration_step3 = t3_end - t3_start
            msg = f"Step 3: Execute Cypher on Neo4j - took {duration_step3:.4f} seconds"
            logger.info(msg)
            print(msg, flush=True)
            if exec_results and not any(r.get("error") for r in exec_results):
                cypher_results_raw = exec_results[:300]
                columns = list(cypher_results_raw[0].keys()) if cypher_results_raw else []
                steps.append(QAProcessStep(
                    step=3, label="Eksekusi Cypher di Neo4j",
                    detail=f"{len(cypher_results_raw)} hasil ditemukan",
                    status="done",
                    data={
                        "type": "results",
                        "columns": columns,
                        "rows": cypher_results_raw[:10],
                        "total": len(cypher_results_raw),
                    },
                    duration=round(duration_step3, 4),
                ))
            else:
                error_detail = str(exec_results[0]["error"]) if exec_results and exec_results[0].get("error") else ""
                steps.append(QAProcessStep(
                    step=3, label="Eksekusi Cypher di Neo4j",
                    detail=f"Query error: {error_detail}" if error_detail else "Tidak ada hasil",
                    status="error" if error_detail else "done",
                    data={"type": "results", "columns": [], "rows": [], "total": 0, "error": error_detail},
                    duration=round(duration_step3, 4),
                ))
        except Exception as e:
            t3_end = time.perf_counter()
            duration_step3 = t3_end - t3_start
            msg = f"Step 3: Execute Cypher on Neo4j (Failed) - took {duration_step3:.4f} seconds"
            logger.info(msg)
            print(msg, flush=True)
            steps.append(QAProcessStep(
                step=3, label="Eksekusi Cypher di Neo4j",
                detail=f"Execution error: {str(e)[:200]}",
                status="error",
                data={"type": "results", "columns": [], "rows": [], "total": 0, "error": str(e)[:200]},
                duration=round(duration_step3, 4),
            ))
    else:
        t3_end = time.perf_counter()
        duration_step3 = t3_end - t3_start
        msg = f"Step 3: Execute Cypher on Neo4j (Skipped) - took {duration_step3:.4f} seconds"
        logger.info(msg)
        print(msg, flush=True)
        steps.append(QAProcessStep(
            step=3, label="Eksekusi Cypher di Neo4j",
            detail="Skipped — tidak ada Cypher query",
            status="skipped",
            data={"type": "results", "columns": [], "rows": [], "total": 0},
            duration=round(duration_step3, 4),
        ))

    # Step 4: Search KG by keywords
    t4_start = time.perf_counter()
    search_results = _search_kg_by_keywords(question, doc_ids=doc_ids)
    search_labels = [r.get("label", "?") for r in search_results]
    t4_end = time.perf_counter()
    duration_step4 = t4_end - t4_start
    msg = f"Step 4: Search KG by keywords - took {duration_step4:.4f} seconds"
    logger.info(msg)
    print(msg, flush=True)

    steps.append(QAProcessStep(
        step=4, label="Pencarian Knowledge Graph",
        detail=f"{len(search_results)} node ditemukan: {', '.join(search_labels[:5])}{'...' if len(search_labels) > 5 else ''}",
        status="done" if search_results else "error",
        duration=round(duration_step4, 4),
    ))

    # Step 5: Enrich with relations
    t5_start = time.perf_counter()
    enriched = []
    if search_results:
        enriched, mini_graph = _enrich_with_relations(search_results)
        step5_detail = f"Mengambil detail dan relasi dari {len(enriched)} node -> {len(mini_graph['nodes'])} nodes, {len(mini_graph['edges'])} edges"
        step5_status = "done"
    else:
        step5_detail = "Skipped - tidak ada node ditemukan"
        step5_status = "skipped"

    # Build graph from Cypher results (more accurate) — replaces keyword graph
    if cypher_results_raw:
        cypher_graph = _build_graph_from_cypher(cypher_query, cypher_results_raw)
        if cypher_graph["nodes"]:
            # Cypher results are the actual answer — use them exclusively
            mini_graph = cypher_graph

    # Connect all nodes in the mini graph to their parent Regulasi
    mini_graph = _connect_to_regulasi(mini_graph)

    t5_end = time.perf_counter()
    duration_step5 = t5_end - t5_start
    msg = f"Step 5: Enrich context & build graph - took {duration_step5:.4f} seconds"
    logger.info(msg)
    print(msg, flush=True)

    steps.append(QAProcessStep(
        step=5, label="Memperkaya konteks dengan relasi",
        detail=step5_detail,
        status=step5_status,
        duration=round(duration_step5, 4),
    ))

    # Step 6: Generate response
    t6_start = time.perf_counter()
    # Merge BOTH cypher results and keyword search results into context
    context_parts = []
    if cypher_results_raw:
        context_parts.append("=== Hasil Cypher Query ===")
        context_parts.append(_format_kg_context(cypher_results_raw))
    if enriched:
        context_parts.append("\n=== Hasil Pencarian Keyword ===")
        context_parts.append(_format_kg_context(enriched))
    kg_context_text = "\n".join(context_parts) if context_parts else "(tidak ada data)"

    # Safety net: cap context size to prevent overwhelming the LLM
    MAX_CONTEXT_CHARS = 30000
    if len(kg_context_text) > MAX_CONTEXT_CHARS:
        kg_context_text = kg_context_text[:MAX_CONTEXT_CHARS] + \
            "\n\n[... konteks dipotong karena terlalu panjang. " \
            "Berikan rangkuman/ringkasan dari data di atas saja.]"

    response_result = await LLMService.generate_response(question, kg_context_text)
    answer = response_result.get("answer", "Maaf, terjadi kesalahan saat memproses pertanyaan.")
    t6_end = time.perf_counter()
    duration_step6 = t6_end - t6_start
    msg = f"Step 6: Generate response - took {duration_step6:.4f} seconds"
    logger.info(msg)
    print(msg, flush=True)

    steps.append(QAProcessStep(
        step=6, label="Menyusun jawaban",
        detail="Jawaban berhasil dibuat" if response_result["status"] == "ok" else f"Error: {response_result.get('error', '')}",
        status="done" if response_result["status"] == "ok" else "error",
        duration=round(duration_step6, 4),
    ))

    # Extract references
    references = _extract_references(answer)

    total_duration = time.perf_counter() - start_total_time
    msg = f"=== QA Pipeline Completed in {total_duration:.4f} seconds ==="
    logger.info(msg)
    print(msg, flush=True)

    return QAResponse(
        answer=answer,
        cypher_query=cypher_query,
        kg_context=enriched[:10],
        references=references,
        process_steps=steps,
        graph=mini_graph,
    )

