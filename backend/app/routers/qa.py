"""QA API router — hybrid pipeline: question → keyword search → context → answer."""

import re
from fastapi import APIRouter
from app.models.schemas import QARequest, QAResponse, QAProcessStep
from app.services.neo4j_service import Neo4jService
from app.services.llm_service import LLMService

router = APIRouter()


def _format_kg_context(records: list[dict]) -> str:
    """Format Neo4j results into readable text."""
    if not records:
        return "(tidak ada hasil)"
    lines = []
    for i, r in enumerate(records[:20], 1):
        parts = []
        for k, v in r.items():
            if v is not None and k != "error":
                val = str(v)[:300]
                parts.append(f"{k}: {val}")
        lines.append(f"{i}. " + " | ".join(parts))
    return "\n".join(lines)


def _extract_references(text: str) -> list[str]:
    """Extract legal references from answer text."""
    refs = set()
    for m in re.finditer(r"Pasal\s+\d+(?:\s+ayat\s+\(\d+\))?", text):
        refs.add(m.group())
    for m in re.finditer(r"UU\s+(?:No\.\s*)?\d+(?:\s+Tahun\s+\d+)?", text):
        refs.add(m.group())
    return sorted(refs)


def _search_kg_by_keywords(question: str) -> list[dict]:
    """Search KG using multiple keyword strategies to find relevant nodes."""
    all_results = {}

    # Strategy 1: Extract key phrases and search
    # Remove common question words
    cleaned = re.sub(
        r'\b(apa|siapa|bagaimana|mengapa|dimana|kapan|berapa|yang|di|dan|atau|itu|ini|adalah|menurut|dalam|untuk|dari|ke|pada|dengan|oleh|tentang|saja)\b',
        ' ', question.lower()
    )
    keywords = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]

    # Search each keyword
    for kw in keywords[:5]:
        results = Neo4jService.search(kw, mode="keyword", limit=5)
        for r in results:
            if r.get("id") and r["id"] not in all_results:
                all_results[r["id"]] = r

    # Strategy 2: Search for pasal references (e.g. "Pasal 27")
    pasal_matches = re.findall(r'[Pp]asal\s+\d+', question)
    for pm in pasal_matches:
        results = Neo4jService.search(pm, mode="keyword", limit=3)
        for r in results:
            if r.get("id") and r["id"] not in all_results:
                all_results[r["id"]] = r

    # Strategy 3: Search full question if few results
    if len(all_results) < 3:
        # Try 2-word phrases from the question
        words = [w for w in question.split() if len(w) > 2]
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            results = Neo4jService.search(phrase, mode="keyword", limit=3)
            for r in results:
                if r.get("id") and r["id"] not in all_results:
                    all_results[r["id"]] = r
            if len(all_results) >= 5:
                break

    return list(all_results.values())[:10]


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
            }

    for node in nodes[:5]:  # Limit to avoid too many queries
        node_id = node.get("id", "")
        if not node_id:
            continue

        detail = Neo4jService.get_node(node_id)
        if not detail:
            enriched.append(node)
            continue

        entry = {
            "label": node.get("label", ""),
            "type": ", ".join(detail.get("labels", [])),
            "content": str(detail.get("properties", {}).get("content", ""))[:500],
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
            for r in in_rels[:8]:
                rel_strs.append(f"{r.get('source_label', '')} → {r.get('type', '')}")
                sid = r.get("source_id", "")
                if sid:
                    if sid not in graph_nodes:
                        graph_nodes[sid] = {
                            "id": sid,
                            "labels": r.get("source_type", []),
                            "label": r.get("source_label", ""),
                        }
                    graph_edges.append({
                        "source": sid,
                        "target": node_id,
                        "type": r.get("type", ""),
                    })

        enriched.append(entry)

    graph = {
        "nodes": list(graph_nodes.values()),
        "edges": graph_edges,
    }
    return enriched, graph


@router.post("/qa", response_model=QAResponse)
async def ask_question(request: QARequest):
    """Hybrid QA pipeline: question → Cypher → keyword search → context → answer."""
    steps = []
    question = request.question
    mini_graph = {"nodes": [], "edges": []}
    cypher_query = ""
    cypher_results_raw = []

    # Step 1: Understand question
    steps.append(QAProcessStep(
        step=1, label="Memahami pertanyaan",
        detail=f'Pertanyaan: "{question}"',
    ))

    # Step 2: Generate Cypher query via LLM
    cypher_result = await LLMService.generate_cypher(question)
    cypher_query = cypher_result.get("cypher", "")

    if cypher_result["status"] == "ok" and cypher_query:
        steps.append(QAProcessStep(
            step=2, label="Generate Cypher query",
            detail=cypher_query,
            status="done",
            data={"type": "cypher", "cypher": cypher_query},
        ))
    else:
        error_msg = cypher_result.get("error", "Gagal membuat Cypher query")
        steps.append(QAProcessStep(
            step=2, label="Generate Cypher query",
            detail=f"Error: {error_msg}",
            status="error",
            data={"type": "cypher", "cypher": "", "error": error_msg},
        ))

    # Step 3: Execute Cypher on Neo4j
    if cypher_query:
        try:
            exec_results = Neo4jService.execute_cypher(cypher_query)
            if exec_results and not any(r.get("error") for r in exec_results):
                cypher_results_raw = exec_results[:20]
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
                ))
            else:
                error_detail = str(exec_results[0]["error"]) if exec_results and exec_results[0].get("error") else ""
                steps.append(QAProcessStep(
                    step=3, label="Eksekusi Cypher di Neo4j",
                    detail=f"Query error: {error_detail}" if error_detail else "Tidak ada hasil",
                    status="error" if error_detail else "done",
                    data={"type": "results", "columns": [], "rows": [], "total": 0, "error": error_detail},
                ))
        except Exception as e:
            steps.append(QAProcessStep(
                step=3, label="Eksekusi Cypher di Neo4j",
                detail=f"Execution error: {str(e)[:200]}",
                status="error",
                data={"type": "results", "columns": [], "rows": [], "total": 0, "error": str(e)[:200]},
            ))
    else:
        steps.append(QAProcessStep(
            step=3, label="Eksekusi Cypher di Neo4j",
            detail="Skipped — tidak ada Cypher query",
            status="skipped",
            data={"type": "results", "columns": [], "rows": [], "total": 0},
        ))

    # Step 4: Search KG by keywords (existing flow, unchanged)
    search_results = _search_kg_by_keywords(question)
    search_labels = [r.get("label", "?") for r in search_results]

    steps.append(QAProcessStep(
        step=4, label="Pencarian Knowledge Graph",
        detail=f"{len(search_results)} node ditemukan: {', '.join(search_labels[:5])}{'...' if len(search_labels) > 5 else ''}",
        status="done" if search_results else "error",
    ))

    # Step 5: Enrich with relations
    enriched = []
    if search_results:
        enriched, mini_graph = _enrich_with_relations(search_results)
        steps.append(QAProcessStep(
            step=5, label="Memperkaya konteks dengan relasi",
            detail=f"Mengambil detail dan relasi dari {len(enriched)} node → {len(mini_graph['nodes'])} nodes, {len(mini_graph['edges'])} edges",
            status="done",
        ))
    else:
        steps.append(QAProcessStep(
            step=5, label="Memperkaya konteks dengan relasi",
            detail="Skipped — tidak ada node ditemukan",
            status="skipped",
        ))

    # Step 6: Generate response
    # Merge BOTH cypher results and keyword search results into context
    context_parts = []
    if cypher_results_raw:
        context_parts.append("=== Hasil Cypher Query ===")
        context_parts.append(_format_kg_context(cypher_results_raw))
    if enriched:
        context_parts.append("\n=== Hasil Pencarian Keyword ===")
        context_parts.append(_format_kg_context(enriched))
    kg_context_text = "\n".join(context_parts) if context_parts else "(tidak ada data)"
    response_result = await LLMService.generate_response(question, kg_context_text)
    answer = response_result.get("answer", "Maaf, terjadi kesalahan saat memproses pertanyaan.")

    steps.append(QAProcessStep(
        step=6, label="Menyusun jawaban",
        detail="Jawaban berhasil dibuat" if response_result["status"] == "ok" else f"Error: {response_result.get('error', '')}",
        status="done" if response_result["status"] == "ok" else "error",
    ))

    # Extract references
    references = _extract_references(answer)

    return QAResponse(
        answer=answer,
        cypher_query=cypher_query,
        kg_context=enriched[:10],
        references=references,
        process_steps=steps,
        graph=mini_graph,
    )

