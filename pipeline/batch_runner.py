"""
Batch Pipeline Runner for Multi-Document KG Processing.

Orchestrates the end-to-end pipeline for processing multiple regulation PDFs
into a unified Knowledge Graph. Supports resume, single-doc, and dry-run modes.

Usage:
    python pipeline/batch_runner.py --all              # Process all 10 docs
    python pipeline/batch_runner.py --doc UU_19_2016    # Process 1 doc
    python pipeline/batch_runner.py --all --resume      # Resume from last step
    python pipeline/batch_runner.py --all --dry-run     # Validate only
    python pipeline/batch_runner.py --load-edges        # Load inter-doc edges only
"""

import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Configuration
# ============================================================

DEFAULT_PATHS = {
    "regulation_list": "data/regulation_list.json",
    "raw_dir": "data/raw",
    "extracted_dir": "data/extracted",
    "parsed_dir": "data/parsed",
    "chunks_dir": "data/chunks",
    "triples_dir": "data/triples",
    "validated_dir": "data/validated",
    "deduped_dir": "data/deduped",
    "embedded_dir": "data/embedded",
    "status_file": "data/pipeline_status.json",
    "logs_dir": "data/logs",
}

# Pipeline steps in order (9 steps per document)
PIPELINE_STEPS = [
    "extract_pdf",         # 1. PDF → extracted JSON
    "parse_structure",     # 2. extracted → parsed (hierarchical components)
    "detect_references",   # 3. regex reference detection (Lex2KG approach)
    "create_chunks",       # 4. parsed → chunks
    "extract_triples",     # 5. chunks → triples (LLM)
    "validate_triples",    # 6. triples → validated
    "deduplicate",         # 7. validated → deduped
    "embed_nodes",         # 8. deduped → embedded (LLM)
    "load_neo4j",          # 9. embedded → Neo4j
]


# ============================================================
# Status Tracker
# ============================================================

class PipelineStatus:
    """Tracks processing status for each document."""

    def __init__(self, status_path: str):
        self.path = status_path
        self.data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_status(self, doc_id: str) -> dict:
        return self.data.get(doc_id, {
            "step": 0,
            "step_name": "pending",
            "status": "pending",
            "last_error": None,
            "started_at": None,
            "completed_at": None,
            "nodes": 0,
            "edges": 0,
        })

    def update(self, doc_id: str, step: int, step_name: str,
               status: str, error: str = None, nodes: int = 0, edges: int = 0):
        self.data[doc_id] = {
            "step": step,
            "step_name": step_name,
            "status": status,
            "last_error": error,
            "updated_at": datetime.now().isoformat(),
            "nodes": nodes,
            "edges": edges,
        }
        self._save()

    def is_step_complete(self, doc_id: str, step: int) -> bool:
        return self.get_status(doc_id).get("step", 0) >= step

    def print_summary(self):
        print("\n" + "=" * 70)
        print("PIPELINE STATUS SUMMARY")
        print("=" * 70)
        for doc_id, info in sorted(self.data.items()):
            status_icon = {
                "complete": "✅",
                "running": "🔄",
                "error": "❌",
                "pending": "⏳",
            }.get(info.get("status"), "❓")
            step = info.get("step", 0)
            step_name = info.get("step_name", "pending")
            nodes = info.get("nodes", 0)
            edges = info.get("edges", 0)
            print(f"  {status_icon} {doc_id:<30} step {step}/9 ({step_name}) "
                  f"nodes={nodes} edges={edges}")
            if info.get("last_error"):
                print(f"     ⚠️  {info['last_error'][:80]}")
        print("=" * 70)


# ============================================================
# Pipeline Step Functions
# ============================================================

def step_extract_pdf(doc_id: str, filename: str, paths: dict, logger: logging.Logger) -> str:
    """Step 1: Extract text from PDF."""
    from pipeline.extract.pdf_extractor import extract_pdf, save_extracted_document, ExtractedDocument

    pdf_path = os.path.join(paths["raw_dir"], filename)
    output_path = os.path.join(paths["extracted_dir"], f"{doc_id}.json")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Extracting PDF: {filename}")
    doc = extract_pdf(pdf_path)

    # Override document_id from regulation_list (more reliable than filename parsing)
    doc = ExtractedDocument(
        document_id=doc_id,
        uu_number=doc.uu_number,
        title=doc.title,
        year=doc.year,
        source_url=doc.source_url,
        total_pages=doc.total_pages,
        pages=doc.pages,
    )

    out = save_extracted_document(doc, paths["extracted_dir"])
    logger.info(f"  → {out} ({doc.total_pages} pages)")
    return out


def step_parse_structure(doc_id: str, paths: dict, logger: logging.Logger) -> str:
    """Step 2: Parse document structure into legal components."""
    from pipeline.extract.structure_parser import parse_document_structure, save_parsed_document

    input_path = os.path.join(paths["extracted_dir"], f"{doc_id}.json")
    with open(input_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    logger.info(f"Parsing structure: {doc_id}")
    components = parse_document_structure(doc)
    out = save_parsed_document(doc_id, components, paths["parsed_dir"])

    type_counts = {}
    for c in components:
        type_counts[c.component_type] = type_counts.get(c.component_type, 0) + 1
    logger.info(f"  → {len(components)} components: {type_counts}")
    return out


def step_detect_references(doc_id: str, paths: dict, logger: logging.Logger) -> str:
    """Step 3: Detect cross-references using regex (Lex2KG approach)."""
    from pipeline.extract.reference_detector import detect_references_in_file

    input_path = os.path.join(paths["parsed_dir"], f"{doc_id}.json")
    regulation_list_path = os.path.join(os.path.dirname(paths["parsed_dir"]), "regulation_list.json")

    logger.info(f"Detecting references: {doc_id}")
    detect_references_in_file(input_path, regulation_list_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ref_info = data.get("reference_summary", {})
    total = ref_info.get("total_references", 0)
    by_type = ref_info.get("by_type", {})
    logger.info(f"  -> {total} references: {by_type}")
    return input_path


def step_create_chunks(doc_id: str, paths: dict, logger: logging.Logger) -> str:
    """Step 4: Split into overlapping chunks."""
    from pipeline.extract.chunker import create_chunks, save_chunks

    input_path = os.path.join(paths["parsed_dir"], f"{doc_id}.json")
    with open(input_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    logger.info(f"Creating chunks: {doc_id}")
    chunks = create_chunks(doc["components"], doc_id)
    out = save_chunks(doc_id, chunks, paths["chunks_dir"])

    if chunks:
        tokens = [c.token_count for c in chunks]
        logger.info(f"  -> {len(chunks)} chunks (tokens: min={min(tokens)}, max={max(tokens)}, avg={sum(tokens)//len(tokens)})")
    return out


def step_extract_triples(doc_id: str, paths: dict, api_key: str,
                         logger: logging.Logger, delay: float = 2.0) -> str:
    """Step 4: LLM-based triple extraction (Gemini API)."""
    from pipeline.transform.llm_extractor import extract_all_triples

    chunks_path = os.path.join(paths["chunks_dir"], f"{doc_id}_chunks.json")
    logger.info(f"Extracting triples (LLM): {doc_id}")

    out = extract_all_triples(
        chunks_path=chunks_path,
        output_dir=paths["triples_dir"],
        api_key=api_key,
        delay_between_calls=delay,
    )

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"  → {data['total_nodes']} nodes, {data['total_edges']} edges")
    return out


def step_validate_triples(doc_id: str, paths: dict, logger: logging.Logger) -> str:
    """Step 5: Validate triples against ontology."""
    from pipeline.transform.validator import validate_triples_file

    input_path = os.path.join(paths["triples_dir"], f"{doc_id}_triples.json")
    logger.info(f"Validating triples: {doc_id}")

    out = validate_triples_file(input_path, paths["validated_dir"])

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    removed_n = data.get("removed_nodes", 0)
    removed_e = data.get("removed_edges", 0)
    logger.info(f"  → valid: {data['total_nodes']} nodes, {data['total_edges']} edges "
                f"(removed: {removed_n} nodes, {removed_e} edges)")
    return out


def step_deduplicate(doc_id: str, paths: dict, logger: logging.Logger) -> str:
    """Step 6: Deduplicate entities."""
    from pipeline.transform.deduplicator import deduplicate_triples_file

    input_path = os.path.join(paths["validated_dir"], f"{doc_id}_triples.json")
    logger.info(f"Deduplicating: {doc_id}")

    out = deduplicate_triples_file(input_path, paths["deduped_dir"])

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = data.get("nodes_merged", 0)
    logger.info(f"  → {data['total_nodes']} nodes, {data['total_edges']} edges "
                f"(merged {merged} duplicates)")
    return out


def step_embed_nodes(doc_id: str, paths: dict, api_key: str, logger: logging.Logger) -> str:
    """Step 7: Generate embeddings (Gemini API)."""
    from pipeline.transform.embedder import embed_triples_file

    input_path = os.path.join(paths["deduped_dir"], f"{doc_id}_triples.json")
    logger.info(f"Generating embeddings: {doc_id}")

    out = embed_triples_file(input_path, paths["embedded_dir"], api_key)
    logger.info(f"  → Embeddings saved to {out}")
    return out


def step_load_neo4j(doc_id: str, paths: dict, neo4j_config: dict,
                    logger: logging.Logger) -> tuple[int, int]:
    """Step 8: Load into Neo4j (incremental, no clear)."""
    from pipeline.load.neo4j_loader import load_from_file

    input_path = os.path.join(paths["embedded_dir"], f"{doc_id}_triples.json")
    logger.info(f"Loading to Neo4j: {doc_id}")

    stats = load_from_file(
        input_path=input_path,
        neo4j_uri=neo4j_config["uri"],
        neo4j_user=neo4j_config["user"],
        neo4j_password=neo4j_config["password"],
        clear_first=False,  # IMPORTANT: incremental append
    )

    nodes = stats.get("total_nodes", 0)
    edges = stats.get("total_edges", 0)
    logger.info(f"  → Neo4j total: {nodes} nodes, {edges} edges")
    return nodes, edges


# ============================================================
# Inter-Document Edge Loading
# ============================================================

def load_inter_document_edges(regulation_list_path: str, neo4j_config: dict,
                              logger: logging.Logger):
    """Load cross-document relationships from regulation_list.json into Neo4j.

    Creates Peraturan nodes for each document and links them with:
    - MENGAMANDEMEN, DIAMANDEMEN_OLEH
    - DITURUNKAN_KE, DITURUNKAN_DARI
    - MENCABUT, DICABUT_OLEH
    - MERUJUK
    """
    from neo4j import GraphDatabase

    with open(regulation_list_path, "r", encoding="utf-8") as f:
        regulations = json.load(f)

    driver = GraphDatabase.driver(
        neo4j_config["uri"],
        auth=(neo4j_config["user"], neo4j_config["password"])
    )
    driver.verify_connectivity()
    logger.info("Connected to Neo4j for inter-document edge loading")

    with driver.session() as session:
        # Step 1: Create/update Peraturan nodes for each document
        for reg in regulations:
            cypher = """
                MERGE (p:Entity:Peraturan {id: $doc_id})
                SET p.label = $title,
                    p.short_name = $short_name,
                    p.regulation_type = $reg_type,
                    p.number = $number,
                    p.year = $year,
                    p.status = $status,
                    p.node_type = 'Peraturan',
                    p.source_document_id = $doc_id,
                    p.created_at = datetime()
            """
            session.run(cypher,
                doc_id=reg["doc_id"],
                title=reg["title"],
                short_name=reg["short_name"],
                reg_type=reg["type"],
                number=reg["number"],
                year=reg["year"],
                status=reg["status"],
            )
            logger.info(f"  Created/updated Peraturan node: {reg['doc_id']}")

        # Step 2: Create inter-document edges from relations
        edge_count = 0
        for reg in regulations:
            for rel in reg.get("relations", []):
                target_id = rel["target_doc_id"]
                rel_type = rel["type"]
                description = rel.get("description", "")

                cypher = f"""
                    MATCH (a:Peraturan {{id: $source_id}})
                    MATCH (b:Peraturan {{id: $target_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.description = $description,
                        r.source = 'regulation_list.json',
                        r.created_at = datetime()
                """
                session.run(cypher,
                    source_id=reg["doc_id"],
                    target_id=target_id,
                    description=description,
                )
                edge_count += 1
                logger.info(f"  Edge: {reg['doc_id']} -[{rel_type}]-> {target_id}")

        # Step 3: Create amendment-level edges if amended_articles exist
        for reg in regulations:
            for art in reg.get("amended_articles", []):
                article = art["article"]
                action = art["action"]  # MENGUBAH, MENYISIPKAN, MENGHAPUS
                desc = art.get("description", "")

                # Link the amending doc to the specific article concept in target doc
                # The article node may have been created by LLM extraction
                cypher = f"""
                    MATCH (amender:Peraturan {{id: $amender_id}})
                    OPTIONAL MATCH (target_article:Entity)
                        WHERE target_article.label CONTAINS $article_label
                        AND target_article.source_document_id IN $target_doc_ids
                    WITH amender, target_article
                    WHERE target_article IS NOT NULL
                    MERGE (amender)-[r:{action}]->(target_article)
                    SET r.description = $description,
                        r.source = 'regulation_list.json',
                        r.created_at = datetime()
                """
                # The target is the original doc that's being amended
                target_doc_ids = [
                    r["target_doc_id"]
                    for r in reg.get("relations", [])
                    if r["type"] == "MENGAMANDEMEN"
                ]

                if target_doc_ids:
                    session.run(cypher,
                        amender_id=reg["doc_id"],
                        article_label=article,
                        target_doc_ids=target_doc_ids,
                        description=desc,
                    )
                    edge_count += 1

        logger.info(f"\nTotal inter-document edges created: {edge_count}")

    driver.close()

    # Step 4: Create VersiPasal nodes (Lex2KG concept)
    logger.info("\nCreating VersiPasal nodes...")
    from pipeline.load.neo4j_loader import Neo4jLoader
    loader = Neo4jLoader(neo4j_config["uri"], neo4j_config["user"], neo4j_config["password"])
    try:
        loader.load_versi_pasal(regulation_list_path)
    finally:
        loader.close()

    # Step 5: Load regex-detected references
    logger.info("\nLoading regex-detected references...")
    parsed_dir = os.path.join(os.path.dirname(regulation_list_path), "parsed")
    loader = Neo4jLoader(neo4j_config["uri"], neo4j_config["user"], neo4j_config["password"])
    try:
        total_regex_refs = 0
        for json_file in Path(parsed_dir).glob("*.json"):
            count = loader.load_regex_references(str(json_file))
            logger.info(f"  {json_file.stem}: {count} references loaded")
            total_regex_refs += count
        logger.info(f"Total regex references loaded: {total_regex_refs}")
    finally:
        loader.close()

# ============================================================
# Output file existence check (for resume)
# ============================================================

def _output_exists(doc_id: str, step_index: int, paths: dict) -> bool:
    """Check if the output file for a given step already exists."""
    output_map = {
        0: os.path.join(paths["extracted_dir"], f"{doc_id}.json"),
        1: os.path.join(paths["parsed_dir"], f"{doc_id}.json"),
        # Step 2 (detect_references) updates parsed file in-place — check for reference_summary
        2: None,  # special handling below
        3: os.path.join(paths["chunks_dir"], f"{doc_id}_chunks.json"),
        4: os.path.join(paths["triples_dir"], f"{doc_id}_triples.json"),
        5: os.path.join(paths["validated_dir"], f"{doc_id}_triples.json"),
        6: os.path.join(paths["deduped_dir"], f"{doc_id}_triples.json"),
        7: os.path.join(paths["embedded_dir"], f"{doc_id}_triples.json"),
        # Step 8 (Neo4j) has no file output — check status tracker
    }
    
    # Special case: detect_references updates parsed file in-place
    if step_index == 2:
        parsed_path = os.path.join(paths["parsed_dir"], f"{doc_id}.json")
        if os.path.exists(parsed_path):
            try:
                with open(parsed_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return "reference_summary" in data
            except Exception:
                return False
        return False
    
    path = output_map.get(step_index)
    if path is None:
        return False
    return os.path.exists(path)


# ============================================================
# Main Pipeline Runner
# ============================================================

def setup_logger(doc_id: str, logs_dir: str) -> logging.Logger:
    """Create a per-document logger."""
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(f"pipeline.{doc_id}")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(
        os.path.join(logs_dir, f"{doc_id}.log"),
        mode="a", encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(f"[{doc_id}] %(message)s"))
    logger.addHandler(ch)

    return logger


def process_document(
    doc_id: str,
    filename: str,
    paths: dict,
    status: PipelineStatus,
    api_key: str,
    neo4j_config: dict,
    resume: bool = False,
    dry_run: bool = False,
    api_delay: float = 2.0,
):
    """Process a single document through the full pipeline."""
    logger = setup_logger(doc_id, paths["logs_dir"])
    logger.info(f"{'='*60}")
    logger.info(f"Starting pipeline for: {doc_id} ({filename})")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'FULL'} | Resume: {resume}")
    logger.info(f"{'='*60}")

    steps = [
        ("extract_pdf",        lambda: step_extract_pdf(doc_id, filename, paths, logger)),
        ("parse_structure",    lambda: step_parse_structure(doc_id, paths, logger)),
        ("detect_references",  lambda: step_detect_references(doc_id, paths, logger)),
        ("create_chunks",      lambda: step_create_chunks(doc_id, paths, logger)),
        ("extract_triples",    lambda: step_extract_triples(doc_id, paths, api_key, logger, api_delay)),
        ("validate_triples",   lambda: step_validate_triples(doc_id, paths, logger)),
        ("deduplicate",        lambda: step_deduplicate(doc_id, paths, logger)),
        ("embed_nodes",        lambda: step_embed_nodes(doc_id, paths, api_key, logger)),
        ("load_neo4j",         lambda: step_load_neo4j(doc_id, paths, neo4j_config, logger)),
    ]

    for step_idx, (step_name, step_fn) in enumerate(steps):
        step_num = step_idx + 1

        # Resume: skip completed steps
        if resume and _output_exists(doc_id, step_idx, paths):
            logger.info(f"[Step {step_num}/9] {step_name} -- SKIPPED (output exists)")
            continue

        # Dry run: only validate steps 1-4 (no API calls)
        if dry_run and step_num > 4:
            logger.info(f"[Step {step_num}/9] {step_name} -- SKIPPED (dry-run)")
            continue

        logger.info(f"\n[Step {step_num}/9] {step_name}")
        status.update(doc_id, step_num, step_name, "running")

        try:
            result = step_fn()

            # Extract node/edge counts from Neo4j step
            if step_name == "load_neo4j" and isinstance(result, tuple):
                nodes, edges = result
            else:
                nodes = status.get_status(doc_id).get("nodes", 0)
                edges = status.get_status(doc_id).get("edges", 0)
                # Try to read counts from output file
                if step_name in ("extract_triples", "validate_triples", "deduplicate"):
                    try:
                        out_dir = {
                            "extract_triples": paths["triples_dir"],
                            "validate_triples": paths["validated_dir"],
                            "deduplicate": paths["deduped_dir"],
                        }[step_name]
                        out_file = os.path.join(out_dir, f"{doc_id}_triples.json")
                        with open(out_file, "r", encoding="utf-8") as f:
                            out_data = json.load(f)
                        nodes = out_data.get("total_nodes", nodes)
                        edges = out_data.get("total_edges", edges)
                    except Exception:
                        pass

            status.update(doc_id, step_num, step_name, "complete" if step_num == 9 else "running",
                         nodes=nodes, edges=edges)
            logger.info(f"  ✅ {step_name} complete")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"  ❌ {step_name} FAILED: {error_msg}", exc_info=True)
            status.update(doc_id, step_num, step_name, "error", error=error_msg)
            raise  # Re-raise to stop this document (continue to next in batch)

    if not dry_run:
        status.update(doc_id, 9, "load_neo4j", "complete",
                     nodes=status.get_status(doc_id).get("nodes", 0),
                     edges=status.get_status(doc_id).get("edges", 0))

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline {'DRY RUN' if dry_run else 'COMPLETE'} for {doc_id}")
    logger.info(f"{'='*60}\n")


def run_batch(
    doc_ids: list[str] = None,
    resume: bool = False,
    dry_run: bool = False,
    load_edges_only: bool = False,
    api_delay: float = 2.0,
):
    """Run the pipeline for multiple documents.

    Args:
        doc_ids: List of doc IDs to process (None = all)
        resume: Skip steps with existing output
        dry_run: Only run non-API steps (1-3)
        load_edges_only: Only load inter-document edges
        api_delay: Seconds between API calls
    """
    from dotenv import load_dotenv
    load_dotenv()

    # Resolve paths relative to project root
    paths = {k: str(PROJECT_ROOT / v) for k, v in DEFAULT_PATHS.items()}

    # Load config
    api_key = os.environ.get("GEMINI_API_KEY", "")
    neo4j_config = {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "password"),
    }

    # Load regulation list
    with open(paths["regulation_list"], "r", encoding="utf-8") as f:
        regulations = json.load(f)

    reg_map = {r["doc_id"]: r for r in regulations}

    # Filter to requested docs
    if doc_ids:
        regulations = [r for r in regulations if r["doc_id"] in doc_ids]
        missing = set(doc_ids) - set(r["doc_id"] for r in regulations)
        if missing:
            print(f"⚠️  Unknown doc_ids: {missing}")
            print(f"   Available: {list(reg_map.keys())}")
            return

    # Status tracker
    status = PipelineStatus(paths["status_file"])

    # Load inter-doc edges only mode
    if load_edges_only:
        logger = setup_logger("inter_doc", paths["logs_dir"])
        print("\n🔗 Loading inter-document edges from regulation_list.json...")
        load_inter_document_edges(paths["regulation_list"], neo4j_config, logger)
        print("✅ Inter-document edges loaded!")
        return

    # Pre-flight checks
    print(f"\n📋 Pipeline Batch Runner")
    print(f"   Documents: {len(regulations)}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'FULL'}")
    print(f"   Resume: {resume}")
    print(f"   API delay: {api_delay}s")

    if not dry_run and not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in environment.")
        print("   Set it in .env or export GEMINI_API_KEY=...")
        return

    # Ensure output directories exist
    for key in ["extracted_dir", "parsed_dir", "chunks_dir", "triples_dir",
                "validated_dir", "deduped_dir", "embedded_dir", "logs_dir"]:
        os.makedirs(paths[key], exist_ok=True)

    # Process each document
    success = 0
    failed = 0
    skipped = 0

    for i, reg in enumerate(regulations):
        doc_id = reg["doc_id"]
        filename = reg["filename"]

        print(f"\n{'─'*60}")
        print(f"📄 [{i+1}/{len(regulations)}] {doc_id} ({filename})")
        print(f"{'─'*60}")

        # Check if already complete
        if resume and status.is_step_complete(doc_id, 9):
            print(f"  ⏭️  Already complete, skipping")
            skipped += 1
            continue

        try:
            process_document(
                doc_id=doc_id,
                filename=filename,
                paths=paths,
                status=status,
                api_key=api_key,
                neo4j_config=neo4j_config,
                resume=resume,
                dry_run=dry_run,
                api_delay=api_delay,
            )
            success += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
            # Continue to next document (error isolation)
            continue

    # Post-pipeline: load inter-document edges
    if not dry_run and success > 0:
        print(f"\n🔗 Loading inter-document edges...")
        logger = setup_logger("inter_doc", paths["logs_dir"])
        try:
            load_inter_document_edges(paths["regulation_list"], neo4j_config, logger)
            print("✅ Inter-document edges loaded!")
        except Exception as e:
            print(f"❌ Inter-document edge loading failed: {e}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"BATCH {'DRY RUN ' if dry_run else ''}COMPLETE")
    print(f"  ✅ Success: {success}")
    print(f"  ❌ Failed:  {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"{'='*60}")

    status.print_summary()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Pipeline Runner for Multi-Document Legal KG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/batch_runner.py --all              # Process all documents
  python pipeline/batch_runner.py --doc UU_19_2016    # Process single document
  python pipeline/batch_runner.py --all --resume      # Resume interrupted run
  python pipeline/batch_runner.py --all --dry-run     # Validate PDFs only
  python pipeline/batch_runner.py --load-edges        # Load inter-doc edges only
        """
    )

    # Document selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process all documents")
    group.add_argument("--doc", type=str, nargs="+", help="Process specific document(s) by doc_id")
    group.add_argument("--load-edges", action="store_true", help="Only load inter-document edges")

    # Mode options
    parser.add_argument("--resume", action="store_true",
                       help="Skip steps with existing output files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only run non-API steps (extract, parse, chunk)")
    parser.add_argument("--api-delay", type=float, default=2.0,
                       help="Seconds between API calls (default: 2.0)")

    args = parser.parse_args()

    doc_ids = args.doc if args.doc else None

    run_batch(
        doc_ids=doc_ids,
        resume=args.resume,
        dry_run=args.dry_run,
        load_edges_only=args.load_edges,
        api_delay=args.api_delay,
    )


if __name__ == "__main__":
    main()
