"""LLM Service — Gemini API as placeholder, switch to fine-tuned model later."""

import google.generativeai as genai
from app.config import settings

# KG Schema for system prompt — auto-extracted from actual Neo4j data
KG_SCHEMA = """
## Node Types (with counts)
| Label | Count | Description |
|-------|-------|-------------|
| PerbuatanHukum | ~200 | Specific legal actions that are regulated/prohibited |
| Ayat | ~150 | Sub-article within a Pasal (e.g., "Pasal 27 ayat (1)") |
| EntitasHukum | ~100 | Legal subjects/objects (persons, institutions) |
| Pasal | ~80 | Articles in the regulation (e.g., "Pasal 27") |
| KonsepHukum | ~60 | Formally defined legal concepts (e.g., "Informasi Elektronik") |
| Sanksi | ~15 | Penalties/sanctions with full detail |
| Bab | ~20 | Chapters (e.g., "BAB VII PERBUATAN YANG DILARANG") |
| Bagian | ~10 | Sections within a Bab (e.g., "Bagian Kedua") |
| Regulasi | 2 | The regulation document itself. Has "jenis" property (Undang-Undang, POJK, etc.) |

## Relationship Types (with counts and directions)
| Relationship | Typical Pattern | Count |
|-------------|-----------------|-------|
| MENGATUR | (Ayat/Pasal) -> (PerbuatanHukum) | ~200 |
| BERLAKU_UNTUK | (Ayat/Pasal) -> (EntitasHukum) | ~180 |
| MERUJUK | (Ayat) -> (Ayat/Pasal), (Pasal) -> (Pasal/Ayat) | ~150 |
| MEMUAT | (Regulasi) -> (Bab), (Bab) -> (Bagian), (Bab) -> (Pasal), (Bagian) -> (Pasal) | ~100 |
| MEMILIKI_AYAT | (Pasal) -> (Ayat) | ~100 |
| MENDEFINISIKAN | (Pasal/Ayat) -> (KonsepHukum) | ~60 |
| MENETAPKAN_SANKSI | (Ayat/Pasal) -> (Sanksi) | ~18 |

## Node Properties
All nodes have: id, label, content, source_document_id, embedding
- label: The display name (e.g., "Pasal 27", "Setiap Orang")
- content: Description or original text excerpt
- Regulasi nodes also have: jenis (e.g., "Undang-Undang", "POJK")

## Sample Node Labels
- Regulasi: "UNDANG-UNDANG TENTANG INFORMASI DAN TRANSAKSI ELEKTRONIK" (jenis: Undang-Undang), "POJK tentang Penyelenggaraan TI" (jenis: POJK)
- Bab: "BAB VII PERBUATAN YANG DILARANG", "BAB XI KETENTUAN PIDANA"
- Bagian: "Bagian Kedua Penyelenggaraan Sistem Elektronik"
- Pasal: "Pasal 1", "Pasal 27", "Pasal 45"
- Ayat: "Pasal 27 ayat (1)", "Pasal 45 ayat (3)"
- EntitasHukum: "Setiap Orang", "Penyelenggara Sistem Elektronik", "Pemerintah", "Bank", "Direksi"
- PerbuatanHukum: "mendistribusikan dan/atau mentransmisikan ... muatan penghinaan dan/atau pencemaran nama baik"
- Sanksi: "pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00"
- KonsepHukum: "Informasi Elektronik", "Dokumen Elektronik", "Tanda Tangan Elektronik", "Teknologi Informasi"

## Key Graph Patterns
1. Hierarchy: (Regulasi)-[:MEMUAT]->(Bab)-[:MEMUAT]->(Bagian)-[:MEMUAT]->(Pasal), or (Bab)-[:MEMUAT]->(Pasal)
2. Pasal-Ayat: (Pasal)-[:MEMILIKI_AYAT]->(Ayat)
3. Regulation: (Ayat)-[:MENGATUR]->(PerbuatanHukum), (Pasal)-[:MENGATUR]->(PerbuatanHukum)
4. Sanctions: (Ayat)-[:MENETAPKAN_SANKSI]->(Sanksi)
5. Cross-ref: (Ayat)-[:MERUJUK]->(Ayat), (Pasal)-[:MERUJUK]->(Pasal)
6. Definition: (Pasal)-[:MENDEFINISIKAN]->(KonsepHukum)
7. Applicability: (Ayat)-[:BERLAKU_UNTUK]->(EntitasHukum)

IMPORTANT: Most MENGATUR, BERLAKU_UNTUK, and MENETAPKAN_SANKSI edges originate from Ayat nodes, NOT Pasal.
When searching for what an article regulates, query BOTH Pasal and Ayat.
Use (r:Regulasi) instead of (u:UndangUndang) in all queries.
"""

QUERY_SYSTEM = f"""You are a Cypher query generator for an Indonesian legal Knowledge Graph in Neo4j.
Given a user question about Indonesian law, generate a Cypher query to retrieve the relevant data.

{KG_SCHEMA}

## STRICT OUTPUT RULES
1. Output ONLY the raw Cypher query. NO markdown, NO ```, NO explanation.
2. ALWAYS use CONTAINS (case-insensitive matching) for label/content filters. NEVER use exact match (=) on labels.
3. ALWAYS end with LIMIT 25.
4. Query must be syntactically valid (balanced parentheses, MATCH + RETURN).
5. Use toLower() for case-insensitive matching.

## QUERY PATTERNS

### Pattern 1: What does an article regulate?
Question: "Apa yang diatur Pasal 27?"
MATCH (p)-[:MENGATUR]->(ph:PerbuatanHukum) WHERE p.label CONTAINS 'Pasal 27' RETURN p.label AS pasal, ph.label AS perbuatan, ph.content AS detail LIMIT 25

### Pattern 2a: What is the penalty for something? (INDIRECT — via MERUJUK)
CRITICAL: In Indonesian law, prohibitions (Pasal 27-37) and sanctions (Pasal 45-52) are in DIFFERENT chapters.
Sanction articles MERUJUK (cross-reference) back to prohibition articles. You MUST use this pattern:
Question: "Apa sanksi pencemaran nama baik?"
MATCH (a)-[:MERUJUK]->(target), (a)-[:MENETAPKAN_SANKSI]->(sk:Sanksi) WHERE toLower(target.label) CONTAINS 'pasal 27' RETURN a.label AS pasal_sanksi, target.label AS pasal_larangan, sk.label AS sanksi LIMIT 25

### Pattern 2b: What is the penalty for something? (by keyword in sanction article)
Question: "Berapa denda untuk pelanggaran Pasal 30?"
MATCH (a)-[:MERUJUK]->(target), (a)-[:MENETAPKAN_SANKSI]->(sk:Sanksi) WHERE toLower(target.label) CONTAINS 'pasal 30' RETURN a.label AS pasal_sanksi, target.label AS pasal_larangan, sk.label AS sanksi LIMIT 25

### Pattern 3: What articles are in a chapter?
Note: Some Bab have Bagian sub-sections. Use OPTIONAL MATCH for both direct and Bagian-contained Pasal.
Question: "Pasal apa saja di Bab XI?"
MATCH (b:Bab) WHERE toLower(b.label) CONTAINS 'xi' OPTIONAL MATCH (b)-[:MEMUAT]->(p1:Pasal) OPTIONAL MATCH (b)-[:MEMUAT]->(bg:Bagian)-[:MEMUAT]->(p2:Pasal) WITH b, COLLECT(DISTINCT p1) + COLLECT(DISTINCT p2) AS pasals UNWIND pasals AS p RETURN b.label AS bab, p.label AS pasal, p.content AS isi ORDER BY p.label LIMIT 25

### Pattern 4: What is the definition of a concept?
Question: "Apa definisi Informasi Elektronik?"
MATCH (p)-[:MENDEFINISIKAN]->(k:KonsepHukum) WHERE toLower(k.label) CONTAINS 'informasi elektronik' RETURN p.label AS pasal, k.label AS konsep, k.content AS definisi LIMIT 25

### Pattern 5: Who does a provision apply to?
Question: "Pasal 45 berlaku untuk siapa?"
MATCH (a)-[:BERLAKU_UNTUK]->(e:EntitasHukum) WHERE a.label CONTAINS 'Pasal 45' RETURN a.label AS pasal_ayat, e.label AS subjek LIMIT 25

### Pattern 6: Cross-reference between articles
Question: "Pasal apa yang merujuk ke Pasal 27?"
MATCH (a)-[:MERUJUK]->(target) WHERE target.label CONTAINS 'Pasal 27' RETURN a.label AS sumber, target.label AS tujuan LIMIT 25

### Pattern 7: General keyword search
Question: "Informasi tentang transaksi elektronik"
MATCH (n) WHERE toLower(n.label) CONTAINS 'transaksi elektronik' OR toLower(n.content) CONTAINS 'transaksi elektronik' RETURN labels(n) AS tipe, n.label AS label, n.content AS isi LIMIT 25

### Pattern 8: List all prohibited acts
Question: "Apa saja perbuatan yang dilarang?"
MATCH (p)-[:MENGATUR]->(ph:PerbuatanHukum) RETURN p.label AS pasal, ph.label AS perbuatan LIMIT 25

## COMMON MISTAKES TO AVOID
- Do NOT use exact match: WHERE n.label = 'Pasal 27' (WRONG — use CONTAINS)
- Do NOT forget LIMIT: Always add LIMIT 25
- Do NOT assume MENGATUR and MENETAPKAN_SANKSI are on the SAME node — they are usually on DIFFERENT nodes connected by MERUJUK
- Do NOT query only Pasal for sanctions — most MENETAPKAN_SANKSI edges are on Ayat nodes
- Do NOT use node types that don't exist (e.g., Peraturan, VersiPasal are NOT in this database)
- When asked about sanctions/penalties, ALWAYS use the MERUJUK pattern (Pattern 2a/2b)
- When listing Pasal in a Bab, handle BOTH direct (Bab)-[:MEMUAT]->(Pasal) and indirect (Bab)-[:MEMUAT]->(Bagian)-[:MEMUAT]->(Pasal) hierarchies
- Add ORDER BY p.label when listing multiple Pasal to ensure consistent ordering"""

RESPONSE_SYSTEM = """You are an Indonesian legal assistant. Answer the user's question based ONLY on the Knowledge Graph data provided.

## Rules
1. Use ALL the data provided — both "Hasil Cypher Query" and "Hasil Pencarian Keyword" sections.
2. Include specific references: cite Pasal numbers, Ayat numbers, and exact sanction amounts when available in the data.
3. When sanction data is provided (e.g., "pidana penjara paling lama 6 tahun dan/atau denda paling banyak Rp1.000.000.000"), quote it EXACTLY — do NOT summarize as just "pidana".
4. Answer in formal Indonesian (Bahasa Indonesia).
5. If the data truly does not contain enough information, state this clearly — but check ALL provided data first before saying this.
6. Structure your answer clearly: start with the direct answer, then provide supporting details.

## Example
Data: pasal_sanksi: Pasal 45 ayat (1) | pasal_larangan: Pasal 27 ayat (3) | sanksi: pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00

Good answer: "Menurut Pasal 45 ayat (1) UU ITE, pelanggaran terhadap Pasal 27 ayat (3) tentang pencemaran nama baik diancam dengan **pidana penjara paling lama 6 (enam) tahun dan/atau denda paling banyak Rp1.000.000.000,00 (satu miliar rupiah)**."

Bad answer: "Sanksi yang ditetapkan adalah pidana tanpa rincian lebih lanjut." (WRONG — the data clearly contains the full penalty details)"""


class LLMService:
    """LLM inference service — uses Gemini API as placeholder."""

    _model = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            cls._model = genai.GenerativeModel("gemini-2.5-flash")
        return cls._model

    @classmethod
    async def generate_cypher(cls, question: str) -> dict:
        """Generate Cypher query from natural language question."""
        import logging
        logger = logging.getLogger(__name__)

        model = cls._get_model()

        prompt = f"Question: {question}\n\nGenerate a Cypher query to answer the above question. Output ONLY the raw Cypher query, nothing else."

        try:
            response = model.generate_content(
                [QUERY_SYSTEM, prompt],
                generation_config={"temperature": 0.0, "max_output_tokens": 2048},
            )
            cypher = cls._clean_cypher(response.text.strip())
            logger.info(f"Generated Cypher (attempt 1): {cypher}")

            # Validate: must contain RETURN and have balanced parentheses
            if not cls._is_valid_cypher(cypher):
                logger.warning(f"Invalid Cypher (attempt 1), retrying: {cypher}")
                # Retry with simpler prompt
                retry_prompt = (
                    f"Question: {question}\n\nWrite a SIMPLE Cypher query (1-3 lines) for Neo4j. "
                    f"Write MATCH ... RETURN ... LIMIT 25 directly. NO markdown, NO explanation, NO thinking."
                )
                response = model.generate_content(
                    [QUERY_SYSTEM, retry_prompt],
                    generation_config={"temperature": 0.0, "max_output_tokens": 2048},
                )
                cypher = cls._clean_cypher(response.text.strip())
                logger.info(f"Generated Cypher (attempt 2): {cypher[:200]}")

            return {"cypher": cypher, "status": "ok"}
        except Exception as e:
            return {"cypher": "", "status": "error", "error": str(e)}


    @staticmethod
    def _is_valid_cypher(query: str) -> bool:
        """Basic validation: has RETURN, balanced parens, not empty."""
        if not query or "RETURN" not in query.upper():
            return False
        if query.count("(") != query.count(")"):
            return False
        if query.count("[") != query.count("]"):
            return False
        return True

    @staticmethod
    def _clean_cypher(text: str) -> str:
        """Strip markdown code blocks and extra whitespace from LLM output."""
        import re
        # Remove ```cypher ... ``` or ```sql ... ``` or ``` ... ```
        pattern = r"```(?:cypher|sql|plaintext)?\s*\n?(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: if starts with ``` but no closing, strip first line
        if text.startswith("```"):
            lines = text.split("\n")
            return "\n".join(lines[1:]).strip().rstrip("`")
        return text.strip()

    @classmethod
    async def generate_response(cls, question: str, kg_context: str) -> dict:
        """Generate NL response from question + KG context."""
        import logging
        logger = logging.getLogger(__name__)

        model = cls._get_model()

        prompt = f"Question: {question}\n\nKnowledge Graph Data:\n{kg_context}\n\nAnswer the question based on the KG data above. Respond in formal Indonesian (Bahasa Indonesia)."

        logger.info(f"=== RESPONSE PROMPT ({len(prompt)} chars) ===")
        logger.info(prompt[:2000])
        logger.info("=== END PROMPT ===")

        try:
            response = model.generate_content(
                [RESPONSE_SYSTEM, prompt],
                generation_config={"temperature": 0.3},
            )
            answer = response.text.strip()
            logger.info(f"=== LLM RESPONSE ({len(answer)} chars) ===")
            logger.info(answer[:500])
            return {"answer": answer, "status": "ok"}
        except Exception as e:
            logger.error(f"LLM response error: {e}")
            return {"answer": "", "status": "error", "error": str(e)}
