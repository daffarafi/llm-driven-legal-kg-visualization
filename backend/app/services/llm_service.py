"""LLM Service — Gemini API as placeholder, switch to fine-tuned model later."""

import google.generativeai as genai
from app.config import settings

# KG Schema for system prompt
KG_SCHEMA = """
Node Types: UndangUndang, Bab, Pasal, Ayat, EntitasHukum, PerbuatanHukum, Sanksi, KonsepHukum
Relation Types: MEMUAT, MENGATUR, MENETAPKAN_SANKSI, BERLAKU_UNTUK, MERUJUK, MENDEFINISIKAN
Properties: label, content, uu_number, source_document_id

Contoh label node:
- UndangUndang: "Undang-Undang tentang Informasi dan Transaksi Elektronik"
- Bab: "Bab I Ketentuan Umum", "Bab VII Perbuatan Yang Dilarang"
- Pasal: "Pasal 1", "Pasal 27", "Pasal 45"
- KonsepHukum: "Informasi Elektronik", "Dokumen Elektronik"
- EntitasHukum: "Setiap Orang", "Penyelenggara Sistem Elektronik"
- PerbuatanHukum: "mendistribusikan informasi yang melanggar kesusilaan", "akses ilegal"
- Sanksi: "pidana penjara paling lama 6 (enam) tahun"

Struktur graf: UndangUndang -[MEMUAT]-> Bab -[MEMUAT]-> Pasal
Pasal -[MENDEFINISIKAN]-> KonsepHukum/EntitasHukum
Pasal -[MENGATUR]-> PerbuatanHukum
Pasal -[MENETAPKAN_SANKSI]-> Sanksi
"""

QUERY_SYSTEM = f"""Anda adalah asisten yang mengubah pertanyaan hukum Indonesia menjadi Cypher query untuk Neo4j.
Schema KG:
{KG_SCHEMA}

PENTING - Rules:
1. Output HANYA Cypher query mentah. TANPA markdown, TANPA ```, TANPA penjelasan
2. SELALU gunakan CONTAINS untuk filter label. JANGAN pernah exact match.
   BENAR: WHERE uu.label CONTAINS 'Informasi' OR uu.label CONTAINS 'Elektronik'
   SALAH: WHERE uu.label = 'UU ITE'
3. Gunakan LIMIT 10 di akhir query
4. Query harus lengkap dan valid (ada MATCH dan RETURN)

Contoh output yang benar:
- "Pasal apa saja di UU ITE?"
  MATCH (uu:UndangUndang)-[:MEMUAT]->(b:Bab)-[:MEMUAT]->(p:Pasal) WHERE uu.label CONTAINS 'Elektronik' RETURN p.label, p.content LIMIT 10

- "Apa sanksi pencemaran nama baik?"
  MATCH (p:Pasal)-[:MENETAPKAN_SANKSI]->(s:Sanksi) WHERE p.content CONTAINS 'nama baik' OR p.label CONTAINS 'nama baik' RETURN p.label, p.content, s.label, s.content LIMIT 10

- "Apa itu informasi elektronik?"
  MATCH (p:Pasal)-[:MENDEFINISIKAN]->(k:KonsepHukum) WHERE k.label CONTAINS 'Informasi Elektronik' RETURN p.label, k.label, k.content LIMIT 10"""

RESPONSE_SYSTEM = """Anda adalah asisten hukum Indonesia. Jawab pertanyaan pengguna berdasarkan data dari Knowledge Graph.
Rules:
1. Sertakan referensi pasal dan UU
2. Gunakan bahasa Indonesia formal
3. Jika data tidak cukup, katakan dengan jelas"""


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
        model = cls._get_model()

        prompt = f"Pertanyaan: {question}\n\nBerikan Cypher query untuk menjawab pertanyaan di atas."

        try:
            response = model.generate_content(
                [QUERY_SYSTEM, prompt],
                generation_config={"temperature": 0.1, "max_output_tokens": 1024},
            )
            cypher = cls._clean_cypher(response.text.strip())

            # Validate: must contain RETURN and have balanced parentheses
            if not cls._is_valid_cypher(cypher):
                # Retry with simpler prompt
                retry_prompt = (
                    f"Pertanyaan: {question}\n\n"
                    "Tulis Cypher query SEDERHANA (1-3 baris saja) untuk Neo4j. "
                    "Langsung tulis MATCH ... RETURN ... LIMIT 10. "
                    "JANGAN gunakan markdown."
                )
                response = model.generate_content(
                    [QUERY_SYSTEM, retry_prompt],
                    generation_config={"temperature": 0.0, "max_output_tokens": 512},
                )
                cypher = cls._clean_cypher(response.text.strip())

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
        model = cls._get_model()

        prompt = (
            f"Pertanyaan: {question}\n\n"
            f"Data dari Knowledge Graph:\n{kg_context}\n\n"
            f"Jawab pertanyaan berdasarkan data KG di atas."
        )

        try:
            response = model.generate_content(
                [RESPONSE_SYSTEM, prompt],
                generation_config={"temperature": 0.3, "max_output_tokens": 1024},
            )
            return {"answer": response.text.strip(), "status": "ok"}
        except Exception as e:
            return {"answer": "", "status": "error", "error": str(e)}
