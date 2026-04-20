"""
Training Data Generator for NL → Cypher Query Model.

Generates (question, cypher_query) pairs from the Neo4j Knowledge Graph,
using two strategies:
1. Template-based: Predefined Cypher templates filled with real KG entities
2. LLM-assisted: Gemini generates diverse question variations + Cypher, validated via EXPLAIN

Output: Google Sheets (train/val/prompt tabs) + local CSV fallback.

Follows patterns from gllm_training reference code (qa_generator.ipynb).
"""

import json
import os
import re
import csv
import random
import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)


# ============================================================
# Data structures
# ============================================================

@dataclass
class TrainingSample:
    """A single NL-Cypher training pair."""
    context: str        # KG schema (same for all samples)
    question: str       # Natural language question
    response: str       # Cypher query (the label)
    category: str       # hierarki | sanksi | entitas | relasi | agregasi | definisi


# ============================================================
# KG Schema embedded in training context
# ============================================================

KG_SCHEMA = """Node Types:
- Regulasi (properties: label, content, jenis, source_document_id) — dokumen regulasi (jenis: Undang-Undang, POJK, PP, etc.)
- Bab (properties: label, content)
- Bagian (properties: label, content) — bagian dalam bab
- Pasal (properties: label, content)
- Ayat (properties: label, content)
- EntitasHukum (properties: label, content) — subjek/objek: orang, badan, institusi
- PerbuatanHukum (properties: label, content) — tindakan yang diatur/dilarang
- Sanksi (properties: label, content) — hukuman pidana/denda
- KonsepHukum (properties: label, content) — konsep abstrak/definisi
- VersiPasal (properties: label, version, status, source_document_id, content) — versi pasal yang diamandemen

Relation Types:
- MEMUAT (Regulasi → Bab, Bab → Bagian, Bab → Pasal, Bagian → Pasal)
- MEMILIKI_AYAT (Pasal → Ayat)
- MENGATUR (Pasal/Ayat → PerbuatanHukum)
- MENETAPKAN_SANKSI (Pasal/Ayat → Sanksi)
- BERLAKU_UNTUK (Pasal/Ayat → EntitasHukum)
- MERUJUK (Pasal/Ayat → Pasal/Ayat)
- MENDEFINISIKAN (Pasal/Ayat → KonsepHukum)
- MENGAMANDEMEN (Regulasi → Regulasi) — amandemen
- DIAMANDEMEN_OLEH (Regulasi → Regulasi)
- DITURUNKAN_KE (Regulasi → Regulasi) — hierarki regulasi
- DITURUNKAN_DARI (Regulasi → Regulasi)
- MENCABUT (Regulasi → Regulasi)
- DICABUT_OLEH (Regulasi → Regulasi)
- MERUJUK_DOKUMEN (Entity → Regulasi) — rujukan antar-dokumen
- MENGUBAH_PASAL (Regulasi → Entity) — perubahan pasal spesifik
- MENYISIPKAN_PASAL (Regulasi → Entity)
- MENGHAPUS_PASAL (Regulasi → Entity)
- MEMILIKI_VERSI (Entity → VersiPasal)
- DIAMANDEMEN_MENJADI (VersiPasal → VersiPasal) — versioning"""


SYSTEM_INSTRUCTION = f"""Anda adalah asisten yang mengubah pertanyaan hukum Indonesia menjadi Cypher query untuk Neo4j Knowledge Graph.

Schema KG:
{KG_SCHEMA}

Aturan:
1. Output HANYA Cypher query, tanpa penjelasan
2. Gunakan MATCH, WHERE, RETURN yang sesuai
3. String comparison gunakan CONTAINS atau exact match
4. Untuk pencarian teks gunakan toLower() untuk case-insensitive"""


# ============================================================
# Template-based generation
# ============================================================

TEMPLATES = [
    # === HIERARKI ===
    {
        "nl": "Apa saja pasal yang terdapat dalam {bab_label}?",
        "cypher": "MATCH (b {{label: '{bab_label}'}})-[:MEMUAT|MEMILIKI_PASAL]->(p:Pasal) RETURN p.label AS pasal, p.content AS isi",
        "category": "hierarki",
        "fill_query": "MATCH (b:Bab) RETURN b.label AS bab_label LIMIT 30",
    },
    {
        "nl": "Bab apa saja yang ada dalam UU ITE?",
        "cypher": "MATCH (r:Regulasi)-[:MEMUAT]->(b:Bab) WHERE toLower(r.label) CONTAINS 'ite' OR toLower(r.label) CONTAINS '11' RETURN b.label AS bab",
        "category": "hierarki",
        "fill_query": None,  # static
    },
    {
        "nl": "Sebutkan isi dari {pasal_label}!",
        "cypher": "MATCH (p:Pasal {{label: '{pasal_label}'}}) RETURN p.label AS pasal, p.content AS isi",
        "category": "hierarki",
        "fill_query": "MATCH (p:Pasal) RETURN p.label AS pasal_label LIMIT 50",
    },
    {
        "nl": "{pasal_label} termasuk dalam bab apa?",
        "cypher": "MATCH (b:Bab)-[:MEMUAT|MEMILIKI_PASAL]->(p:Pasal {{label: '{pasal_label}'}}) RETURN b.label AS bab",
        "category": "hierarki",
        "fill_query": "MATCH (p:Pasal) RETURN p.label AS pasal_label LIMIT 30",
    },

    # === SANKSI ===
    {
        "nl": "Apa sanksi untuk {perbuatan_label}?",
        "cypher": "MATCH (p:Pasal)-[:MENGATUR]->(ph:PerbuatanHukum) WHERE toLower(ph.label) CONTAINS toLower('{perbuatan_label}') MATCH (p)-[:MENETAPKAN_SANKSI]->(s:Sanksi) RETURN p.label AS pasal, ph.label AS perbuatan, s.label AS sanksi",
        "category": "sanksi",
        "fill_query": "MATCH (ph:PerbuatanHukum)<-[:MENGATUR]-(p:Pasal)-[:MENETAPKAN_SANKSI]->(s:Sanksi) RETURN ph.label AS perbuatan_label LIMIT 30",
    },
    {
        "nl": "Pasal mana yang menetapkan sanksi pidana penjara?",
        "cypher": "MATCH (p:Pasal)-[:MENETAPKAN_SANKSI]->(s:Sanksi) WHERE toLower(s.label) CONTAINS 'penjara' RETURN p.label AS pasal, s.label AS sanksi",
        "category": "sanksi",
        "fill_query": None,
    },
    {
        "nl": "Sanksi apa saja yang diatur dalam {pasal_label}?",
        "cypher": "MATCH (p:Pasal {{label: '{pasal_label}'}})-[:MENETAPKAN_SANKSI]->(s:Sanksi) RETURN s.label AS sanksi, s.content AS detail",
        "category": "sanksi",
        "fill_query": "MATCH (p:Pasal)-[:MENETAPKAN_SANKSI]->(s:Sanksi) RETURN p.label AS pasal_label LIMIT 30",
    },
    {
        "nl": "Berapa hukuman denda maksimal untuk {perbuatan_label}?",
        "cypher": "MATCH (p:Pasal)-[:MENGATUR]->(ph:PerbuatanHukum) WHERE toLower(ph.label) CONTAINS toLower('{perbuatan_label}') MATCH (p)-[:MENETAPKAN_SANKSI]->(s:Sanksi) WHERE toLower(s.label) CONTAINS 'denda' RETURN p.label AS pasal, s.label AS sanksi_denda",
        "category": "sanksi",
        "fill_query": "MATCH (ph:PerbuatanHukum)<-[:MENGATUR]-(p:Pasal)-[:MENETAPKAN_SANKSI]->(s:Sanksi) WHERE toLower(s.label) CONTAINS 'denda' RETURN ph.label AS perbuatan_label LIMIT 20",
    },

    # === ENTITAS ===
    {
        "nl": "Pasal apa yang berlaku untuk {entitas_label}?",
        "cypher": "MATCH (p:Pasal)-[:BERLAKU_UNTUK]->(e:EntitasHukum) WHERE toLower(e.label) CONTAINS toLower('{entitas_label}') RETURN p.label AS pasal, p.content AS isi",
        "category": "entitas",
        "fill_query": "MATCH (e:EntitasHukum) RETURN e.label AS entitas_label LIMIT 30",
    },
    {
        "nl": "Siapa saja entitas hukum yang diatur dalam UU ITE?",
        "cypher": "MATCH (e:EntitasHukum)<-[:BERLAKU_UNTUK]-(p:Pasal) RETURN DISTINCT e.label AS entitas ORDER BY entitas",
        "category": "entitas",
        "fill_query": None,
    },
    {
        "nl": "Apa kewajiban {entitas_label} menurut UU ITE?",
        "cypher": "MATCH (p:Pasal)-[:BERLAKU_UNTUK]->(e:EntitasHukum) WHERE toLower(e.label) CONTAINS toLower('{entitas_label}') RETURN p.label AS pasal, p.content AS kewajiban",
        "category": "entitas",
        "fill_query": "MATCH (e:EntitasHukum) RETURN e.label AS entitas_label LIMIT 20",
    },

    # === RELASI / REFERENSI ===
    {
        "nl": "Pasal mana yang merujuk ke {pasal_label}?",
        "cypher": "MATCH (p1:Pasal)-[:MERUJUK]->(p2:Pasal {{label: '{pasal_label}'}}) RETURN p1.label AS pasal_merujuk, p1.content AS isi",
        "category": "relasi",
        "fill_query": "MATCH (p1:Pasal)-[:MERUJUK]->(p2:Pasal) RETURN DISTINCT p2.label AS pasal_label LIMIT 20",
    },
    {
        "nl": "Apa hubungan antara {pasal1_label} dan {pasal2_label}?",
        "cypher": "MATCH (p1:Pasal {{label: '{pasal1_label}'}})-[r]->(p2:Pasal {{label: '{pasal2_label}'}}) RETURN p1.label AS dari, type(r) AS relasi, p2.label AS ke",
        "category": "relasi",
        "fill_query": "MATCH (p1:Pasal)-[r]->(p2:Pasal) RETURN p1.label AS pasal1_label, p2.label AS pasal2_label LIMIT 20",
    },

    # === AGREGASI ===
    {
        "nl": "Berapa jumlah pasal dalam UU ITE?",
        "cypher": "MATCH (p:Pasal) RETURN count(p) AS jumlah_pasal",
        "category": "agregasi",
        "fill_query": None,
    },
    {
        "nl": "Berapa total sanksi yang diatur dalam UU ITE?",
        "cypher": "MATCH (s:Sanksi) RETURN count(s) AS jumlah_sanksi",
        "category": "agregasi",
        "fill_query": None,
    },
    {
        "nl": "Berapa banyak perbuatan hukum yang diatur?",
        "cypher": "MATCH (ph:PerbuatanHukum) RETURN count(ph) AS jumlah_perbuatan",
        "category": "agregasi",
        "fill_query": None,
    },
    {
        "nl": "{bab_label} memiliki berapa pasal?",
        "cypher": "MATCH (b {{label: '{bab_label}'}})-[:MEMUAT|MEMILIKI_PASAL]->(p:Pasal) RETURN b.label AS bab, count(p) AS jumlah_pasal",
        "category": "agregasi",
        "fill_query": "MATCH (b:Bab) RETURN b.label AS bab_label LIMIT 20",
    },

    # === DEFINISI ===
    {
        "nl": "Apa definisi {konsep_label} menurut UU ITE?",
        "cypher": "MATCH (p:Pasal)-[:MENDEFINISIKAN]->(k:KonsepHukum) WHERE toLower(k.label) CONTAINS toLower('{konsep_label}') RETURN k.label AS konsep, k.content AS definisi, p.label AS pasal",
        "category": "definisi",
        "fill_query": "MATCH (k:KonsepHukum) RETURN k.label AS konsep_label LIMIT 30",
    },
    {
        "nl": "Apa saja konsep hukum yang didefinisikan dalam UU ITE?",
        "cypher": "MATCH (k:KonsepHukum) RETURN k.label AS konsep, k.content AS definisi ORDER BY konsep",
        "category": "definisi",
        "fill_query": None,
    },
    {
        "nl": "Pasal berapa yang mendefinisikan tentang {konsep_label}?",
        "cypher": "MATCH (p:Pasal)-[:MENDEFINISIKAN]->(k:KonsepHukum) WHERE toLower(k.label) CONTAINS toLower('{konsep_label}') RETURN p.label AS pasal, k.label AS konsep",
        "category": "definisi",
        "fill_query": "MATCH (k:KonsepHukum) RETURN k.label AS konsep_label LIMIT 20",
    },

    # === CROSS-DOCUMENT ===
    {
        "nl": "Peraturan apa saja yang merupakan turunan dari UU ITE?",
        "cypher": "MATCH (u:Peraturan)-[:DITURUNKAN_KE]->(p:Peraturan) WHERE toLower(u.label) CONTAINS 'informasi dan transaksi elektronik' OR u.short_name = 'UU ITE' RETURN p.label AS peraturan_turunan, p.short_name AS nama_singkat, p.year AS tahun",
        "category": "cross_document",
        "fill_query": None,
    },
    {
        "nl": "Peraturan mana yang mencabut {target_label}?",
        "cypher": "MATCH (p:Peraturan)-[:MENCABUT]->(target:Peraturan) WHERE toLower(target.label) CONTAINS toLower('{target_label}') RETURN p.label AS peraturan_pencabut, p.year AS tahun",
        "category": "cross_document",
        "fill_query": "MATCH (p:Peraturan)-[:MENCABUT]->(t:Peraturan) RETURN t.short_name AS target_label LIMIT 10",
    },
    {
        "nl": "Apa saja peraturan yang merujuk ke {ref_label}?",
        "cypher": "MATCH (src)-[:MERUJUK_DOKUMEN]->(target:Peraturan) WHERE toLower(target.label) CONTAINS toLower('{ref_label}') OR target.short_name = '{ref_label}' RETURN DISTINCT src.label AS sumber, src.source_document_id AS dokumen_sumber",
        "category": "cross_document",
        "fill_query": "MATCH ()-[:MERUJUK_DOKUMEN]->(t:Peraturan) RETURN DISTINCT t.short_name AS ref_label LIMIT 10",
    },
    {
        "nl": "Berapa jumlah peraturan dalam Knowledge Graph?",
        "cypher": "MATCH (p:Peraturan) RETURN count(p) AS jumlah_peraturan",
        "category": "cross_document",
        "fill_query": None,
    },
    {
        "nl": "Tamplikan semua peraturan beserta statusnya",
        "cypher": "MATCH (p:Peraturan) RETURN p.label AS peraturan, p.short_name AS nama_singkat, p.regulation_type AS jenis, p.year AS tahun, p.status AS status ORDER BY p.year",
        "category": "cross_document",
        "fill_query": None,
    },
    {
        "nl": "Apa saja PP yang merupakan pelaksanaan dari UU ITE?",
        "cypher": "MATCH (pp:Peraturan {regulation_type: 'PP'})-[:DITURUNKAN_DARI]->(uu:Peraturan) WHERE uu.short_name = 'UU ITE' RETURN pp.label AS pp_pelaksanaan, pp.year AS tahun, pp.status AS status",
        "category": "cross_document",
        "fill_query": None,
    },
    {
        "nl": "Peraturan apa yang mengamandemen {target_label}?",
        "cypher": "MATCH (amender:Peraturan)-[:MENGAMANDEMEN]->(target:Peraturan) WHERE toLower(target.label) CONTAINS toLower('{target_label}') OR target.short_name = '{target_label}' RETURN amender.label AS peraturan_amandemen, amender.year AS tahun",
        "category": "cross_document",
        "fill_query": "MATCH ()-[:MENGAMANDEMEN]->(t:Peraturan) RETURN DISTINCT t.short_name AS target_label LIMIT 10",
    },
    {
        "nl": "Bagaimana hierarki regulasi dari {source_label}?",
        "cypher": "MATCH path=(src:Peraturan)-[:DITURUNKAN_KE|MENGAMANDEMEN|MENCABUT*1..3]->(target:Peraturan) WHERE src.short_name = '{source_label}' RETURN [n IN nodes(path) | n.short_name] AS hierarki, [r IN relationships(path) | type(r)] AS relasi",
        "category": "cross_document",
        "fill_query": "MATCH (p:Peraturan) WHERE p.regulation_type = 'UU' RETURN p.short_name AS source_label LIMIT 5",
    },

    # === AMANDEMEN / VERSIONING ===
    {
        "nl": "Pasal apa saja yang diubah oleh UU 19/2016?",
        "cypher": "MATCH (amender:Peraturan {id: 'UU_19_2016'})-[:MENGUBAH_PASAL]->(p) RETURN p.label AS pasal_diubah, p.source_document_id AS dokumen_asal",
        "category": "amandemen",
        "fill_query": None,
    },
    {
        "nl": "Apakah {pasal_label} sudah diamandemen?",
        "cypher": "MATCH (p:Entity)-[:MEMILIKI_VERSI]->(v:VersiPasal) WHERE toLower(p.label) CONTAINS toLower('{pasal_label}') RETURN p.label AS pasal, v.label AS versi, v.version AS nomor_versi, v.status AS status ORDER BY v.version",
        "category": "amandemen",
        "fill_query": "MATCH (p:Entity)-[:MEMILIKI_VERSI]->(v:VersiPasal) RETURN DISTINCT p.label AS pasal_label LIMIT 10",
    },
    {
        "nl": "Tampilkan semua pasal yang memiliki versi amandemen",
        "cypher": "MATCH (v1:VersiPasal)-[:DIAMANDEMEN_MENJADI]->(v2:VersiPasal) RETURN v1.label AS versi_lama, v1.status AS status_lama, v2.label AS versi_baru, v2.status AS status_baru",
        "category": "amandemen",
        "fill_query": None,
    },
    {
        "nl": "Berapa pasal yang diamandemen dari UU ITE asli?",
        "cypher": "MATCH (v:VersiPasal {source_document_id: 'UU_11_2008'}) WHERE v.status = 'diamandemen' RETURN count(v) AS jumlah_pasal_diamandemen",
        "category": "amandemen",
        "fill_query": None,
    },
    {
        "nl": "Pasal baru apa yang disisipkan oleh UU 19/2016?",
        "cypher": "MATCH (v:VersiPasal) WHERE v.source_document_id = 'UU_19_2016' AND v.status = 'baru (disisipkan)' RETURN v.label AS pasal_baru, v.content AS keterangan",
        "category": "amandemen",
        "fill_query": None,
    },
    {
        "nl": "Apakah {peraturan_label} masih berlaku?",
        "cypher": "MATCH (p:Peraturan) WHERE toLower(p.label) CONTAINS toLower('{peraturan_label}') OR p.short_name = '{peraturan_label}' RETURN p.label AS peraturan, p.status AS status, p.year AS tahun",
        "category": "amandemen",
        "fill_query": "MATCH (p:Peraturan) RETURN p.short_name AS peraturan_label LIMIT 10",
    },
    {
        "nl": "Apa saja entitas dari dokumen {doc_id}?",
        "cypher": "MATCH (n:Entity) WHERE n.source_document_id = '{doc_id}' RETURN n.node_type AS tipe, n.label AS label, n.content AS konten LIMIT 50",
        "category": "cross_document",
        "fill_query": "MATCH (p:Peraturan) RETURN p.id AS doc_id LIMIT 10",
    },
]


def generate_from_templates(driver) -> list[TrainingSample]:
    """Generate NL-Cypher pairs from templates filled with actual Neo4j data."""
    samples = []

    for template in TEMPLATES:
        fill_query = template.get("fill_query")

        if fill_query is None:
            # Static template — just use it as-is
            samples.append(TrainingSample(
                context=KG_SCHEMA,
                question=template["nl"],
                response=template["cypher"],
                category=template["category"],
            ))
            continue

        # Dynamic template — fill from Neo4j
        try:
            with driver.session() as session:
                results = session.run(fill_query).data()
        except Exception as e:
            print(f"  [WARN] Fill query failed for template '{template['nl'][:40]}': {e}")
            continue

        for row in results:
            try:
                nl = template["nl"].format(**row)
                cypher = template["cypher"].format(**row)
                samples.append(TrainingSample(
                    context=KG_SCHEMA,
                    question=nl,
                    response=cypher,
                    category=template["category"],
                ))
            except KeyError as e:
                print(f"  [WARN] Missing key {e} in row {row}")
                continue

    return samples


# ============================================================
# LLM-assisted generation
# ============================================================

LLM_GENERATION_PROMPT_SYSTEM = f"""<ROLE>
You are an assistant specialized in generating synthetic NL-to-Cypher datasets for Indonesian legal Knowledge Graphs.
Your outputs must be valid Cypher queries that work with the following schema.
</ROLE>

<KG_SCHEMA>
{KG_SCHEMA}
</KG_SCHEMA>

<RULES>
1. Generate diverse questions in Bahasa Indonesia (casual and formal)
2. Each Cypher query must be valid against the schema above
3. Vary question types: siapa, apa, berapa, bagaimana, sebutkan, jelaskan
4. Include questions that combine multiple relationships (multi-hop)
5. Always return valid JSON
</RULES>"""

LLM_GENERATION_PROMPT_USER = """<INPUT>
<TOTAL_DATA>{total_data}</TOTAL_DATA>
<EXAMPLE_ENTITIES>
{example_entities}
</EXAMPLE_ENTITIES>
<EXISTING_QUESTIONS>
{existing_questions}
</EXISTING_QUESTIONS>
</INPUT>

Generate {total_data} new NL-Cypher pairs yang BERBEDA dari existing questions.
Variasikan gaya bahasa, tipe pertanyaan, dan entity yang digunakan.

<OUTPUT_FORMAT>
{{
  "pairs": [
    {{
      "question": "pertanyaan dalam bahasa Indonesia",
      "cypher": "MATCH ... RETURN ...",
      "category": "hierarki|sanksi|entitas|relasi|agregasi|definisi"
    }}
  ]
}}
</OUTPUT_FORMAT>"""


def generate_with_llm(
    driver,
    existing_samples: list[TrainingSample],
    num_samples: int = 50,
    api_key: str = "",
    model_name: str = "gemini-2.5-flash",
) -> list[TrainingSample]:
    """Generate diverse NL-Cypher pairs using Gemini, then validate Cypher syntax."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Gather example entities from Neo4j
    entity_examples = {}
    entity_queries = {
        "Pasal": "MATCH (p:Pasal) RETURN p.label LIMIT 10",
        "Bab": "MATCH (b:Bab) RETURN b.label LIMIT 10",
        "EntitasHukum": "MATCH (e:EntitasHukum) RETURN e.label LIMIT 10",
        "PerbuatanHukum": "MATCH (ph:PerbuatanHukum) RETURN ph.label LIMIT 10",
        "Sanksi": "MATCH (s:Sanksi) RETURN s.label LIMIT 10",
        "KonsepHukum": "MATCH (k:KonsepHukum) RETURN k.label LIMIT 10",
    }

    with driver.session() as session:
        for node_type, query in entity_queries.items():
            try:
                results = session.run(query).data()
                entity_examples[node_type] = [list(r.values())[0] for r in results]
            except Exception:
                entity_examples[node_type] = []

    example_entities_str = json.dumps(entity_examples, ensure_ascii=False, indent=2)
    existing_qs = "\n".join([f"- {s.question}" for s in existing_samples[:30]])

    # Generate in batches of 20
    all_samples = []
    batch_size = min(20, num_samples)
    batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(batches):
        remaining = min(batch_size, num_samples - len(all_samples))
        if remaining <= 0:
            break

        user_prompt = LLM_GENERATION_PROMPT_USER.format(
            total_data=remaining,
            example_entities=example_entities_str,
            existing_questions=existing_qs,
        )

        try:
            response = model.generate_content(
                [LLM_GENERATION_PROMPT_SYSTEM, user_prompt],
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.9,
                },
            )

            raw = json.loads(response.text)
            pairs = raw.get("pairs", raw.get("data", []))

            # Validate each Cypher query
            validated = 0
            with driver.session() as session:
                for pair in pairs:
                    cypher = pair.get("cypher", "")
                    try:
                        session.run(f"EXPLAIN {cypher}")
                        all_samples.append(TrainingSample(
                            context=KG_SCHEMA,
                            question=pair.get("question", ""),
                            response=cypher,
                            category=pair.get("category", "lainnya"),
                        ))
                        validated += 1
                    except Exception:
                        pass  # Skip invalid Cypher

            print(f"  Batch {batch_idx+1}/{batches}: {validated}/{len(pairs)} valid")

        except Exception as e:
            print(f"  [ERROR] LLM batch {batch_idx+1}: {e}")

        time.sleep(2)  # Rate limiting

    return all_samples


# ============================================================
# Cypher validation
# ============================================================

def validate_cypher_queries(samples: list[TrainingSample], driver) -> tuple[list[TrainingSample], list[dict]]:
    """Validate all Cypher queries via EXPLAIN. Returns (valid, errors)."""
    valid = []
    errors = []

    with driver.session() as session:
        for s in samples:
            try:
                session.run(f"EXPLAIN {s.response}")
                valid.append(s)
            except Exception as e:
                errors.append({
                    "question": s.question,
                    "cypher": s.response,
                    "error": str(e),
                })

    return valid, errors


# ============================================================
# Output: CSV + optional Google Sheets
# ============================================================

def save_to_csv(samples: list[TrainingSample], output_dir: str, prefix: str = "") -> tuple[str, str]:
    """Save samples to train/val CSV files (80/20 split)."""
    os.makedirs(output_dir, exist_ok=True)

    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train = samples[:split]
    val = samples[split:]

    train_path = os.path.join(output_dir, f"{prefix}training_data.csv")
    val_path = os.path.join(output_dir, f"{prefix}validation_data.csv")

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["context", "question", "response", "category"])
            writer.writeheader()
            for s in data:
                writer.writerow(asdict(s))

    return train_path, val_path


def save_prompt_template_csv(output_dir: str) -> str:
    """Save the prompt template as CSV (for reference / Google Sheets upload)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "prompt_data.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "model", "topic", "system", "user", "notes", "date"])
        writer.writeheader()
        writer.writerow({
            "name": "nl_to_cypher_1",
            "model": "Qwen/Qwen3-4b",
            "topic": "LEGAL_KG_QUERY",
            "system": SYSTEM_INSTRUCTION,
            "user": "<INPUT>\n<CONTEXT>\n{context}\n</CONTEXT>\n<QUESTION>\n{question}\n</QUESTION>\n</INPUT>",
            "notes": "NL to Cypher for Indonesian legal KG",
            "date": "",
        })

    return path


def upload_to_google_sheets(
    samples: list[TrainingSample],
    spreadsheet_id: str,
    train_sheet: str = "training_data",
    val_sheet: str = "validation_data",
    client_email: str = "",
    private_key: str = "",
    batch_size: int = 10,
    max_retries: int = 5,
    batch_delay: float = 2.0,
):
    """Upload samples to Google Sheets with batch processing, retry, and progress tracking.

    Uses GoogleSheetsWriter for robust uploads with:
    - Batch processing to manage API rate limits
    - Exponential backoff retry on 429 errors
    - tqdm progress bar for visibility
    - BatchWriteResult summary (successful/failed/errors)

    Args:
        samples: List of TrainingSample to upload
        spreadsheet_id: Google Spreadsheet ID
        train_sheet: Worksheet name for training data
        val_sheet: Worksheet name for validation data
        client_email: Google service account email (falls back to env var)
        private_key: Google service account private key (falls back to env var)
        batch_size: Number of rows per batch (default: 10)
        max_retries: Max retry attempts on rate limit errors (default: 5)
        batch_delay: Seconds to wait between batches (default: 2.0)
    """
    try:
        import pandas as pd
        from modules.google_sheets_utils import GoogleUtil, GoogleSheetsWriter
    except ImportError:
        print("[WARN] google_sheets_utils not available. Skipping Sheets upload.")
        return

    # Use provided credentials, fall back to env vars
    client_email = client_email or os.getenv("GOOGLE_SHEETS_CLIENT_EMAIL", "")
    private_key = private_key or os.getenv("GOOGLE_SHEETS_PRIVATE_KEY", "")

    if not client_email or not private_key:
        print("[WARN] Google Sheets credentials not configured. Skipping upload.")
        return

    google = GoogleUtil(private_key, client_email)

    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train = samples[:split]
    val = samples[split:]

    print(f"Uploading {len(train)} train + {len(val)} val samples to Sheets...")

    for sheet_name, data in [(train_sheet, train), (val_sheet, val)]:
        print(f"\n--- Writing {len(data)} rows to '{sheet_name}' ---")

        # Convert TrainingSamples to DataFrame
        df = pd.DataFrame([asdict(s) for s in data])

        # Use GoogleSheetsWriter for robust batch upload
        writer = GoogleSheetsWriter(
            google_util=google,
            sheet_id=spreadsheet_id,
            worksheet_name=sheet_name,
            batch_size=batch_size,
            max_retries=max_retries,
            batch_delay=batch_delay,
        )

        result = writer.write_dataframe(df, show_progress=True)

        # Summary
        print(f"  ✓ {result.successful_rows} rows written successfully")
        if result.failed_rows > 0:
            print(f"  ✗ {result.failed_rows} rows failed")
            for error in result.errors[:3]:
                print(f"    Row {error['row_number']}: {error['error'][:100]}")

    print("\nGoogle Sheets upload complete!")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate NL-Cypher training data from Neo4j KG")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "passwd123"))
    parser.add_argument("--output", default="finetuning/query_model/data", help="Output directory")
    parser.add_argument("--num-llm-samples", type=int, default=50, help="Number of LLM-generated samples")
    parser.add_argument("--spreadsheet-id", default=os.getenv("GOOGLE_SPREADSHEET_ID", ""), help="Google Sheets ID")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-assisted generation")
    parser.add_argument("--skip-sheets", action="store_true", help="Skip Google Sheets upload")
    args = parser.parse_args()

    from neo4j import GraphDatabase

    api_key = os.getenv("GEMINI_API_KEY", "")

    print("=" * 60)
    print("NL -> Cypher Training Data Generator")
    print("=" * 60)

    # Connect to Neo4j
    print(f"\n1. Connecting to Neo4j at {args.neo4j_uri}...")
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))

    try:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS cnt").single()
            print(f"   → Connected! {result['cnt']} nodes in graph")
    except Exception as e:
        print(f"   [ERROR] Cannot connect to Neo4j: {e}")
        return

    # Step 1: Template-based generation
    print("\n2. Generating template-based pairs...")
    template_samples = generate_from_templates(driver)
    print(f"   → {len(template_samples)} pairs generated")

    # Step 2: LLM-assisted generation
    llm_samples = []
    if not args.skip_llm and api_key:
        print(f"\n3. Generating {args.num_llm_samples} LLM-assisted pairs...")
        llm_samples = generate_with_llm(
            driver, template_samples, args.num_llm_samples, api_key
        )
        print(f"   → {len(llm_samples)} valid pairs generated")
    else:
        print("\n3. Skipping LLM-assisted generation")

    all_samples = template_samples + llm_samples

    # Step 3: Validate all Cypher queries
    print(f"\n4. Validating {len(all_samples)} Cypher queries...")
    valid_samples, errors = validate_cypher_queries(all_samples, driver)
    print(f"   → {len(valid_samples)} valid, {len(errors)} invalid")

    if errors:
        print(f"   First 3 errors:")
        for err in errors[:3]:
            print(f"     - {err['question'][:50]}: {err['error'][:80]}")

    # Step 4: Save to CSV
    print(f"\n5. Saving to CSV...")
    train_path, val_path = save_to_csv(valid_samples, args.output)
    prompt_path = save_prompt_template_csv(args.output)
    print(f"   → Train: {train_path}")
    print(f"   → Val:   {val_path}")
    print(f"   → Prompt: {prompt_path}")

    # Step 5: Upload to Google Sheets (optional)
    if not args.skip_sheets and args.spreadsheet_id:
        print(f"\n6. Uploading to Google Sheets...")
        upload_to_google_sheets(valid_samples, args.spreadsheet_id)
    else:
        print(f"\n6. Skipping Google Sheets upload")

    # Stats
    categories = {}
    for s in valid_samples:
        categories[s.category] = categories.get(s.category, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total samples:     {len(valid_samples)}")
    print(f"  Template-based:  {len(template_samples)}")
    print(f"  LLM-assisted:    {len(llm_samples)}")
    print(f"  Invalid removed: {len(errors)}")
    split = int(len(valid_samples) * 0.8)
    print(f"Train/Val split:   {split} / {len(valid_samples) - split}")
    print(f"Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    driver.close()


if __name__ == "__main__":
    main()
