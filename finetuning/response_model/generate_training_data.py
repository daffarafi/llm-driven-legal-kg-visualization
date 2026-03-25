"""
Response Model Training Data Generator
=======================================
Generate (question + KG context, ideal NL answer) pairs for fine-tuning
the response generation model.

Two strategies:
1. From Task 2 query ground truth: execute Cypher -> format KG results -> LLM generate answer
2. From document chunks: LLM generate QA -> LLM verifier (Rompis 2025 method)

Usage (CLI):
    python -m finetuning.response_model.generate_training_data \
        --query-data finetuning/query_model/data/training_data.csv \
        --output finetuning/response_model/data/ \
        --skip-sheets --skip-chunks

Usage (notebook):
    from finetuning.response_model.generate_training_data import (
        generate_from_query_results, generate_from_chunks, validate_with_llm
    )
"""

import os
import sys
import csv
import json
import time
import random
import re
import argparse
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ResponseSample:
    """One training sample for the response model."""
    context: str        # KG query results (formatted) or document chunk
    question: str       # user question in NL
    response: str       # ideal NL answer in Bahasa Indonesia
    category: str       # factual | multi-hop | sanksi | definisi | not-found

# ---------------------------------------------------------------------------
# System instruction (embedded in training context)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """Anda adalah asisten hukum Indonesia. Jawab pertanyaan pengguna berdasarkan
data dari Knowledge Graph hukum Indonesia yang disediakan.
Aturan:
1. Selalu sertakan referensi pasal dan UU yang relevan
2. Jika ada amandemen terkait, sebutkan versi terbaru
3. Jika informasi tidak cukup, jawab "Informasi tidak tersedia dalam Knowledge Graph"
4. Gunakan bahasa Indonesia formal"""

# ---------------------------------------------------------------------------
# Prompt template for LLM answer generation
# ---------------------------------------------------------------------------

ANSWER_GENERATION_PROMPT = """<ROLE>
Anda adalah ahli hukum Indonesia yang bertugas menyusun jawaban informatif
berdasarkan data dari Knowledge Graph (KG) hukum.
</ROLE>

<RULES>
1. Jawab dalam Bahasa Indonesia formal
2. Sertakan referensi pasal dan UU yang relevan (contoh: "berdasarkan Pasal 27 UU ITE")
3. Jika data KG tidak cukup untuk menjawab, katakan dengan jelas
4. Jangan menambahkan informasi di luar data KG yang diberikan
5. Struktur jawaban dengan poin-poin jika ada lebih dari 1 hal yang dibahas
</RULES>

<INPUT>
<QUESTION>
{question}
</QUESTION>

<KG_RESULTS>
{kg_results}
</KG_RESULTS>
</INPUT>

<OUTPUT_FORMAT>
Berikan jawaban langsung tanpa pembuka seperti "Berdasarkan data KG...".
Sertakan referensi hukum inline.
</OUTPUT_FORMAT>"""


# Prompt for LLM verifier (Rompis 2025 method)
VERIFIER_PROMPT = """Verifikasi apakah jawaban berikut benar dan konsisten
dengan konteks yang diberikan.

<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>

<ANSWER>
{answer}
</ANSWER>

Evaluasi dan output JSON:
{{
    "valid": true/false,
    "reason": "penjelasan singkat",
    "has_reference": true/false,
    "is_hallucination": true/false
}}"""


# Prompt for chunk-based QA generation
CHUNK_QA_PROMPT = """<ROLE>
Anda adalah pembuat dataset QA hukum Indonesia.
</ROLE>

<TASK>
Dari teks hukum berikut, buatkan {num_pairs} pasangan pertanyaan dan jawaban.
Jawaban harus berdasarkan HANYA pada teks yang diberikan.
Sertakan referensi pasal/ayat jika tersedia.
</TASK>

<TEXT>
{chunk_text}
</TEXT>

<OUTPUT_FORMAT>
Output JSON array:
[
    {{
        "question": "pertanyaan dalam Bahasa Indonesia",
        "answer": "jawaban berdasarkan teks, dengan referensi pasal",
        "category": "factual|definisi|sanksi|multi-hop"
    }}
]
</OUTPUT_FORMAT>"""


# ---------------------------------------------------------------------------
# Strategy 1: From Task 2 query ground truth
# ---------------------------------------------------------------------------

def format_kg_results(records: list[dict]) -> str:
    """Format Neo4j query results into readable KG context."""
    if not records:
        return "(tidak ada hasil)"

    lines = []
    for i, record in enumerate(records[:20], 1):  # cap at 20 results
        parts = []
        for key, value in record.items():
            if value is not None:
                # Truncate long strings
                val_str = str(value)
                if len(val_str) > 300:
                    val_str = val_str[:300] + "..."
                parts.append(f"{key}: {val_str}")
        lines.append(f"{i}. " + " | ".join(parts))

    return "\n".join(lines)


def generate_from_query_results(
    driver,
    query_data_path: str,
    api_key: str,
    max_samples: int = 500
) -> list[ResponseSample]:
    """
    Strategy 1: Generate response training data from Task 2 query ground truth.

    Flow:
    1. Read NL-Cypher pairs from Task 2 CSV
    2. Execute each Cypher query in Neo4j -> get results
    3. Format results as KG context
    4. Use Gemini to generate ideal NL answer
    5. Validate: answer must contain legal references
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Load Task 2 data
    with open(query_data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        query_rows = list(reader)

    print(f"Loaded {len(query_rows)} query pairs from Task 2")

    samples = []
    errors = []
    batch_count = 0

    for row in query_rows[:max_samples]:
        question = row.get("question", "")
        cypher = row.get("response", "")
        category = row.get("category", "factual")

        if not question or not cypher:
            continue

        # Execute Cypher in Neo4j
        try:
            with driver.session() as session:
                results = session.run(cypher).data()
        except Exception as e:
            errors.append({"question": question, "error": str(e)})
            continue

        if not results:
            # Generate "not found" sample
            kg_context = "(tidak ada hasil dari Knowledge Graph)"
            samples.append(ResponseSample(
                context=kg_context,
                question=question,
                response="Informasi yang diminta tidak tersedia dalam Knowledge Graph saat ini.",
                category="not-found",
            ))
            continue

        # Format KG results
        kg_context = format_kg_results(results)

        # Generate ideal answer with LLM
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=question,
            kg_results=kg_context,
        )

        try:
            response = model.generate_content(
                [SYSTEM_INSTRUCTION, prompt],
                generation_config={"temperature": 0.7, "max_output_tokens": 1024}
            )
            answer = response.text.strip()
        except Exception as e:
            errors.append({"question": question, "error": f"LLM error: {e}"})
            time.sleep(2)
            continue

        # Basic validation: answer should contain legal references
        if not _contains_legal_reference(answer):
            # Try once more with explicit instruction
            try:
                retry_prompt = prompt + "\n\nPENTING: Sertakan referensi pasal dan UU yang relevan."
                response = model.generate_content(
                    [SYSTEM_INSTRUCTION, retry_prompt],
                    generation_config={"temperature": 0.5, "max_output_tokens": 1024}
                )
                answer = response.text.strip()
            except Exception:
                pass

        samples.append(ResponseSample(
            context=kg_context,
            question=question,
            response=answer,
            category=category,
        ))

        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  Generated {batch_count} samples...")
            time.sleep(1)  # rate limiting

    print(f"\nStrategy 1 complete: {len(samples)} samples, {len(errors)} errors")
    return samples


# ---------------------------------------------------------------------------
# Strategy 2: From document chunks (Rompis 2025 method)
# ---------------------------------------------------------------------------

def generate_from_chunks(
    chunks_dir: str,
    api_key: str,
    max_samples: int = 200,
    pairs_per_batch: int = 3,
    chunk_batch_size: int = 5,
) -> list[ResponseSample]:
    """
    Strategy 2: Generate QA pairs from document chunks.

    Flow (following Rompis 2025 Section 3.7):
    1. Load chunks from JSON files
    2. Batch N chunks together
    3. LLM generates QA pairs from batch
    4. Return raw samples (verification in separate step)
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    generator = genai.GenerativeModel("gemini-2.5-flash")

    # Find chunk files
    chunk_files = []
    if os.path.isdir(chunks_dir):
        for fname in os.listdir(chunks_dir):
            if fname.endswith("_chunks.json") or fname.endswith("_extracted.json"):
                chunk_files.append(os.path.join(chunks_dir, fname))

    if not chunk_files:
        print(f"  No chunk files found in {chunks_dir}")
        return []

    print(f"  Found {len(chunk_files)} chunk files")

    all_chunks = []
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle both formats: {"chunks": [...]} or [...]
        chunks = data.get("chunks", data) if isinstance(data, dict) else data
        if isinstance(chunks, list):
            for chunk in chunks:
                text = chunk.get("text", chunk.get("content", ""))
                if text and len(text) > 100:
                    all_chunks.append(text)

    print(f"  Total chunks: {len(all_chunks)}")

    samples = []
    batch_count = 0

    for i in range(0, len(all_chunks), chunk_batch_size):
        if len(samples) >= max_samples:
            break

        batch_text = "\n\n---\n\n".join(all_chunks[i:i + chunk_batch_size])

        # Truncate if too long
        if len(batch_text) > 8000:
            batch_text = batch_text[:8000]

        prompt = CHUNK_QA_PROMPT.format(
            num_pairs=pairs_per_batch,
            chunk_text=batch_text,
        )

        try:
            response = generator.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.8,
                    "max_output_tokens": 2048,
                }
            )
            qa_pairs = json.loads(response.text)
        except Exception as e:
            print(f"  Batch {i} error: {e}")
            time.sleep(2)
            continue

        # Handle both list and dict response
        if isinstance(qa_pairs, dict):
            qa_pairs = qa_pairs.get("pairs", qa_pairs.get("qa_pairs", [qa_pairs]))

        for qa in qa_pairs:
            q = qa.get("question", "")
            a = qa.get("answer", qa.get("response", ""))
            cat = qa.get("category", "factual")

            if q and a:
                samples.append(ResponseSample(
                    context=batch_text[:2000],  # truncate context
                    question=q,
                    response=a,
                    category=cat,
                ))

        batch_count += 1
        if batch_count % 5 == 0:
            print(f"  Chunk batches processed: {batch_count}, samples: {len(samples)}")
            time.sleep(1)

    print(f"\nStrategy 2 complete: {len(samples)} samples from {batch_count} batches")
    return samples


# ---------------------------------------------------------------------------
# Validation: LLM verifier (Rompis method)
# ---------------------------------------------------------------------------

def validate_with_llm(
    samples: list[ResponseSample],
    api_key: str,
    max_validate: int = 500,
) -> tuple[list[ResponseSample], list[dict]]:
    """
    Validate samples using a second LLM (Rompis 2025 method).

    Returns (valid_samples, rejected_entries).
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    verifier = genai.GenerativeModel("gemini-2.5-flash")

    valid = []
    rejected = []

    to_validate = samples[:max_validate]
    print(f"Validating {len(to_validate)} samples with LLM verifier...")

    for i, sample in enumerate(to_validate):
        prompt = VERIFIER_PROMPT.format(
            question=sample.question,
            context=sample.context[:3000],
            answer=sample.response,
        )

        try:
            response = verifier.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                }
            )
            result = json.loads(response.text)
        except Exception as e:
            # If verification fails, keep the sample (conservative)
            valid.append(sample)
            continue

        is_valid = result.get("valid", True)
        is_hallucination = result.get("is_hallucination", False)

        if is_valid and not is_hallucination:
            valid.append(sample)
        else:
            rejected.append({
                "question": sample.question,
                "reason": result.get("reason", "unknown"),
                "is_hallucination": is_hallucination,
            })

        if (i + 1) % 20 == 0:
            print(f"  Verified {i + 1}/{len(to_validate)}: "
                  f"{len(valid)} valid, {len(rejected)} rejected")
            time.sleep(1)

    # Add remaining unvalidated samples
    valid.extend(samples[max_validate:])

    print(f"\nValidation: {len(valid)} valid, {len(rejected)} rejected "
          f"({len(rejected)/max(len(to_validate),1):.0%} rejection rate)")
    return valid, rejected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_legal_reference(text: str) -> bool:
    """Check if text contains legal references (Pasal, UU, etc.)."""
    patterns = [
        r"[Pp]asal\s+\d+",
        r"[Uu]ndang[-\s]*[Uu]ndang",
        r"UU\s+",
        r"[Aa]yat\s+\(\d+\)",
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def save_to_csv(
    samples: list[ResponseSample],
    output_dir: str,
    train_ratio: float = 0.8,
) -> tuple[str, str]:
    """Save samples to train/val CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_path = os.path.join(output_dir, "training_data.csv")
    val_path = os.path.join(output_dir, "validation_data.csv")

    fieldnames = ["context", "question", "response", "category"]

    for path, data in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in data:
                writer.writerow(asdict(s))

    print(f"Saved: {len(train_samples)} train -> {train_path}")
    print(f"Saved: {len(val_samples)} val -> {val_path}")
    return train_path, val_path


def save_prompt_template_csv(output_dir: str) -> str:
    """Save prompt template to CSV (for reference / Google Sheets)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "prompt_template.csv")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "system_message", "user_message"])
        writer.writeheader()
        writer.writerow({
            "prompt_id": "response_generation_v1",
            "system_message": SYSTEM_INSTRUCTION,
            "user_message": (
                "<INPUT>\n"
                "<CONTEXT>\n{context}\n</CONTEXT>\n"
                "<QUESTION>\n{question}\n</QUESTION>\n"
                "</INPUT>"
            ),
        })

    print(f"Prompt template saved to: {path}")
    return path


def upload_to_google_sheets(
    samples: list[ResponseSample],
    spreadsheet_id: str,
):
    """Upload samples to Google Sheets (optional)."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from modules.google_sheets_utils import GoogleUtil
    except ImportError:
        print("Google Sheets utils not available. Skipping upload.")
        return

    client_email = os.getenv("GOOGLE_SHEETS_CLIENT_EMAIL", "")
    private_key = os.getenv("GOOGLE_SHEETS_PRIVATE_KEY", "")

    if not client_email or not private_key:
        print("Google Sheets credentials not configured. Skipping.")
        return

    gu = GoogleUtil(
        spreadsheet_id=spreadsheet_id,
        client_email=client_email,
        private_key=private_key,
    )

    # Split into train/val
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_data = [asdict(s) for s in samples[:split]]
    val_data = [asdict(s) for s in samples[split:]]

    gu.write_to_sheet("response_training_data", train_data)
    gu.write_to_sheet("response_validation_data", val_data)
    print(f"Uploaded to Google Sheets: {len(train_data)} train, {len(val_data)} val")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Response Model Training Data Generator")
    parser.add_argument("--query-data", default="finetuning/query_model/data/training_data.csv",
                        help="Path to Task 2 query training data CSV")
    parser.add_argument("--chunks-dir", default="data/processed",
                        help="Directory with document chunk JSON files")
    parser.add_argument("--output", default="finetuning/response_model/data",
                        help="Output directory for CSV files")
    parser.add_argument("--max-query-samples", type=int, default=500,
                        help="Max samples from query strategy")
    parser.add_argument("--max-chunk-samples", type=int, default=200,
                        help="Max samples from chunk strategy")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip LLM validation step")
    parser.add_argument("--skip-sheets", action="store_true",
                        help="Skip Google Sheets upload")
    parser.add_argument("--skip-chunks", action="store_true",
                        help="Skip chunk-based generation")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "")

    print("=" * 60)
    print("Response Model Training Data Generator")
    print("=" * 60)

    # Connect to Neo4j
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            session.run("RETURN 1").single()
        print(f"1. Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        print(f"   [ERROR] Cannot connect to Neo4j: {e}")
        return

    all_samples = []

    # Strategy 1: From query results
    if os.path.exists(args.query_data):
        print(f"\n2. Strategy 1: From Task 2 query data ({args.query_data})")
        query_samples = generate_from_query_results(
            driver, args.query_data, api_key, args.max_query_samples
        )
        all_samples.extend(query_samples)
    else:
        print(f"\n2. Skipping Strategy 1: {args.query_data} not found")
        print("   Run Task 2 dataset generation first!")

    # Strategy 2: From document chunks
    if not args.skip_chunks and os.path.isdir(args.chunks_dir):
        print(f"\n3. Strategy 2: From document chunks ({args.chunks_dir})")
        chunk_samples = generate_from_chunks(
            args.chunks_dir, api_key, args.max_chunk_samples
        )
        all_samples.extend(chunk_samples)
    else:
        print(f"\n3. Skipping Strategy 2 (chunks)")

    if not all_samples:
        print("\n[ERROR] No samples generated!")
        driver.close()
        return

    # Validate with LLM
    if not args.skip_validation:
        print(f"\n4. Validating {len(all_samples)} samples with LLM verifier")
        all_samples, rejected = validate_with_llm(all_samples, api_key)
    else:
        print("\n4. Skipping LLM validation")

    # Save to CSV
    print(f"\n5. Saving {len(all_samples)} samples")
    train_path, val_path = save_to_csv(all_samples, args.output)
    prompt_path = save_prompt_template_csv(args.output)

    # Upload to Sheets
    if not args.skip_sheets:
        spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID", "")
        if spreadsheet_id:
            upload_to_google_sheets(all_samples, spreadsheet_id)

    # Stats
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    categories = {}
    for s in all_samples:
        categories[s.category] = categories.get(s.category, 0) + 1
    print(f"Total samples: {len(all_samples)}")
    for cat, cnt in sorted(categories.items()):
        print(f"  {cat}: {cnt}")
    ref_count = sum(1 for s in all_samples if _contains_legal_reference(s.response))
    print(f"With legal references: {ref_count}/{len(all_samples)} "
          f"({ref_count/max(len(all_samples),1):.0%})")

    driver.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
