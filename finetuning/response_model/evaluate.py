"""
Response Model Evaluation
=========================
Evaluate fine-tuned response model using:
1. LLM-as-Judge (Gemini, 0-2 score following Rompis 2025)
2. Reference accuracy (legal reference check)
3. Hallucination detection
4. BERTScore F1

Usage (CLI):
    python finetuning/response_model/evaluate.py \
        --test-data finetuning/response_model/data/validation_data.csv \
        --api-base http://localhost:8000/v1 \
        --model-name Qwen/Qwen3-4b

Usage (notebook):
    from finetuning.response_model.evaluate import evaluate
"""

import os
import csv
import json
import time
import re
import argparse
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# LLM Judge prompt (Rompis 2025 Section 3.8)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """Evaluasi jawaban model terhadap jawaban referensi untuk pertanyaan hukum Indonesia.

<QUESTION>
{question}
</QUESTION>

<REFERENCE_ANSWER>
{expected}
</REFERENCE_ANSWER>

<MODEL_ANSWER>
{predicted}
</MODEL_ANSWER>

Berikan evaluasi dalam format JSON:
{{
    "score": 0 | 1 | 2,
    "reason": "penjelasan singkat",
    "has_correct_reference": true/false,
    "is_hallucination": true/false
}}

Skala skor:
- 0: Salah, tidak responsif, atau sepenuhnya hallucination
- 1: Sebagian benar, ada ketidakakuratan atau elemen penting yang hilang
- 2: Benar dan lengkap, referensi hukum sesuai"""


# System instruction for inference
SYSTEM_INSTRUCTION = """Anda adalah asisten hukum Indonesia. Jawab pertanyaan pengguna berdasarkan
data dari Knowledge Graph hukum Indonesia yang disediakan.
Aturan:
1. Selalu sertakan referensi pasal dan UU yang relevan
2. Jika ada amandemen terkait, sebutkan versi terbaru
3. Jika informasi tidak cukup, jawab "Informasi tidak tersedia dalam Knowledge Graph"
4. Gunakan bahasa Indonesia formal"""


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    predicted_answer: str
    category: str
    llm_judge_score: int        # 0=salah, 1=parsial, 2=benar
    has_reference: bool         # menyebut pasal/UU?
    is_hallucination: bool      # info dibuat-buat?
    bert_score_f1: float        # BERTScore F1
    inference_time_ms: float    # waktu inference


def _contains_legal_reference(text: str) -> bool:
    """Check if text contains legal references."""
    patterns = [
        r"[Pp]asal\s+\d+",
        r"[Uu]ndang[-\s]*[Uu]ndang",
        r"UU\s+",
        r"[Aa]yat\s+\(\d+\)",
    ]
    return any(re.search(p, text) for p in patterns)


def evaluate(
    test_data_path: str,
    api_base: str,
    api_key: str,
    model_name: str,
    judge_api_key: str = "",
    output_dir: str = "",
) -> dict:
    """
    Evaluate response model.

    Args:
        test_data_path: Path to validation CSV
        api_base: OpenAI-compatible API base URL
        api_key: API key for model server
        model_name: Model name for inference
        judge_api_key: Gemini API key for LLM judge
        output_dir: Directory for output files
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base)

    # Load test data
    with open(test_data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_samples = list(reader)

    print(f"Evaluating {len(test_samples)} samples...")
    print(f"Model: {model_name}")
    print(f"API: {api_base}")

    if not output_dir:
        output_dir = os.path.dirname(test_data_path)

    results = []
    all_predicted = []
    all_expected = []

    for i, sample in enumerate(test_samples):
        context = sample.get("context", "")
        question = sample.get("question", "")
        expected = sample.get("response", "")
        category = sample.get("category", "factual")

        # Inference
        user_msg = (
            f"<INPUT>\n<CONTEXT>\n{context}\n</CONTEXT>\n"
            f"<QUESTION>\n{question}\n</QUESTION>\n</INPUT>"
        )

        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            predicted = response.choices[0].message.content.strip()
        except Exception as e:
            predicted = f"[ERROR] {e}"
        inference_ms = (time.time() - start_time) * 1000

        all_predicted.append(predicted)
        all_expected.append(expected)

        results.append(EvalResult(
            question=question,
            expected_answer=expected,
            predicted_answer=predicted,
            category=category,
            llm_judge_score=0,
            has_reference=_contains_legal_reference(predicted),
            is_hallucination=False,
            bert_score_f1=0.0,
            inference_time_ms=inference_ms,
        ))

        if (i + 1) % 10 == 0:
            print(f"  Inference: {i + 1}/{len(test_samples)}")

    # --- LLM Judge ---
    if judge_api_key:
        print("\nRunning LLM Judge evaluation...")
        import google.generativeai as genai
        genai.configure(api_key=judge_api_key)
        judge = genai.GenerativeModel("gemini-2.5-flash")

        for i, result in enumerate(results):
            if result.predicted_answer.startswith("[ERROR]"):
                result.llm_judge_score = 0
                result.is_hallucination = True
                continue

            prompt = JUDGE_PROMPT.format(
                question=result.question,
                expected=result.expected_answer,
                predicted=result.predicted_answer,
            )

            try:
                judge_response = judge.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                    }
                )
                judge_data = json.loads(judge_response.text)
                result.llm_judge_score = int(judge_data.get("score", 0))
                result.is_hallucination = judge_data.get("is_hallucination", False)
            except Exception as e:
                print(f"  Judge error for sample {i}: {e}")
                time.sleep(2)

            if (i + 1) % 20 == 0:
                print(f"  Judge: {i + 1}/{len(results)}")
                time.sleep(1)
    else:
        print("\nSkipping LLM Judge (no judge_api_key provided)")

    # --- BERTScore ---
    try:
        from bert_score import score as bert_score_fn
        print("\nCalculating BERTScore...")
        P, R, F1 = bert_score_fn(all_predicted, all_expected, lang="id", verbose=False)
        for i, result in enumerate(results):
            result.bert_score_f1 = F1[i].item()
    except ImportError:
        print("\nbert-score not installed. Skipping BERTScore.")
    except Exception as e:
        print(f"\nBERTScore error: {e}")

    # --- Compute metrics ---
    n = len(results)
    metrics = {
        "total_samples": n,
        "llm_judge_avg": sum(r.llm_judge_score for r in results) / max(n, 1),
        "reference_accuracy": sum(r.has_reference for r in results) / max(n, 1),
        "hallucination_rate": sum(r.is_hallucination for r in results) / max(n, 1),
        "bert_score_f1_avg": sum(r.bert_score_f1 for r in results) / max(n, 1),
        "avg_inference_ms": sum(r.inference_time_ms for r in results) / max(n, 1),
    }

    # Per-category breakdown
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    metrics["per_category"] = {}
    for cat, cat_results in categories.items():
        cn = len(cat_results)
        metrics["per_category"][cat] = {
            "count": cn,
            "llm_judge_avg": sum(r.llm_judge_score for r in cat_results) / max(cn, 1),
            "reference_accuracy": sum(r.has_reference for r in cat_results) / max(cn, 1),
            "hallucination_rate": sum(r.is_hallucination for r in cat_results) / max(cn, 1),
        }

    # --- Print results ---
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  LLM Judge avg:       {metrics['llm_judge_avg']:.2f} / 2.00 (target >= 1.30)")
    print(f"  Reference Accuracy:  {metrics['reference_accuracy']:.1%} (target >= 90%)")
    print(f"  Hallucination Rate:  {metrics['hallucination_rate']:.1%} (target <= 10%)")
    print(f"  BERTScore F1 avg:    {metrics['bert_score_f1_avg']:.3f} (target >= 0.85)")
    print(f"  Avg Inference:       {metrics['avg_inference_ms']:.0f}ms")

    print(f"\nPer-category:")
    for cat, data in metrics["per_category"].items():
        print(f"  {cat}: {data['count']} samples, "
              f"judge={data['llm_judge_avg']:.2f}, "
              f"ref={data['reference_accuracy']:.0%}, "
              f"hall={data['hallucination_rate']:.0%}")

    # --- Save outputs ---
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    details_path = os.path.join(output_dir, "evaluation_details.csv")
    with open(details_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "expected_answer", "predicted_answer", "category",
            "llm_judge_score", "has_reference", "is_hallucination",
            "bert_score_f1", "inference_time_ms",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\nSaved: {results_path}")
    print(f"Saved: {details_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate response generation model")
    parser.add_argument("--test-data", required=True, help="Path to validation CSV")
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="not-needed", help="API key for model server")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4b", help="Model name")
    parser.add_argument("--judge-api-key", default="", help="Gemini API key for LLM judge")
    parser.add_argument("--output-dir", default="", help="Output directory")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    if not args.judge_api_key:
        args.judge_api_key = os.getenv("GEMINI_API_KEY", "")

    evaluate(
        test_data_path=args.test_data,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
        judge_api_key=args.judge_api_key,
        output_dir=args.output_dir or os.path.dirname(args.test_data),
    )


if __name__ == "__main__":
    main()
