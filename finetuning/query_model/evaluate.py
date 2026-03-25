"""
Evaluation Script for NL → Cypher Query Model.

Evaluates a fine-tuned model's ability to generate valid Cypher queries
from natural language questions about Indonesian legal KG.

Follows patterns from sft_inference.ipynb reference:
- Load test data from CSV or Google Sheets
- Inference via OpenAI-compatible API (vLLM / local / cloud)
- 4 metrics: syntax validity, execution success, result accuracy, exact match

Usage:
    python finetuning/query_model/evaluate.py \
        --test-data finetuning/query_model/data/validation_data.csv \
        --api-base http://localhost:8000/v1 \
        --model-name "Qwen/Qwen3-4b-finetuned"

Or with Neo4j for execution testing:
    python finetuning/query_model/evaluate.py \
        --test-data finetuning/query_model/data/validation_data.csv \
        --api-base http://localhost:8000/v1 \
        --neo4j-uri bolt://localhost:7687
"""

import json
import os
import re
import csv
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class EvalResult:
    """Result of evaluating a single NL-Cypher pair."""
    question: str
    expected_cypher: str
    predicted_cypher: str
    category: str
    syntax_valid: bool
    execution_success: bool
    result_correct: bool
    exact_match: bool
    inference_time_ms: float


# ============================================================
# Inference
# ============================================================

def get_system_prompt():
    """Get the system prompt for inference (same as training)."""
    from finetuning.query_model.generate_training_data import SYSTEM_INSTRUCTION, KG_SCHEMA
    return SYSTEM_INSTRUCTION


def infer_cypher(
    client,
    model_name: str,
    context: str,
    question: str,
    system_prompt: str,
) -> tuple[str, float]:
    """Run inference to generate Cypher from NL question.

    Returns (predicted_cypher, inference_time_ms).
    """
    user_message = f"<INPUT>\n<CONTEXT>\n{context}\n</CONTEXT>\n<QUESTION>\n{question}\n</QUESTION>\n</INPUT>"

    start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    elapsed_ms = (time.time() - start) * 1000

    predicted = response.choices[0].message.content.strip()

    # Clean up: extract Cypher from markdown code blocks if present
    cypher_match = re.search(r"```(?:cypher)?\s*(.*?)```", predicted, re.DOTALL)
    if cypher_match:
        predicted = cypher_match.group(1).strip()

    return predicted, elapsed_ms


# ============================================================
# Cypher comparison
# ============================================================

def normalize_cypher(cypher: str) -> str:
    """Normalize Cypher for comparison: lowercase, remove whitespace."""
    s = re.sub(r"\s+", " ", cypher.strip().lower())
    s = re.sub(r"\s*([{}()\[\],:])\s*", r"\1", s)
    return s


def check_syntax(cypher: str, driver=None) -> bool:
    """Check Cypher syntax validity."""
    if driver:
        try:
            with driver.session() as session:
                session.run(f"EXPLAIN {cypher}")
            return True
        except Exception:
            return False
    # Basic heuristic if no driver
    cypher_lower = cypher.lower().strip()
    return (
        cypher_lower.startswith("match")
        or cypher_lower.startswith("return")
        or cypher_lower.startswith("call")
        or cypher_lower.startswith("with")
    )


def check_execution(cypher: str, driver) -> tuple[bool, Optional[list]]:
    """Execute Cypher and return (success, results)."""
    try:
        with driver.session() as session:
            result = session.run(cypher).data()
        return True, result
    except Exception:
        return False, None


def check_result_match(predicted_result, expected_result) -> bool:
    """Compare query results (order-independent)."""
    if predicted_result is None or expected_result is None:
        return False

    # Normalize: sort by all values
    def normalize_result(res):
        return sorted([tuple(sorted(r.items())) for r in res])

    try:
        return normalize_result(predicted_result) == normalize_result(expected_result)
    except Exception:
        return False


# ============================================================
# Main evaluation
# ============================================================

def evaluate(
    test_data_path: str,
    api_base: str,
    api_key: str,
    model_name: str,
    neo4j_uri: str = "",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "",
) -> dict:
    """Run full evaluation."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base)
    system_prompt = get_system_prompt()

    # Load test data
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_samples = list(reader)
    print(f"  → {len(test_samples)} test samples")

    # Optional: connect Neo4j for execution testing
    driver = None
    if neo4j_uri:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"  → Neo4j connected: {neo4j_uri}")

    # Evaluate each sample
    results = []
    for i, sample in enumerate(test_samples):
        question = sample["question"]
        expected = sample["response"]
        context = sample.get("context", "")
        category = sample.get("category", "")

        # Inference
        try:
            predicted, inference_ms = infer_cypher(
                client, model_name, context, question, system_prompt
            )
        except Exception as e:
            print(f"  [{i+1}] Inference error: {e}")
            results.append(EvalResult(
                question=question, expected_cypher=expected,
                predicted_cypher="", category=category,
                syntax_valid=False, execution_success=False,
                result_correct=False, exact_match=False,
                inference_time_ms=0,
            ))
            continue

        # Check syntax
        syntax_ok = check_syntax(predicted, driver)

        # Check execution
        exec_ok = False
        pred_result = None
        expected_result = None
        if driver and syntax_ok:
            exec_ok, pred_result = check_execution(predicted, driver)
            _, expected_result = check_execution(expected, driver)

        # Check result correctness
        result_ok = check_result_match(pred_result, expected_result)

        # Exact match
        exact = normalize_cypher(predicted) == normalize_cypher(expected)

        results.append(EvalResult(
            question=question,
            expected_cypher=expected,
            predicted_cypher=predicted,
            category=category,
            syntax_valid=syntax_ok,
            execution_success=exec_ok,
            result_correct=result_ok,
            exact_match=exact,
            inference_time_ms=inference_ms,
        ))

        status = "✅" if syntax_ok else "❌"
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_samples)}] {status} {question[:60]}")

    # Compute metrics
    n = len(results)
    metrics = {
        "total_samples": n,
        "syntax_validity": sum(r.syntax_valid for r in results) / n if n else 0,
        "execution_success": sum(r.execution_success for r in results) / n if n else 0,
        "result_accuracy": sum(r.result_correct for r in results) / n if n else 0,
        "exact_match": sum(r.exact_match for r in results) / n if n else 0,
        "avg_inference_ms": sum(r.inference_time_ms for r in results) / n if n else 0,
    }

    # Category breakdown
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"total": 0, "syntax": 0, "exec": 0, "correct": 0}
        categories[r.category]["total"] += 1
        categories[r.category]["syntax"] += int(r.syntax_valid)
        categories[r.category]["exec"] += int(r.execution_success)
        categories[r.category]["correct"] += int(r.result_correct)

    metrics["per_category"] = {
        cat: {k: v / data["total"] if k != "total" else v for k, v in data.items()}
        for cat, data in categories.items()
    }

    # Print report
    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Samples:            {metrics['total_samples']}")
    print(f"Syntax Validity:    {metrics['syntax_validity']:.1%} (target ≥ 95%)")
    print(f"Execution Success:  {metrics['execution_success']:.1%} (target ≥ 90%)")
    print(f"Result Accuracy:    {metrics['result_accuracy']:.1%} (target ≥ 80%)")
    print(f"Exact Match:        {metrics['exact_match']:.1%}")
    print(f"Avg Inference:      {metrics['avg_inference_ms']:.0f}ms (target < 3000ms)")

    if categories:
        print(f"\nPer-category breakdown:")
        for cat, data in sorted(categories.items()):
            n_cat = data["total"]
            print(f"  {cat}: syntax={data['syntax']/n_cat:.0%} exec={data['exec']/n_cat:.0%} correct={data['correct']/n_cat:.0%} (n={n_cat})")

    # Save results
    output_dir = os.path.dirname(test_data_path)
    metrics_path = os.path.join(output_dir, "evaluation_results.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    details_path = os.path.join(output_dir, "evaluation_details.csv")
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\nSaved: {metrics_path}")
    print(f"       {details_path}")

    if driver:
        driver.close()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate NL → Cypher query model")
    parser.add_argument("--test-data", required=True, help="Path to validation CSV")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="OpenAI-compatible API endpoint")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "not-needed"), help="API key")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4b", help="Model name for API")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", ""), help="Neo4j URI for exec testing")
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    args = parser.parse_args()

    evaluate(
        args.test_data, args.api_base, args.api_key, args.model_name,
        args.neo4j_uri, args.neo4j_user, args.neo4j_password,
    )


if __name__ == "__main__":
    main()
