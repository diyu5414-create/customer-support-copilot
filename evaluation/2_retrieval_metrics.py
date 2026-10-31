# INSTALLATION / RUN NOTES
# Run from the project root after installing requirements.txt.

"""
Step 2: Retrieval Evaluation

Follows the mentor's evaluation approach:
- Precision@K
- Recall@K
- F1
- Refusal accuracy

Routing accuracy is also reported for the assignment's multi-index pipeline.
The project's existing TOP_K selection is preserved at 5.

Run from the project root:
    python evaluation/2_retrieval_metrics.py
"""

import json
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
GOLDEN_TEST_SET = EVALUATION_DIR / "golden_test_set.json"

RESULT_FILES = {
    "V1 Basic RAG": RESULTS_DIR / "v1_results.json",
    "V2 Single-Index RAG": RESULTS_DIR / "v2_single_index_results.json",
    "V2 Multi-Index RAG": RESULTS_DIR / "v2_multi_index_results.json",
}

TOP_K = 5
REFUSAL_TEXT = "I don't know based on the available information."


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_retrieved_ids(result):
    """Return unique retrieved document IDs, preserving retrieval order."""
    ids = result.get("retrieved_document_ids")
    if ids is None:
        ids = result.get("sources", [])

    unique_ids = []
    for item in ids or []:
        if isinstance(item, dict):
            item = item.get("id") or item.get("source_id") or item.get("document_id")
        if item is not None and str(item) not in unique_ids:
            unique_ids.append(str(item))
    return unique_ids[:TOP_K]


def evaluate_retrieval(result, k=TOP_K):
    """Calculate Precision, Recall and F1 like the mentor's program."""
    retrieved_sources = set(get_retrieved_ids(result))
    expected_sources = set(result.get("expected_sources", []))

    if not expected_sources:
        return {"precision": None, "recall": None, "f1": None}

    true_positives = len(retrieved_sources & expected_sources)
    precision = true_positives / len(retrieved_sources) if retrieved_sources else 0
    recall = true_positives / len(expected_sources)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def is_refusal(answer):
    normalized = str(answer or "").lower().replace("’", "'")
    return "i don't know based on the available information" in normalized


def check_refusal_accuracy(result):
    should_refuse = len(result.get("expected_sources", [])) == 0
    actual_refusal = is_refusal(result.get("answer", ""))

    if should_refuse and actual_refusal:
        return "Correct refusal"
    if should_refuse and not actual_refusal:
        return "Should have refused but didn't"
    if not should_refuse and actual_refusal:
        return "False refusal"
    return "Correctly answered"


def calculate_routing_accuracy(result):
    """Check selected indexes against the golden expected indexes."""
    expected_sources = result.get("expected_sources", [])
    expected_indexes = set(result.get("expected_indexes", []))
    routing = result.get("routing") or {}
    actual_indexes = set(routing.get("indexes", []))
    intents = routing.get("intents", [])

    if not expected_sources and not expected_indexes:
        return "out_of_scope" in intents

    missing = expected_indexes - actual_indexes
    extra = actual_indexes - expected_indexes
    return not missing and len(extra) <= 1


def evaluate_version(results, version_name):
    in_scope = [r for r in results if r.get("expected_sources")]
    per_question = [evaluate_retrieval(r) for r in in_scope]

    precision_values = [m["precision"] for m in per_question if m["precision"] is not None]
    recall_values = [m["recall"] for m in per_question if m["recall"] is not None]
    f1_values = [m["f1"] for m in per_question if m["f1"] is not None]

    refusal_checks = [check_refusal_accuracy(r) for r in results]
    refusal_accuracy = sum(x in ("Correct refusal", "Correctly answered") for x in refusal_checks) / len(results) if results else None
    false_refusals = refusal_checks.count("False refusal")
    should_have_refused = refusal_checks.count("Should have refused but didn't")

    routing_accuracy = None
    if version_name == "V2 Multi-Index RAG":
        routing_checks = [calculate_routing_accuracy(r) for r in results]
        routing_accuracy = mean(routing_checks) if routing_checks else None

    return {
        "version": version_name,
        "question_count": len(results),
        "in_scope_count": len(in_scope),
        "out_of_scope_count": len(results) - len(in_scope),
        "precision_at_5": mean(precision_values) if precision_values else None,
        "recall_at_5": mean(recall_values) if recall_values else None,
        "f1_at_5": mean(f1_values) if f1_values else None,
        "routing_accuracy": routing_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "false_refusals": false_refusals,
        "should_have_refused": should_have_refused,
        "per_question": [
            {"id": r.get("id"), "question": r.get("question"), **evaluate_retrieval(r)}
            for r in in_scope
        ],
    }


def format_metric(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def main():
    golden = load_json(GOLDEN_TEST_SET)
    expected_ids = [q["id"] for q in golden]
    all_metrics = []

    for version_name, result_path in RESULT_FILES.items():
        if not result_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_path}")
        results = load_json(result_path)
        if [r.get("id") for r in results] != expected_ids:
            raise ValueError(f"Question IDs/order mismatch in {result_path.name}")
        all_metrics.append(evaluate_version(results, version_name))

    print("\nRETRIEVAL EVALUATION (MENTOR METHOD)")
    print("Version | Precision@5 | Recall@5 | F1@5 | Refusal Accuracy | Routing Accuracy")
    for item in all_metrics:
        print(" | ".join([
            item["version"], format_metric(item["precision_at_5"]),
            format_metric(item["recall_at_5"]), format_metric(item["f1_at_5"]),
            format_metric(item["refusal_accuracy"]), format_metric(item["routing_accuracy"]),
        ]))

    output_path = RESULTS_DIR / "retrieval_metrics.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
