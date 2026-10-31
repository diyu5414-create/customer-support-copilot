
"""
Step 2: Retrieval Evaluation

Calculates:
- Precision@5
- Recall@5
- F1@5
- Coverage
- Routing Accuracy
- Refusal Accuracy
- False Refusals

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


def load_json(path):
    """Load JSON data from a file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_retrieved_ids(result, k=TOP_K):
    """Return unique retrieved document IDs in retrieval order."""
    ids = result.get("retrieved_document_ids")

    if ids is None:
        ids = result.get("sources", [])

    unique_ids = []

    for item in ids or []:
        if isinstance(item, dict):
            item = (
                item.get("id")
                or item.get("source_id")
                or item.get("document_id")
            )

        if item is not None:
            item = str(item)

            if item not in unique_ids:
                unique_ids.append(item)

    return unique_ids[:k]


def evaluate_retrieval(result, k=TOP_K):
    """Calculate Precision@K, Recall@K, and F1@K for one question."""
    retrieved_sources = set(get_retrieved_ids(result, k))

    expected_sources = {
        str(source)
        for source in result.get("expected_sources", [])
    }

    if not expected_sources:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "coverage": None,
        }

    true_positives = len(
        retrieved_sources.intersection(expected_sources)
    )

    precision = (
        true_positives / len(retrieved_sources)
        if retrieved_sources
        else 0.0
    )

    recall = true_positives / len(expected_sources)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    # Coverage for this question: at least one expected source retrieved.
    coverage = (
        1.0
        if true_positives > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "coverage": coverage,
    }


def is_refusal(answer):
    """Check whether the generated answer is a refusal."""
    normalized = str(answer or "").lower().replace("’", "'")
    refusal_phrase = "i don't know based on the available information"
    return refusal_phrase in normalized


def check_refusal_accuracy(result):
    """Check whether the answer correctly handles in-scope/out-of-scope questions."""
    expected_sources = result.get("expected_sources", [])
    should_refuse = len(expected_sources) == 0
    actual_refusal = is_refusal(result.get("answer", ""))

    if should_refuse and actual_refusal:
        return "Correct refusal"

    if should_refuse and not actual_refusal:
        return "Should have refused but didn't"

    if not should_refuse and actual_refusal:
        return "False refusal"

    return "Correctly answered"


def evaluate_routing(result):
    """
    Compare expected indexes with the indexes selected by the router.
    Exact set match is considered correct.
    """
    expected_indexes = set(result.get("expected_indexes", []))
    actual_indexes = set(
        result.get("routing", {}).get("indexes", [])
    )

    return expected_indexes == actual_indexes


def evaluate_version(results, version_name):
    """Calculate aggregated retrieval, coverage, routing, and refusal metrics."""
    in_scope = [
        result
        for result in results
        if result.get("expected_sources")
    ]

    per_question = [
        evaluate_retrieval(result)
        for result in in_scope
    ]

    precision_values = [
        metric["precision"]
        for metric in per_question
        if metric["precision"] is not None
    ]

    recall_values = [
        metric["recall"]
        for metric in per_question
        if metric["recall"] is not None
    ]

    f1_values = [
        metric["f1"]
        for metric in per_question
        if metric["f1"] is not None
    ]

    coverage_values = [
        metric["coverage"]
        for metric in per_question
        if metric["coverage"] is not None
    ]

    refusal_checks = [
        check_refusal_accuracy(result)
        for result in results
    ]

    refusal_accuracy = (
        sum(
            status in (
                "Correct refusal",
                "Correctly answered",
            )
            for status in refusal_checks
        ) / len(results)
        if results
        else None
    )

    false_refusals = refusal_checks.count("False refusal")

    # Routing accuracy is applicable to the Multi-Index version.
    routing_accuracy = None

    if "Multi-Index" in version_name:
        routing_checks = [
            evaluate_routing(result)
            for result in results
        ]

        routing_accuracy = (
            sum(routing_checks) / len(routing_checks)
            if routing_checks
            else None
        )

    return {
        "version": version_name,
        "question_count": len(results),
        "in_scope_count": len(in_scope),
        "out_of_scope_count": len(results) - len(in_scope),

        "precision_at_5": (
            mean(precision_values)
            if precision_values
            else None
        ),

        "recall_at_5": (
            mean(recall_values)
            if recall_values
            else None
        ),

        "f1_at_5": (
            mean(f1_values)
            if f1_values
            else None
        ),

        "coverage": (
            mean(coverage_values)
            if coverage_values
            else None
        ),

        "routing_accuracy": routing_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "false_refusals": false_refusals,

        "per_question": [
            {
                "id": result.get("id"),
                "question": result.get("question"),
                **evaluate_retrieval(result),
                "routing_correct": (
                    evaluate_routing(result)
                    if "Multi-Index" in version_name
                    else None
                ),
            }
            for result in results
        ],
    }


def format_metric(value):
    """Format a metric as a percentage."""
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def main():
    """Run retrieval evaluation for all three RAG versions."""
    golden = load_json(GOLDEN_TEST_SET)

    expected_ids = [
        question["id"]
        for question in golden
    ]

    all_metrics = []

    for version_name, result_path in RESULT_FILES.items():
        if not result_path.exists():
            raise FileNotFoundError(
                f"Result file not found: {result_path}"
            )

        results = load_json(result_path)

        actual_ids = [
            result.get("id")
            for result in results
        ]

        if actual_ids != expected_ids:
            raise ValueError(
                f"Question IDs/order mismatch in {result_path.name}"
            )

        metrics = evaluate_version(results, version_name)
        all_metrics.append(metrics)

    print("\nRETRIEVAL EVALUATION")
    print(
        "Version | Precision@5 | Recall@5 | F1@5 | Coverage "
        "| Routing Accuracy | Refusal Accuracy | False Refusals"
    )

    for item in all_metrics:
        print(
            " | ".join(
                [
                    item["version"],
                    format_metric(item["precision_at_5"]),
                    format_metric(item["recall_at_5"]),
                    format_metric(item["f1_at_5"]),
                    format_metric(item["coverage"]),
                    format_metric(item["routing_accuracy"]),
                    format_metric(item["refusal_accuracy"]),
                    str(item["false_refusals"]),
                ]
            )
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RESULTS_DIR / "retrieval_metrics.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
