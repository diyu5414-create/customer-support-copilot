
"""Create a comparison table for all RAG evaluation metrics."""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

OUTPUT_FILE = BASE_DIR / "evaluation" / "comparison_table.csv"
RETRIEVAL_FILE = RESULTS_DIR / "retrieval_metrics.json"
GENERATION_FILE = RESULTS_DIR / "generation_metrics.json"

VERSIONS = [
    ("V1 Basic RAG", "v1_basic_rag"),
    ("V2 Single-Index RAG", "v2_single_index_rag"),
    ("V2 Multi-Index RAG", "v2_multi_index_rag"),
]


def load_json(path):
    """Load JSON data and raise an error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_retrieval(data, version_name):
    """Find retrieval metrics for a specific RAG version."""
    for item in data:
        if item.get("version") == version_name:
            return item

    raise ValueError(
        f"Retrieval metrics missing for {version_name}"
    )


def format_percent(value):
    """Format a ratio as a percentage."""
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def main():
    """Generate and print the consolidated comparison table."""
    retrieval_data = load_json(RETRIEVAL_FILE)
    generation_data = load_json(GENERATION_FILE).get("versions", {})

    metric_rows = [
        # Retrieval metrics
        ("Precision@5", "precision_at_5", "retrieval", "ratio"),
        ("Recall@5", "recall_at_5", "retrieval", "ratio"),
        ("F1@5", "f1_at_5", "retrieval", "ratio"),
        ("Coverage", "coverage", "retrieval", "ratio"),
        ("Routing Accuracy", "routing_accuracy", "retrieval", "ratio"),

        # Generation metrics
        (
            "Faithfulness (LLM Judge)",
            "faithful_percentage",
            "faithfulness",
            "percent",
        ),

        # Refusal metrics
        (
            "Refusal Accuracy",
            "accuracy_percentage",
            "refusal_accuracy",
            "percent",
        ),
        (
            "False Refusals",
            "false_refusals",
            "refusal_accuracy",
            "integer",
        ),
    ]

    rows = []

    for label, key, source, value_type in metric_rows:
        row = {"Metric": label}

        for display_name, generation_key in VERSIONS:
            retrieval = find_retrieval(
                retrieval_data,
                display_name,
            )

            generation = generation_data.get(
                generation_key,
                {},
            ).get("summary", {})

            if source == "retrieval":
                value = retrieval.get(key)
            else:
                value = generation.get(source, {}).get(key)

            if value_type == "ratio":
                row[display_name] = format_percent(value)

            elif value_type == "percent":
                row[display_name] = (
                    "N/A"
                    if value is None
                    else f"{float(value):.2f}%"
                )

            else:
                row[display_name] = str(
                    value if value is not None else "N/A"
                )

        rows.append(row)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "Metric",
        *[version[0] for version in VERSIONS],
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nComparison table created: {OUTPUT_FILE}\n")

    for row in rows:
        print(
            f"{row['Metric']} | "
            + " | ".join(
                row[version[0]]
                for version in VERSIONS
            )
        )


if __name__ == "__main__":
    main()
