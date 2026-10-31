# INSTALLATION / RUN NOTES
# Run from the project root after installing requirements.txt.

"""
Step 3: Generation Evaluation

Follows the mentor's evaluation method:
- Faithfulness (LLM-as-judge): Faithful / Partial / Unfaithful
- Refusal accuracy: correct refusal / should have refused / false refusal

Uses answers saved by evaluation/1_run_versions.py.
No RAGAS and no separate answer-correctness metric.

Run from the project root:
    python evaluation/3_generation_metrics.py
"""

import json
import re
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = BASE_DIR / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
DATASET_PATH = BASE_DIR / "data" / "dandes_electronics_kb.json"
GOLDEN_PATH = EVALUATION_DIR / "golden_test_set.json"
OUTPUT_PATH = RESULTS_DIR / "generation_metrics.json"

load_dotenv(BASE_DIR / ".env")
JUDGE_MODEL = os.getenv("JUDGE_MODEL")
if not JUDGE_MODEL:
    raise ValueError("JUDGE_MODEL is missing from your .env file.")

llm_brain = ChatOpenAI(model=JUDGE_MODEL)

VERSION_FILES = {
    "v1_basic_rag": RESULTS_DIR / "v1_results.json",
    "v2_single_index_rag": RESULTS_DIR / "v2_single_index_results.json",
    "v2_multi_index_rag": RESULTS_DIR / "v2_multi_index_results.json",
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_source_map():
    """Create source_id -> title and content mapping from the KB."""
    dataset = load_json(DATASET_PATH)
    documents = dataset.get("documents", []) if isinstance(dataset, dict) else []
    source_map = {}
    for document in documents:
        doc_id = document.get("id")
        if doc_id:
            source_map[str(doc_id)] = (
                f"Title: {document.get('title', '')}\n"
                f"Content: {document.get('content', '')}"
            )
    if not source_map:
        raise ValueError("No document IDs found in the electronics knowledge base.")
    return source_map


def get_retrieved_ids(result):
    ids = result.get("retrieved_document_ids") or result.get("sources") or []
    unique_ids = []
    for item in ids:
        if isinstance(item, dict):
            item = item.get("id") or item.get("source_id") or item.get("document_id")
        if item is not None and str(item) not in unique_ids:
            unique_ids.append(str(item))
    return unique_ids


def format_context(result, source_map):
    parts = []
    for source_id in get_retrieved_ids(result):
        content = source_map.get(source_id)
        if content:
            parts.append(f"[Source: {source_id}]\n{content}")
    return "\n\n".join(parts)


def is_refusal(answer):
    normalized = str(answer or "").lower().replace("’", "'")
    return "i don't know based on the available information" in normalized


def judge_faithfulness(context, generated_answer):
    """Ask the LLM judge whether every answer claim is grounded in context."""
    if is_refusal(generated_answer):
        return "N/A (refusal)"
    if not context.strip() or not generated_answer.strip():
        return "Unfaithful"

    judge_prompt = f"""Context:
{context}

Generated answer: {generated_answer}

Is every claim in the generated answer directly supported by the context above?
Respond with ONLY one word: Faithful, Partial, or Unfaithful.
"""
    response = llm_brain.invoke([HumanMessage(content=judge_prompt)])
    verdict = response.content.strip()
    match = re.search(r"\b(Faithful|Partial|Unfaithful)\b", verdict, re.IGNORECASE)
    return match.group(1).capitalize() if match else "Unfaithful"


def check_refusal_accuracy(expected_sources, generated_answer):
    """Use the mentor's four-way refusal check with this project's refusal text."""
    actual_refusal = is_refusal(generated_answer)
    should_refuse = len(expected_sources) == 0

    if should_refuse and actual_refusal:
        return "Correct refusal"
    if should_refuse and not actual_refusal:
        return "Should have refused but didn't (hallucination risk)"
    if not should_refuse and actual_refusal:
        return "False refusal (info existed but model refused)"
    return "Correctly answered (no refusal needed)"


def evaluate_version(version_name, result_path):
    results = load_json(result_path)
    evaluated = []
    for index, result in enumerate(results, start=1):
        answer = result.get("answer", "")
        expected_sources = result.get("expected_sources", [])
        context = format_context(result, SOURCE_MAP)
        faithfulness = judge_faithfulness(context, answer)
        refusal_check = check_refusal_accuracy(expected_sources, answer)
        item = {
            "id": result.get("id"),
            "question": result.get("question"),
            "answer": answer,
            "retrieved_sources": get_retrieved_ids(result),
            "faithfulness": faithfulness,
            "refusal_check": refusal_check,
        }
        evaluated.append(item)
        print(f"[{version_name}] {index}/{len(results)} {item['id']}: "
              f"Faithfulness={faithfulness} | Refusal={refusal_check}")
        save_progress(version_name, evaluated)
    return evaluated


def summarize(questions):
    faithfulness_values = [q["faithfulness"] for q in questions
                           if q["faithfulness"] in ("Faithful", "Partial", "Unfaithful")]
    refusal_values = [q["refusal_check"] for q in questions]
    refusal_correct = sum(
        value in ("Correct refusal", "Correctly answered (no refusal needed)")
        for value in refusal_values
    )
    return {
        "total_questions": len(questions),
        "faithfulness": {
            "faithful": faithfulness_values.count("Faithful"),
            "partial": faithfulness_values.count("Partial"),
            "unfaithful": faithfulness_values.count("Unfaithful"),
            "evaluated": len(faithfulness_values),
            "faithful_percentage": round(
                faithfulness_values.count("Faithful") / len(faithfulness_values) * 100, 2
            ) if faithfulness_values else None,
        },
        "refusal_accuracy": {
            "correct": refusal_correct,
            "incorrect": len(refusal_values) - refusal_correct,
            "evaluated": len(refusal_values),
            "accuracy_percentage": round(refusal_correct / len(refusal_values) * 100, 2)
                if refusal_values else None,
            "false_refusals": sum("False refusal" in value for value in refusal_values),
            "should_have_refused": sum("Should have refused" in value for value in refusal_values),
        },
    }


def save_progress(version_name, questions):
    if OUTPUT_PATH.exists():
        try:
            output = load_json(OUTPUT_PATH)
        except (json.JSONDecodeError, OSError):
            output = {}
    else:
        output = {}
    output.setdefault("versions", {})
    output["versions"][version_name] = {
        "questions": questions,
        "summary": summarize(questions),
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)


def main():
    golden = load_json(GOLDEN_PATH)
    SOURCE_MAP.update(build_source_map())
    expected_ids = [q["id"] for q in golden]

    for version_name, result_path in VERSION_FILES.items():
        results = load_json(result_path)
        if [r.get("id") for r in results] != expected_ids:
            raise ValueError(f"Question IDs/order mismatch in {result_path.name}")
        evaluate_version(version_name, result_path)

    final = load_json(OUTPUT_PATH)
    print("\nGENERATION EVALUATION SUMMARY (MENTOR METHOD)")
    for version_name, version in final["versions"].items():
        summary = version["summary"]
        print(f"\n{version_name}")
        print("Faithfulness:", summary["faithfulness"])
        print("Refusal accuracy:", summary["refusal_accuracy"])
    print(f"\nSaved: {OUTPUT_PATH}")


SOURCE_MAP = {}

if __name__ == "__main__":
    main()
