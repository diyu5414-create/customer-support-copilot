# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


"""
Customer Support Copilot — Evaluation Runner

Runs all golden test questions through:
1. V1 Basic RAG
2. V2 Advanced Single-Index RAG
3. V2 Multi-Index RAG with Intent Routing

Saves standardized JSON results for later evaluation.
"""

import json
import sys
import shutil
import importlib.util
from pathlib import Path


# ==========================================================
# PART 1: DEFINE PROJECT PATHS
# ==========================================================

# Find the project root from this script's location.
# This works even if CMD is opened from another directory.
ROOT_DIR = Path(__file__).resolve().parent.parent

EVALUATION_DIR = ROOT_DIR / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
GOLDEN_FILE = EVALUATION_DIR / "golden_test_set.json"

# Make sure the results directory exists.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Allow project modules to find other project files.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ==========================================================
# PART 2: LOAD A PYTHON FILE AS A MODULE
# ==========================================================

def load_module(module_name, file_path):
    """
    Load a Python module from its file path.

    This is useful because our pipeline filenames begin
    with numbers, such as 2_online_pipeline.py.
    """

    # Confirm that the pipeline file exists.
    if not file_path.exists():
        raise FileNotFoundError(
            f"Pipeline file not found: {file_path}"
        )

    # Create a module specification for this file.
    spec = importlib.util.spec_from_file_location(
        module_name, file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {file_path}"
        )

    # Create the module object.
    module = importlib.util.module_from_spec(spec)

    # Register it before execution for reliable module loading.
    sys.modules[module_name] = module

    # Execute the Python file and initialize its functions.
    spec.loader.exec_module(module)

    return module


# ==========================================================
# PART 3: NORMALIZE DOCUMENT IDS
# ==========================================================

def unique_document_ids(document_ids, limit=5):
    """
    Remove duplicate document IDs while preserving their order.
    Return no more than the requested number of IDs.
    """

    unique_ids = []

    for document_id in document_ids or []:
        # Some pipelines may return strings, while others
        # may return dictionaries containing an ID.
        if isinstance(document_id, dict):
            document_id = (
                document_id.get("id")
                or document_id.get("source_id")
                or document_id.get("document_id")
            )

        if document_id is None:
            continue

        document_id = str(document_id)

        if document_id not in unique_ids:
            unique_ids.append(document_id)

        if len(unique_ids) == limit:
            break

    return unique_ids


# ==========================================================
# PART 4: STANDARDIZE EACH QUESTION'S RESULT
# ==========================================================

def standardize_result(
    question_data,
    answer,
    retrieved_ids,
    sources=None,
    routing=None,
):
    """
    Use the same result structure for all three RAG versions.
    Keeping expected sources and indexes helps later metrics.
    """

    retrieved_ids = unique_document_ids(retrieved_ids)

    if sources is None:
        sources = retrieved_ids

    return {
        "id": question_data["id"],
        "question": question_data["question"],
        "reference_answer": question_data.get(
            "reference_answer", ""
        ),
        "expected_sources": question_data.get(
            "expected_sources", []
        ),
        "expected_indexes": question_data.get(
            "expected_indexes", []
        ),
        "answer": answer,
        "retrieved_document_ids": retrieved_ids,
        "sources": sources,
        "routing": routing,
    }


# ==========================================================
# PART 5: SAVE RESULTS TO JSON
# ==========================================================

def save_results(filename, results):
    """
    Save the results collected so far.
    Saving after every question preserves completed work.
    """

    output_path = RESULTS_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved {len(results)} questions: {output_path.name}"
    )


# ==========================================================
# PART 6: BACK UP EXISTING MULTI-INDEX RESULTS
# ==========================================================

def backup_existing_results():
    """
    Preserve the previous multi-index result file before
    replacing it with results from all 20 questions.
    """

    old_file = RESULTS_DIR / "v2_multi_index_results.json"
    backup_file = (
        RESULTS_DIR / "v2_multi_index_results_previous.json"
    )

    if old_file.exists() and not backup_file.exists():
        shutil.copy2(old_file, backup_file)
        print(
            "Existing multi-index results backed up to:",
            backup_file.name,
        )


# ==========================================================
# PART 7: RUN ALL THREE PIPELINES
# ==========================================================

def main():

    # -----------------------------------------------
    # Step 7A: Load and validate the golden test set
    # -----------------------------------------------

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Golden test set not found: {GOLDEN_FILE}"
        )

    with GOLDEN_FILE.open("r", encoding="utf-8") as file:
        test_questions = json.load(file)

    if not isinstance(test_questions, list):
        raise ValueError(
            "golden_test_set.json must contain a list of questions."
        )

    if len(test_questions) != 20:
        raise ValueError(
            f"Expected 20 questions, found {len(test_questions)}."
        )

    print(f"Loaded {len(test_questions)} golden test questions.")

    # Back up the existing multi-index result file, if needed.
    backup_existing_results()

    # -----------------------------------------------
    # Step 7B: Load the existing pipeline modules
    # -----------------------------------------------

    print("\nLoading V1 Basic RAG...")
    v1 = load_module(
        "v1_online_pipeline",
        ROOT_DIR / "v1_basic_rag" / "2_online_pipeline.py",
    )

    print("\nLoading V2 Single-Index RAG...")
    v2_single = load_module(
        "v2_single_pipeline",
        ROOT_DIR / "v2_advanced_rag" / "5_v2_pipeline.py",
    )

    print("\nLoading V2 Multi-Index RAG...")
    v2_multi = load_module(
        "v2_multi_pipeline",
        ROOT_DIR / "v2_multi_index"
        / "4_v2_multi_index_pipeline.py",
    )

    # -----------------------------------------------
    # Step 7C: Create lists for the three result sets
    # -----------------------------------------------

    v1_results = []
    v2_single_results = []
    v2_multi_results = []

    # -----------------------------------------------
    # Step 7D: Evaluate every question
    # -----------------------------------------------

    for number, item in enumerate(test_questions, start=1):

        question = item["question"]

        print("\n" + "=" * 70)
        print(f"QUESTION {number}/20 — {item['id']}")
        print(question)
        print("=" * 70)

        # ==========================================
        # VERSION 1: BASIC RAG
        # ==========================================

        print("\nRunning V1 Basic RAG...")

        # Each test question must be independent.
        # Otherwise, previous questions may influence V1.
        v1.chat_history.clear()

        # V1 returns a dictionary containing answer and sources.
        result_v1 = v1.chat(question)

        v1_results.append(
            standardize_result(
                question_data=item,
                answer=result_v1["answer"],
                retrieved_ids=result_v1.get("sources", []),
                sources=result_v1.get("sources", []),
                routing=None,
            )
        )

        # ==========================================
        # VERSION 2: ADVANCED SINGLE-INDEX RAG
        # ==========================================

        print("\nRunning V2 Single-Index RAG...")

        # This pipeline returns (retrieved_results, answer).
        results_v2, answer_v2 = v2_single.v2_pipeline(question)

        # Each retrieved tuple has:
        # (document ID, metadata, text, reranking score)
        ids_v2 = [
            result[0] for result in results_v2
        ]

        v2_single_results.append(
            standardize_result(
                question_data=item,
                answer=answer_v2,
                retrieved_ids=ids_v2,
                sources=ids_v2,
                routing=None,
            )
        )

        # ==========================================
        # VERSION 3: MULTI-INDEX RAG WITH ROUTING
        # ==========================================

        print("\nRunning V2 Multi-Index RAG...")

        # The multi-index pipeline returns a dictionary.
        result_multi = v2_multi.run_pipeline(question)

        v2_multi_results.append(
            standardize_result(
                question_data=item,
                answer=result_multi["answer"],
                retrieved_ids=result_multi.get(
                    "retrieved_ids", []
                ),
                sources=result_multi.get("sources", []),
                routing=result_multi.get("routing"),
            )
        )

        # ==========================================
        # SAVE PROGRESS AFTER EACH QUESTION
        # ==========================================

        save_results("v1_results.json", v1_results)

        save_results(
            "v2_single_index_results.json",
            v2_single_results,
        )

        save_results(
            "v2_multi_index_results.json",
            v2_multi_results,
        )

    # ======================================================
    # PART 8: FINAL SUMMARY
    # ======================================================

    print("\n" + "=" * 70)
    print("EVALUATION RUN COMPLETED")
    print("=" * 70)

    print("Golden questions:", len(test_questions))
    print("V1 results:", len(v1_results))
    print("V2 single-index results:", len(v2_single_results))
    print("V2 multi-index results:", len(v2_multi_results))

    print("\nResult files:")
    print(RESULTS_DIR / "v1_results.json")
    print(RESULTS_DIR / "v2_single_index_results.json")
    print(RESULTS_DIR / "v2_multi_index_results.json")


# Run the evaluation only when this script is executed directly.
if __name__ == "__main__":
    main()
