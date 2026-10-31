# INSTALLATION / RUN NOTES (Windows CMD)
# Activate .venv and install requirements.txt before running application scripts.
# Run this offline test file from the project root with:
# python -m unittest discover -s tests -v

"""Offline integrity tests for the Customer Support Copilot submission.

Run from the repository root with:
    python -m unittest discover -s tests -v
These tests do not call an LLM API or create a Chroma database.
"""
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectIntegrityTests(unittest.TestCase):
    def test_required_project_files_exist(self):
        required = [
            "README.md", "requirements.txt", ".gitignore",
            "data/dandes_electronics_kb.json",
            "evaluation/golden_test_set.json",
            "v1_basic_rag/1_offline_pipeline.py",
            "v1_basic_rag/2_online_pipeline.py",
            "v1_basic_rag/3_conversational_rag.py",
            "v2_advanced_rag/5_v2_pipeline.py",
            "v2_multi_index/1_build_indexes.py",
            "v2_multi_index/4_v2_multi_index_pipeline.py",
            "evaluation/1_run_versions.py",
            "evaluation/2_retrieval_metrics.py",
            "evaluation/3_generation_metrics.py",
            "evaluation/4_comparison_table.py",
        ]
        missing = [item for item in required if not (ROOT / item).is_file()]
        self.assertEqual(missing, [], f"Missing required project files: {missing}")

    def test_dataset_json_and_document_count(self):
        payload = json.loads((ROOT / "data/dandes_electronics_kb.json").read_text(encoding="utf-8"))
        docs = payload.get("documents")
        self.assertIsInstance(docs, list)
        self.assertEqual(len(docs), 50)
        self.assertEqual(len({doc.get("id") for doc in docs}), 50, "Document IDs must be unique")

    def test_required_document_fields_and_metadata(self):
        payload = json.loads((ROOT / "data/dandes_electronics_kb.json").read_text(encoding="utf-8"))
        required_meta = {"product", "brand", "category", "policy_code", "date", "related_policy"}
        for i, doc in enumerate(payload["documents"]):
            with self.subTest(document=i, doc_id=doc.get("id")):
                self.assertTrue(doc.get("id"))
                self.assertTrue(doc.get("doc_type"))
                self.assertTrue(doc.get("title"))
                self.assertTrue(doc.get("content"))
                self.assertIsInstance(doc.get("metadata"), dict)
                self.assertTrue(required_meta.issubset(doc["metadata"].keys()))

    def test_expected_document_type_distribution(self):
        payload = json.loads((ROOT / "data/dandes_electronics_kb.json").read_text(encoding="utf-8"))
        counts = Counter(doc["doc_type"] for doc in payload["documents"])
        self.assertEqual(counts, {
            "product_faq": 12,
            "return_policy": 5,
            "shipping_policy": 5,
            "order_guide": 6,
            "known_issue": 8,
            "resolved_ticket": 14,
        })

    def test_golden_test_set_has_twenty_questions(self):
        questions = json.loads((ROOT / "evaluation/golden_test_set.json").read_text(encoding="utf-8"))
        self.assertEqual(len(questions), 20)
        required = {"id", "question", "reference_answer", "expected_sources"}
        for q in questions:
            self.assertTrue(required.issubset(q.keys()), f"Question missing fields: {q.get('id')}")

    def test_gitignore_excludes_secrets_and_local_artifacts(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in [".env", ".venv/", ".venv_ragas/", "chroma_db/", "__pycache__/", "*.pyc"]:
            self.assertIn(entry, ignore)

    def test_all_python_files_have_install_notes(self):
        scripts = list(ROOT.rglob("*.py"))
        self.assertGreaterEqual(len(scripts), 17)
        for script in scripts:
            with self.subTest(script=str(script.relative_to(ROOT))):
                first = script.read_text(encoding="utf-8").lstrip()
                self.assertTrue(first.startswith("# INSTALLATION / RUN NOTES"), "Missing setup comments")

    def test_results_files_are_valid_json_when_present(self):
        result_dir = ROOT / "evaluation/results"
        self.assertTrue(result_dir.is_dir())
        result_files = list(result_dir.glob("*.json"))
        self.assertGreaterEqual(len(result_files), 5)
        for result_file in result_files:
            with self.subTest(file=result_file.name):
                json.loads(result_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
