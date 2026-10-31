# Customer Support Copilot — RAG Assignment

## Project Overview

This project implements a **Customer Support Copilot using Retrieval-Augmented Generation (RAG)** for the Dandes Electronics knowledge base.

The system answers customer questions using a knowledge base containing **50 documents**. It retrieves relevant information and generates context-grounded responses using an LLM.

The project implements and compares three RAG approaches.

### V1 — Basic RAG

- Document loading and chunking
- OpenAI embeddings
- Chroma vector database
- Dense vector similarity search
- Context-based answer generation
- Offline, online, and conversational RAG

### V2 — Advanced RAG (Single Index)

- Metadata filtering
- Dense and sparse retrieval
- BM25 keyword search
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Query transformation and multi-query retrieval
- Conversational RAG pipeline

### V2 — Advanced RAG (Multi-Index)

- Multiple knowledge indexes
- Intent-based query routing
- Index-specific retrieval
- Multi-index search
- Context-grounded response generation
- Out-of-scope query handling

---

## Project Requirements

- Python 3.10+
- OpenAI API key
- ChromaDB
- LangChain integration packages
- Sentence Transformers
- BM25 retrieval
- Git

---

## Project Structure

```text
customer_support_copilot/
├── data/
│   └── dandes_electronics_kb.json
├── v1_basic_rag/
│   ├── basic_rag.py
│   ├── offline_rag.py
│   ├── online_rag.py
│   └── conversational_rag.py
├── v2_advanced_rag/
│   ├── metadata_filter.py
│   ├── hybrid_search.py
│   ├── reranking.py
│   ├── query_transformation.py
│   └── advanced_rag_pipeline.py
├── v2_multi_index/
│   ├── build_indexes.py
│   ├── multi_index_search.py
│   ├── intent_router.py
│   └── multi_index_pipeline.py
├── evaluation/
│   ├── golden_test_set.json
│   ├── 1_run_versions.py
│   ├── 2_retrieval_metrics.py
│   ├── 3_generation_metrics.py
│   ├── 4_comparison_table.py
│   └── results/
│       ├── v1_results.json
│       ├── v2_single_index_results.json
│       ├── v2_multi_index_results.json
│       ├── retrieval_metrics.json
│       ├── generation_metrics.json
│       └── comparison_table.csv
├── tests/
│   └── test_project_integrity.py
├── demo_output.txt
├── report.pdf
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/diyu5414-create/customer-support-copilot.git
cd customer-support-copilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
JUDGE_MODEL=your_judge_model
```

Replace the placeholders with your actual API credentials and model name.

**Security:** Never commit `.env`, expose API keys, or include credentials in a public repository or submission archive.

---

## Execution Guide

Run commands from the project root.

### V1 — Basic RAG

```bash
python v1_basic_rag/basic_rag.py
python v1_basic_rag/offline_rag.py
python v1_basic_rag/online_rag.py
python v1_basic_rag/conversational_rag.py
```

### V2 — Advanced RAG (Single Index)

```bash
python v2_advanced_rag/metadata_filter.py
python v2_advanced_rag/hybrid_search.py
python v2_advanced_rag/reranking.py
python v2_advanced_rag/query_transformation.py
python v2_advanced_rag/advanced_rag_pipeline.py
```

### V2 — Advanced RAG (Multi-Index)

Build the knowledge indexes first:

```bash
python v2_multi_index/build_indexes.py
```

Then run:

```bash
python v2_multi_index/multi_index_search.py
python v2_multi_index/intent_router.py
python v2_multi_index/multi_index_pipeline.py
```

---

## Evaluation

The project compares three RAG approaches using a shared **20-question evaluation set per version**, resulting in 60 evaluated answers.

### Retrieval Metrics

- **Precision@5:** Proportion of retrieved documents that are relevant.
- **Recall@5:** Proportion of expected relevant documents successfully retrieved.
- **F1@5:** Harmonic mean of Precision@5 and Recall@5.
- **Coverage:** Measures whether expected relevant sources are represented in the retrieved results, according to the evaluation implementation.
- **Routing Accuracy:** Measures whether the multi-index router selects the expected knowledge indexes.
- **Refusal Accuracy:** Measures correct handling of in-scope and out-of-scope questions.
- **False Refusals:** Counts in-scope questions incorrectly refused.

### Generation Metrics

The project uses custom LLM-as-Judge evaluation.

**Faithfulness (LLM Judge):**
- Each generated answer is classified as `Faithful`, `Partial`, or `Unfaithful`.
- Faithfulness percentage is calculated from answers receiving the `Faithful` classification.
- Refusals are excluded from the faithfulness percentage.

**Refusal Evaluation:**
- Correct refusal
- Should have refused but didn't
- False refusal
- Correctly answered without refusal

The faithfulness percentage and refusal accuracy measure different aspects of generation quality.

---

## Retrieval Results

| Metric | V1 Basic RAG | V2 Single-Index | V2 Multi-Index |
||---:|---:|---:|
| Precision@5 | 28.75% | 30.00% | 28.75% |
| Recall@5 | 91.25% | 92.50% | 94.38% |
| F1@5 | 41.84% | 43.09% | 42.14% |
| Coverage | 93.75% | 93.75% | 100.00% |
| Routing Accuracy | N/A | N/A | 95.00% |
| Refusal Accuracy | 95.00% | 95.00% | 100.00% |
| False Refusals | 1 | 1 | 0 |
| Questions Evaluated | 20 | 20 | 20 |

## Generation Results

| Metric | V1 Basic RAG | V2 Single-Index | V2 Multi-Index |
||---:|---:|---:| 
| Faithfulness (LLM Judge) | 100.00% | 73.33% | 12.50% |
| Refusal Accuracy | 95.00% | 95.00% | 100.00% |
| False Refusals | 1 | 1 | 0 |
| Questions Evaluated | 20 | 20 | 20 |

---

## V2 Multi-Index Target Assessment

| Target | Recorded Result | Assessment |
||---:|---|
| Routing Accuracy ≥ 85% | 95.00% | Met |
| Recall@5 ≥ 0.80 | 94.38% | Met |
| Faithfulness (LLM Judge) ≥ 85% | 12.50% | Not met |
| Refusal Accuracy = 100% | 100.00% | Met |
| False Refusals = 0 | 0 | Met |

---

## Interpretation

V2 Multi-Index achieves the highest Recall@5, Coverage, Routing Accuracy, and refusal accuracy. It also produces zero false refusals.

However, its Faithfulness (LLM Judge) score is lower than the other versions. This suggests that although retrieval and routing performance are strong, the generated answers require further improvement in grounding and factual consistency.

V2 Single-Index achieves the highest Precision@5 and F1@5.

The results demonstrate trade-offs across the three approaches rather than a universal winner. Improving answer grounding while preserving retrieval and routing performance is an important next step.

---

## Testing

Run the automated integrity tests:

```bash
python -m unittest discover -s tests -v
```

Expected result:

```text
Ran 8 tests
OK
```

### Python Compilation Check

```bash
python -m compileall -q v1_basic_rag v2_advanced_rag v2_multi_index evaluation
```

No output indicates successful compilation.

---

## Demo and Report

- `demo_output.txt` — sample project execution output.
- `report.pdf` — detailed RAG architecture comparison and evaluation report.

---

## Security Considerations

- API keys are managed through environment variables.
- Credentials are excluded from version control.
- Retrieved knowledge base context is used to ground generated answers.
- Out-of-scope questions are handled through refusal mechanisms.
- Evaluation is used to assess retrieval and generation performance.

---

## Important Implementation Notes

- The project uses `TOP_K = 5` for retrieval evaluation.
- Retriever `k` and `top_k` values are retained from the project scripts.
- Core dependencies include `langchain-core`, `langchain-openai`, `langchain-chroma`, `chromadb`, `rank-bm25`, and `sentence-transformers`.
- The umbrella `langchain` package is not required by the core project.
- The environment variables used by the project are `OPENAI_API_KEY` and `JUDGE_MODEL`.
- Custom evaluation includes retrieval metrics, refusal evaluation, and LLM-as-Judge faithfulness.
- Evaluation results are available in `evaluation/results/`.

---

## Technologies Used

Python · LangChain · OpenAI API · OpenAI Embeddings · ChromaDB · BM25 · Reciprocal Rank Fusion (RRF) · Sentence Transformers · Cross-Encoder Reranking · RAG · LLM-as-Judge · Git · GitHub

---

## Repository

GitHub: https://github.com/diyu5414-create/customer-support-copilot
