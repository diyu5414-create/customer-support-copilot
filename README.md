# Customer Support Copilot

## Project Overview

Customer Support Copilot is a Retrieval-Augmented Generation (RAG) application built using the Dandes Electronics knowledge base. It compares three RAG approaches:

1. **V1 Basic RAG** — basic document retrieval and answer generation.
2. **V2 Advanced RAG** — metadata filtering, hybrid search, reranking, and query transformation.
3. **V2 Multi-Index RAG** — intent-based routing across specialized knowledge indexes.

The project evaluates retrieval quality, answer faithfulness, routing accuracy, and out-of-scope refusal behavior.

## Project Requirements

- Python
- OpenAI API key
- Required Python packages listed in `requirements.txt`

## Setup

### 1. Open the project directory

Open a terminal in the project root directory.

### 2. Create and activate the virtual environment

On Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

If `.venv` already exists, activate it instead of recreating it.

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root directory. Add your own OpenAI API key and the model and database settings required by the project.

Example configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANSWER_MODEL=your_supported_chat_model
JUDGE_MODEL=your_supported_judge_model
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PATH=chroma_db
COLLECTION_NAME=dandes_v1
```

Replace both model placeholders with model IDs actually available to your API account. The placeholders are examples, not valid model IDs. The multi-index router and pipeline intentionally stop with a clear error if ANSWER_MODEL is missing. Never commit or share your real API key.

## Execution Order

Run commands from the project root unless otherwise specified.

### Step 1: Build the V1 knowledge base

```cmd
python v1_basic_rag\1_offline_pipeline.py
```

### Step 2: Run V1 basic RAG

```cmd
python v1_basic_rag\2_online_pipeline.py
```

### Step 3: Run V1 conversational RAG

```cmd
python v1_basic_rag\3_conversational_rag.py
```

### Step 4: Run V2 advanced RAG components

Run the following scripts as required by their dependencies:

```cmd
python v2_advanced_rag\1_metadata_filtering.py
python v2_advanced_rag\2_hybrid_search.py
python v2_advanced_rag\3_reranking.py
python v2_advanced_rag\4_query_transformation.py
python v2_advanced_rag\5_v2_pipeline.py
```

### Step 5: Build the multi-index knowledge base

```cmd
python v2_multi_index\1_build_indexes.py
```

### Step 6: Test multi-index retrieval and routing

```cmd
python v2_multi_index\2_multi_index_search.py
python v2_multi_index\3_intent_router.py
```

### Step 7: Run the V2 multi-index pipeline

```cmd
python v2_multi_index\4_v2_multi_index_pipeline.py
```

### Step 8: Run evaluation for all three versions

Run the scripts in this order:

```cmd
python evaluation\1_run_versions.py
python evaluation\2_retrieval_metrics.py
python evaluation\3_generation_metrics.py
python evaluation\4_comparison_table.py
```

## Offline integrity tests

These tests validate the packaged project structure and data without calling an API or building vector indexes. Run them from the project root:

```cmd
python -m unittest discover -s tests -v
```

A passing integrity test is not a substitute for integration tests. To verify embedding, Chroma, routing, and generation end to end, configure a valid `.env`, install the required dependencies, and run the pipeline commands above. Those integration runs may use API credits and can download a cross-encoder model.

## Evaluation Outputs

Evaluation results are saved under `evaluation/results/`.

- `v1_results.json` — V1 answers, retrieval results, and sources.
- `v2_single_index_results.json` — V2 single-index answers and retrieval results.
- `v2_multi_index_results.json` — V2 multi-index answers, routing, and retrieval results.
- `retrieval_metrics.json` — retrieval, routing, refusal, and related metrics.
- `generation_metrics.json` — custom LLM-as-Judge results.

The comparison script generates `evaluation/comparison_table.csv`.

## Demo and Report

- `demo_output.txt` contains the multi-index pipeline demonstration output.
- `report.pdf` contains the project evaluation report.

## Security

- Keep `.env` local and never include it in a public repository or submission ZIP.
- Do not share API keys in screenshots, logs, or source files.
- Keep the original knowledge-base dataset unchanged.


## Evaluation method

The evaluation follows the mentor program: Precision@K, Recall@K and F1 compare retrieved document IDs with expected source IDs; an LLM judge labels faithfulness as Faithful, Partial or Unfaithful; and refusal checking identifies correct refusals, missed refusals and false refusals. Your project TOP_K remains 5. Routing accuracy is retained as an assignment-specific measure for the multi-index router. There is no RAGAS evaluation and no separate answer-correctness judge.

## Important implementation note

- The `top_k` / retriever `k` values are retained from the supplied project scripts; they have not been changed to the mentor demo's default.
- Core imports follow the mentor's package style: `langchain-core`, `langchain-openai`, `langchain-chroma`, `chromadb`, `rank-bm25`, and `sentence-transformers`. The umbrella `langchain` package is not required by the core project.
- `OPENAI_API_KEY` is the environment variable used by these project scripts. Keep the name consistent in `.env`.
- Evaluation uses only the mentor-style custom metrics; no RAGAS package is required.

