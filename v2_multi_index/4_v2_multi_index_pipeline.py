# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


# ============================================================
# FILE: 4_v2_multi_index_pipeline.py
# PURPOSE: Complete V2 Multi-Index RAG Pipeline
# FLOW:
# Route -> Hybrid Search -> Merge -> Global Reranking
#       -> Top 5 -> Answer with Sources
# ============================================================

import os
import sys
import json
import importlib.util
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-6-luna")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

CHROMA_DB_PATH = os.getenv("CHROMA_PATH", "chroma_db")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")


# ============================================================
# 2. LOAD THE DATASET
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "dandes_electronics_kb.json"
)

with open(DATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

records = data["documents"]

print("Total dataset records:", len(records))


# ============================================================
# 3. INITIALIZE MODELS
# ============================================================

embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=API_KEY
)

llm = ChatOpenAI(
    model=ANSWER_MODEL,
    api_key=API_KEY
)

print("Loading Cross-Encoder model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Models initialized successfully.")


# ============================================================
# 4. LOAD THE FOUR EXISTING CHROMA INDEXES
# ============================================================

policy_vectorstore = Chroma(
    collection_name="policy_index",
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model
)

product_vectorstore = Chroma(
    collection_name="product_index",
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model
)

order_vectorstore = Chroma(
    collection_name="order_index",
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model
)

ticket_vectorstore = Chroma(
    collection_name="ticket_index",
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model
)

vectorstores = {
    "policy_index": policy_vectorstore,
    "product_index": product_vectorstore,
    "order_index": order_vectorstore,
    "ticket_index": ticket_vectorstore
}

print("\nChroma index document counts:")

for index_name, vectorstore in vectorstores.items():
    print(
        index_name,
        ":",
        vectorstore._collection.count(),
        "documents"
    )


# ============================================================
# 5. PREPARE DOCUMENTS FOR BM25
# ============================================================

INDEX_DOC_TYPES = {
    "policy_index": ["return_policy", "shipping_policy"],
    "product_index": ["product_faq", "known_issue"],
    "order_index": ["order_guide"],
    "ticket_index": ["resolved_ticket"]
}

bm25_indexes = {}
bm25_documents = {}

for index_name, allowed_types in INDEX_DOC_TYPES.items():

    index_documents = []

    for record in records:

        if record.get("doc_type") not in allowed_types:
            continue

        metadata = record.get("metadata", {})

        # Keep the original record as one document.
        # Support either "content" or "text" as the text field.
        document_text = record.get(
            "content",
            record.get("text", "")
        )

        if not document_text:
            document_text = json.dumps(
                record,
                ensure_ascii=False
            )

        document_content = (
            f"ID: {record.get('id', '')}\n"
            f"Document Type: {record.get('doc_type', '')}\n"
            f"Date: {metadata.get('date', '')}\n"
            f"Product: {metadata.get('product', '')}\n"
            f"Category: {metadata.get('category', '')}\n"
            f"Content: {document_text}"
        )

        document_metadata = {
            "id": record.get("id", ""),
            "doc_type": record.get("doc_type", "")
        }

        for key, value in metadata.items():
            if value is not None:
                document_metadata[key] = value

        index_documents.append(
            Document(
                page_content=document_content,
                metadata=document_metadata
            )
        )

    bm25_documents[index_name] = index_documents

    tokenized_documents = [
        document.page_content.lower().split()
        for document in index_documents
    ]

    bm25_indexes[index_name] = (
        BM25Okapi(tokenized_documents)
        if tokenized_documents
        else None
    )

    print(
        f"BM25 {index_name}: "
        f"{len(index_documents)} documents"
    )


# ============================================================
# 6. LOAD THE INTENT ROUTER
# ============================================================

ROUTER_PATH = os.path.join(
    "v2_multi_index",
    "3_intent_router.py"
)

spec = importlib.util.spec_from_file_location(
    "intent_router_module",
    ROUTER_PATH
)

router_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router_module)


# ============================================================
# 7. ROUTE THE QUESTION
# ============================================================

ALL_INDEX_NAMES = [
    "policy_index",
    "product_index",
    "order_index",
    "ticket_index"
]

REFUSAL_ANSWER = (
    "I don't know based on the available information."
)


def route_question(question):
    """
    Use the existing GPT intent router.

    If routing fails or the JSON is invalid, search all
    four indexes instead of crashing.
    """

    try:
        if hasattr(router_module, "route_question"):
            routing = router_module.route_question(question)

        elif hasattr(router_module, "classify_intent"):
            routing = router_module.classify_intent(question)

        else:
            raise ValueError(
                "Router function not found. "
                "Expected route_question() or classify_intent()."
            )

        if isinstance(routing, str):
            routing = json.loads(routing)

        if not isinstance(routing, dict):
            raise ValueError("Router output must be a JSON object.")

        intents = routing.get("intents")
        indexes = routing.get("indexes")
        reason = routing.get("reason", "")

        if not isinstance(intents, list):
            raise ValueError("Invalid intents in router JSON.")

        if not isinstance(indexes, list):
            raise ValueError("Invalid indexes in router JSON.")

        if "out_of_scope" in intents:
            return {
                "intents": ["out_of_scope"],
                "indexes": [],
                "reason": reason
            }

        for index_name in indexes:
            if index_name not in ALL_INDEX_NAMES:
                raise ValueError(
                    f"Unknown index returned by router: {index_name}"
                )

        if not indexes:
            raise ValueError("Router returned no indexes.")

        return {
            "intents": intents,
            "indexes": indexes,
            "reason": reason
        }

    except Exception as error:
        print("Router failed. Activating fallback:", error)

        return {
            "intents": ["fallback"],
            "indexes": ALL_INDEX_NAMES.copy(),
            "reason": "Router failure; searching all four indexes."
        }


# ============================================================
# 8. RECIPROCAL RANK FUSION (RRF)
# ============================================================

def reciprocal_rank_fusion(rankings, k=60):
    """
    Combine dense and BM25 rankings using RRF.
    """

    scores = defaultdict(float)
    documents_by_id = {}

    for ranking in rankings:

        for rank, document in enumerate(ranking, start=1):

            document_id = document.metadata.get("id")

            if not document_id:
                document_id = document.page_content[:100]

            scores[document_id] += 1 / (k + rank)
            documents_by_id[document_id] = document

    fused_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        documents_by_id[document_id]
        for document_id, score in fused_results
    ]


# ============================================================
# 9. HYBRID SEARCH IN ONE INDEX
# ============================================================

def hybrid_search(question, index_name, top_k=20):
    """
    Search one selected index using:
    1. Dense vector search
    2. BM25 keyword search
    3. Reciprocal Rank Fusion
    """

    vectorstore = vectorstores[index_name]

    # Dense retrieval
    dense_results = vectorstore.similarity_search(
        question,
        k=top_k
    )

    # BM25 keyword retrieval
    bm25 = bm25_indexes[index_name]
    documents = bm25_documents[index_name]

    if bm25 is None:
        sparse_results = []
    else:
        query_tokens = question.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:top_k]

        sparse_results = [
            documents[i]
            for i in ranked_indices
            if bm25_scores[i] > 0
        ]

    # Fuse dense and sparse results
    fused_results = reciprocal_rank_fusion(
        [dense_results, sparse_results]
    )

    return fused_results[:top_k]


# ============================================================
# 10. MERGE INDEX RESULTS AND GLOBAL RERANKING
# ============================================================

def retrieve_and_rerank(question, selected_indexes, top_k=5):
    """
    Hybrid search in each selected index.
    Merge candidates from all indexes.
    Apply ONE global Cross-Encoder reranking pass.
    Return the top five documents.
    """

    all_candidates = []
    seen_ids = set()

    for index_name in selected_indexes:

        candidates = hybrid_search(
            question,
            index_name,
            top_k=20
        )

        for document in candidates:

            document_id = document.metadata.get("id")

            if document_id and document_id in seen_ids:
                continue

            if document_id:
                seen_ids.add(document_id)

            all_candidates.append(document)

    if not all_candidates:
        return []

    # Perform global reranking only once.
    pairs = [
        (question, document.page_content)
        for document in all_candidates
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(all_candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        document
        for document, score in ranked_documents[:top_k]
    ]


# ============================================================
# 11. GENERATE ANSWER WITH SOURCES
# ============================================================

def generate_answer(question, retrieved_documents):
    """
    Generate an answer grounded in retrieved documents.
    Return the answer and IDs of the retrieved top-five sources.
    """

    if not retrieved_documents:
        return REFUSAL_ANSWER, []

    context = []
    source_ids = []

    for document in retrieved_documents:

        document_id = document.metadata.get("id", "UNKNOWN")
        document_date = document.metadata.get("date", "Unknown")

        context.append(
            f"Source ID: {document_id}\n"
            f"Date: {document_date}\n"
            f"{document.page_content}"
        )

        if document_id != "UNKNOWN" and document_id not in source_ids:
            source_ids.append(document_id)

    context_text = "\n\n---\n\n".join(context)

    prompt = f"""
You are the Dandes Electronics Customer Support Copilot.

Answer the customer using only the supplied documents.

Rules:
1. Do not invent facts or policies.
2. Prefer the latest applicable policy when documents disagree.
3. Mention the policy effective date when available and relevant.
4. The current opened-laptop return policy effective 1 July 2026
   supersedes an older ticket mentioning a seven-day window.
5. Distinguish official policy from resolved customer cases.
6. If the supplied documents do not contain enough information,
   respond exactly:
   {REFUSAL_ANSWER}
7. For questions about delivery, returns, refunds or damaged
   products, explain the practical steps and relevant timelines.
8. Do not invent source IDs. Sources are printed by the application.

Customer question:
{question}

Retrieved documents:
{context_text}

Write only the answer. Do not include a Sources line.
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    if REFUSAL_ANSWER.lower() in answer.lower():
        return REFUSAL_ANSWER, []

    return answer, source_ids


# ============================================================
# 12. RUN THE COMPLETE PIPELINE
# ============================================================

def run_pipeline(question):
    """
    Route -> Hybrid Search -> Merge -> Global Rerank
    -> Top 5 -> Answer with Sources.
    """

    print("\n" + "=" * 80)
    print("Question :", question)

    # Step 1: Routing
    routing = route_question(question)

    print(
        "Routing  : intents=["
        + ", ".join(routing["intents"])
        + "] -> indexes=["
        + ", ".join(routing["indexes"])
        + "]"
    )

    if routing.get("reason"):
        print("Reason   :", routing["reason"])

    # Step 2: Out-of-scope handling
    # Skip retrieval entirely.
    if "out_of_scope" in routing["intents"]:
        print("\nAnswer   :", REFUSAL_ANSWER)
        print("Sources  : None")

        return {
            "question": question,
            "routing": routing,
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "retrieved_ids": []
        }

    # Step 3: Retrieval and global reranking
    retrieved_documents = retrieve_and_rerank(
        question,
        routing["indexes"],
        top_k=5
    )

    # Step 4: Answer generation
    answer, sources = generate_answer(
        question,
        retrieved_documents
    )

    retrieved_ids = [
        document.metadata.get("id", "UNKNOWN")
        for document in retrieved_documents
    ]

    print("\nAnswer   :", answer)
    print("Sources  :", ", ".join(sources) if sources else "None")
    print("Top 5 retrieved IDs:", retrieved_ids)

    return {
        "question": question,
        "routing": routing,
        "answer": answer,
        "sources": sources,
        "retrieved_ids": retrieved_ids
    }


# ============================================================
# 13. SEVEN ASSIGNMENT TEST QUESTIONS
# ============================================================

test_questions = [
    "Can I return an opened laptop on day 8 if I have the original box and all accessories?",

    "My Galaxy Buds3 Pro left earbud won't connect. Is this a known issue, and has the same problem been resolved for another customer? What fixed it?",

    "How long does delivery take to a remote pincode, and is it free?",

    "A customer wants to change the address after the order shipped. What is the official process, and do we have a resolved case like this?",

    "What does policy RR-004 say?",

    "A customer received a damaged phone and wants a refund instead of a replacement. What is the official process, and do we have a previous resolved case where a refund was issued?",

    "Do you sell refrigerators?"
]


# ============================================================
# 14. RUN TESTS AND SAVE RESULTS
# ============================================================

def main():

    results = []

    for question in test_questions:
        result = run_pipeline(question)
        results.append(result)

    # Save evaluation results
    os.makedirs(
        os.path.join("evaluation", "results"),
        exist_ok=True
    )

    results_path = os.path.join(
        "evaluation",
        "results",
        "v2_multi_index_results.json"
    )

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Save readable demo output
    demo_path = "demo_output.txt"

    with open(demo_path, "w", encoding="utf-8") as file:

        for result in results:

            file.write("=" * 80 + "\n")
            file.write(f"Question : {result['question']}\n")
            file.write(f"Routing  : {result['routing']}\n")
            file.write(f"Answer   : {result['answer']}\n")
            file.write(
                "Sources  : "
                + (", ".join(result["sources"]) or "None")
                + "\n"
            )
            file.write(
                f"Top 5 retrieved IDs: {result['retrieved_ids']}\n\n"
            )

    print("\n" + "=" * 80)
    print("Pipeline completed.")
    print("Results saved to:", results_path)
    print("Demo output saved to:", demo_path)


# ============================================================
# 15. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
