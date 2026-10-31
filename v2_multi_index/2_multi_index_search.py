# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


#=======================================
# V2 Multi-Index RAG — Multi-Index Search
# Hybrid Search + RRF + Cross-Encoder
#=======================================

import os
import json
import re
from collections import defaultdict

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder


#=======================================
# Setup
#=======================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

embeddings_model = OpenAIEmbeddings(
    api_key=API_KEY,
    model=EMBEDDING_MODEL
)

INDEX_NAMES = [
    "policy_index",
    "product_index",
    "order_index",
    "ticket_index"
]

TOP_K = 5
RETRIEVE_K = 20

print("Loading Chroma indexes...")

vectorstores = {}

for index_name in INDEX_NAMES:
    vectorstores[index_name] = Chroma(
        collection_name=index_name,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model
    )

    print(
        f"{index_name}: "
        f"{vectorstores[index_name]._collection.count()} documents"
    )


#=======================================
# Load Documents for BM25
#=======================================

def tokenize(text):
    return re.findall(r"[a-z]+-\d+|\w+", text.lower())


bm25_indexes = {}

for index_name, vectorstore in vectorstores.items():

    stored_data = vectorstore.get(include=["documents", "metadatas"])

    documents = stored_data["documents"]
    metadatas = stored_data["metadatas"]
    ids = stored_data["ids"]

    bm25_indexes[index_name] = {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "bm25": BM25Okapi([
            tokenize(document)
            for document in documents
        ]) if documents else None
    }


#=======================================
# Reciprocal Rank Fusion (RRF)
#=======================================

def reciprocal_rank_fusion(rankings, k=60):

    scores = defaultdict(float)
    document_map = {}

    for ranking in rankings:

        for rank, doc in enumerate(ranking, start=1):

            doc_id = doc["id"]

            scores[doc_id] += 1 / (k + rank)
            document_map[doc_id] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        document_map[doc_id]
        for doc_id in ranked_ids
    ]


#=======================================
# Hybrid Search in One Index
#=======================================

def hybrid_search_one_index(question, index_name):

    vectorstore = vectorstores[index_name]
    index_data = bm25_indexes[index_name]

    # Dense retrieval
    dense_docs = vectorstore.similarity_search(
        question,
        k=min(RETRIEVE_K, vectorstore._collection.count())
    )

    dense_ranking = []

    for doc in dense_docs:
        dense_ranking.append({
            "id": str(doc.metadata["id"]),
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    # Sparse BM25 retrieval
    sparse_ranking = []

    if index_data["bm25"] is not None:

        query_tokens = tokenize(question)

        sparse_scores = index_data["bm25"].get_scores(
            query_tokens
        )

        top_positions = sorted(
            range(len(sparse_scores)),
            key=lambda i: sparse_scores[i],
            reverse=True
        )[:RETRIEVE_K]

        for position in top_positions:

            sparse_ranking.append({
                "id": str(index_data["ids"][position]),
                "content": index_data["documents"][position],
                "metadata": index_data["metadatas"][position]
            })

    # Merge dense and sparse rankings using RRF.
    return reciprocal_rank_fusion([
        dense_ranking,
        sparse_ranking
    ])


#=======================================
# Search Selected Indexes
#=======================================

def multi_index_search(question, selected_indexes):

    all_rankings = []

    for index_name in selected_indexes:

        if index_name not in vectorstores:
            continue

        results = hybrid_search_one_index(
            question,
            index_name
        )

        all_rankings.append(results)

    # Merge candidates from all selected indexes.
    merged_candidates = reciprocal_rank_fusion(
        all_rankings
    )

    if not merged_candidates:
        return []

    #=======================================
    # Global Cross-Encoder Reranking
    #=======================================

    pairs = [
        (question, doc["content"])
        for doc in merged_candidates
    ]

    scores = cross_encoder.predict(pairs)

    reranked_results = sorted(
        zip(merged_candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        {
            **doc,
            "rerank_score": float(score)
        }
        for doc, score in reranked_results[:TOP_K]
    ]


#=======================================
# Display Search Results
#=======================================

def display_results(title, results):

    print(f"\n{title}")
    print("=" * 70)

    if not results:
        print("No documents found.")
        return

    for position, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\n{position}. ID: {metadata.get('id', result['id'])}")
        print(f"Type: {metadata.get('doc_type', 'unknown')}")
        print(f"Content: {result['content'][:350]}")

        if "rerank_score" in result:
            print(f"Cross-Encoder score: {result['rerank_score']:.4f}")


#=======================================
# Demo: Address Change After Shipping
#=======================================

QUESTION = (
    "Can I change my delivery address after my order has shipped? "
    "What is the official process and is there a previous case?"
)

# V1 baseline: search the original single index using dense retrieval.
v1_vectorstore = Chroma(
    collection_name=os.getenv("COLLECTION_NAME", "dandes_v1"),
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings_model
)

baseline_docs = v1_vectorstore.similarity_search(
    QUESTION,
    k=TOP_K
)

baseline_results = [
    {
        "id": str(doc.metadata.get("id", "unknown")),
        "content": doc.page_content,
        "metadata": doc.metadata
    }
    for doc in baseline_docs
]

display_results("BEFORE: V1 Dense Search (Top 5)", baseline_results)


# Search all four indexes.
cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

all_index_results = multi_index_search(
    QUESTION,
    INDEX_NAMES
)

display_results(
    "AFTER: Hybrid Search Across All Four Indexes + Reranking",
    all_index_results
)


# Search only order and resolved-ticket indexes.
focused_results = multi_index_search(
    QUESTION,
    ["order_index", "ticket_index"]
)

display_results(
    "FOCUSED: Order + Ticket Indexes + Reranking",
    focused_results
)


#=======================================
# Explain the Results
#=======================================

print("\nEXPLANATION")
print("=" * 70)
print(
    "Searching all four indexes gives broad coverage. "
    "Searching only order_index and ticket_index focuses on "
    "address-change procedures and previous resolved cases. "
    "Compare the relevant document IDs in both result sets to "
    "determine which search is more useful for this question."
)
