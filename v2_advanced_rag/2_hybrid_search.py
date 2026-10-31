# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


#=======================================
# V2 Advanced RAG — Hybrid Search
# BM25 + Dense Search + Reciprocal Rank Fusion (RRF)
#=======================================

import os
import re

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


#=======================================
# 1. Setup — Connect to Existing Vector Store
#=======================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

embeddings_model = OpenAIEmbeddings(
    api_key=API_KEY,
    model=EMBEDDING_MODEL
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings_model,
    collection_name=COLLECTION_NAME
)

print(
    f"Connected to vector store. "
    f"Items indexed: {vectorstore._collection.count()}"
)


#=======================================
# 2. Build BM25 Index Over Same Documents
#=======================================

all_data = vectorstore.get()

chunk_texts = all_data["documents"]
chunk_metadatas = all_data["metadatas"]

print(
    f"Pulled {len(chunk_texts)} documents "
    f"from the vector store for BM25 indexing."
)


def tokenize(text):
    # Keep identifiers such as RR-004 together.
    return re.findall(r"[a-z]+-\d+|\w+", text.lower())


tokenized_text = [
    tokenize(text) for text in chunk_texts
]

bm25 = BM25Okapi(tokenized_text)

print("BM25 index built over the same documents as dense search.")


#=======================================
# 3. BM25 Search
#=======================================

def bm25_search(question, k=20):

    tokenized_query = tokenize(question)

    # Score every document against the query.
    scores = bm25.get_scores(tokenized_query)

    # Pair each document with its metadata and score.
    scored_documents = list(
        zip(chunk_texts, chunk_metadatas, scores)
    )

    scored_documents.sort(
        key=lambda item: item[2],
        reverse=True
    )

    return scored_documents[:k]


#=======================================
# 4. Exact Policy ID Matching
#=======================================

def add_exact_id_match(question, results):

    # Detect IDs such as RR-004 or SH-002.
    match = re.search(
        r"\b[A-Z]{2,}-\d+\b",
        question.upper()
    )

    if match:
        policy_id = match.group(0)

        for text, metadata in zip(
            chunk_texts, chunk_metadatas
        ):
            if metadata.get("id") == policy_id:

                # Remove the existing result if present.
                results = [
                    item for item in results
                    if item[1].get("id") != policy_id
                ]

                # Place the exact match at the beginning.
                # This is an exact-ID fallback, not a BM25 score.
                results.insert(
                    0,
                    (text, metadata, 999.0)
                )

                print(
                    f"Exact ID match added: {policy_id}"
                )

                break

    return results


#=======================================
# 5. Reciprocal Rank Fusion (RRF)
#=======================================

def reciprocal_rank_fusion(result_lists, k=60):

    scores = {}
    documents_by_id = {}

    for result_list in result_lists:

        for rank, doc in enumerate(
            result_list, start=1
        ):
            doc_id = doc.metadata["id"]

            # RRF score = sum of 1 / (k + rank).
            scores[doc_id] = (
                scores.get(doc_id, 0)
                + 1 / (k + rank)
            )

            documents_by_id[doc_id] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    # Diagnostic for the assignment's RR-004 demo.
    if "RR-004" in scores:
        print(
            f"\nRR-004 RRF score: "
            f"{scores['RR-004']:.6f}"
        )
        print(
            "RR-004 RRF position:",
            ranked_ids.index("RR-004") + 1
        )

    return [
        documents_by_id[doc_id]
        for doc_id in ranked_ids
    ]


#=======================================
# 6. Configure Assignment Demo
#=======================================

TOP_K = 5
CANDIDATE_K = 20

question = "What does policy RR-004 say?"


#=======================================
# 7. BEFORE — Basic V1 Dense Search
#=======================================

print("\n" + "=" * 70)
print("BEFORE: BASIC V1 DENSE SEARCH")
print("=" * 70)

before_results = vectorstore.similarity_search(
    question,
    k=TOP_K
)

for rank, doc in enumerate(
    before_results, start=1
):
    print(
        f"Rank {rank}: {doc.metadata['id']} | "
        f"{doc.metadata['doc_type']}"
    )


#=======================================
# 8. Dense Search — Retrieve Top 20
#=======================================

dense_results = vectorstore.similarity_search(
    question,
    k=CANDIDATE_K
)

print("\nDENSE TOP RESULTS")

for rank, doc in enumerate(
    dense_results, start=1
):
    print(
        f"Rank {rank}: {doc.metadata['id']}"
    )


#=======================================
# 9. BM25 Search — Retrieve Top 20
#=======================================

bm25_results = bm25_search(
    question,
    k=CANDIDATE_K
)

# Add an exact policy-ID match when specified.
bm25_results = add_exact_id_match(
    question,
    bm25_results
)

print("\n" + "=" * 70)
print("BM25 TOP RESULTS")
print("=" * 70)

for rank, (text, metadata, score) in enumerate(
    bm25_results[:TOP_K], start=1
):
    print(
        f"Rank {rank}: {metadata['id']} | "
        f"{metadata['doc_type']} | "
        f"score={score:.4f}"
    )


# Check whether RR-004 is in the BM25 candidates.
rr004_rank = None

for rank, (text, metadata, score) in enumerate(
    bm25_results, start=1
):
    if metadata.get("id") == "RR-004":
        rr004_rank = rank
        break

if rr004_rank is not None:
    print(
        f"\nRR-004 found at BM25 candidate rank "
        f"{rr004_rank}."
    )
else:
    print("\nRR-004 not found in BM25 candidates.")


# Convert BM25 results into LangChain Documents.
bm25_documents = []

for text, metadata, score in bm25_results:

    doc = Document(
        page_content=text,
        metadata=metadata
    )

    bm25_documents.append(doc)


#=======================================
# 10. AFTER — Hybrid Search with RRF
#=======================================

hybrid_results = reciprocal_rank_fusion(
    [
        dense_results,
        bm25_documents
    ]
)

after_results = hybrid_results[:TOP_K]

print("\n" + "=" * 70)
print("AFTER: HYBRID SEARCH WITH RRF")
print("=" * 70)

for rank, doc in enumerate(
    after_results, start=1
):
    print(
        f"Rank {rank}: {doc.metadata['id']} | "
        f"{doc.metadata['doc_type']}"
    )


#=======================================
# 11. Explain the Improvement
#=======================================

print("\nExplanation:")
print(
    "Basic V1 uses dense semantic search alone."
)
print(
    "Hybrid search combines dense semantic search "
    "with BM25 keyword search using RRF."
)
print(
    "Exact policy-ID matching is an additional "
    "fallback for queries containing a policy ID."
)
