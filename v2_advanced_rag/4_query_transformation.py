# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


#=======================================
# V2 Advanced RAG — Query Transformation
# Query Rewriting + Multi-Query Retrieval
#=======================================

import os
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


#=======================================
# Setup — Load Environment Variables
#=======================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
ANSWER_MODEL = os.getenv("ANSWER_MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")


#=======================================
# Connect to Existing Vector Store
#=======================================

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
    "Connected to vector store. Items indexed:",
    vectorstore._collection.count()
)


#=======================================
# Initialize LLM
#=======================================

llm_brain = ChatOpenAI(
    api_key=API_KEY,
    model=ANSWER_MODEL,
)


#=======================================
# Build BM25 Index Over the Same Documents
#=======================================

all_data = vectorstore.get(
    include=["documents", "metadatas"]
)

chunk_texts = all_data["documents"]
chunk_metadatas = all_data["metadatas"]
chunk_ids = all_data["ids"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


tokenized_text = [
    tokenize(text)
    for text in chunk_texts
]

bm25 = BM25Okapi(tokenized_text)

print("BM25 index built over the same documents.")


#=======================================
# Load Cross-Encoder
#=======================================

print("\nLoading Cross-Encoder model...")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-Encoder model loaded.")


#=======================================
# Configuration
#=======================================

TOP_K = 5
RETRIEVE_K = 20


#=======================================
# Dense Search
#=======================================

def dense_search(question, k=RETRIEVE_K):

    results = vectorstore.similarity_search(
        question,
        k=min(k, len(chunk_texts))
    )

    return [
        (
            doc.metadata.get("id", doc_id),
            doc.page_content,
            doc.metadata
        )
        for doc, doc_id in zip(
            results,
            [
                doc.metadata.get("id", "unknown")
                for doc in results
            ]
        )
    ]


#=======================================
# Sparse Search — BM25
#=======================================

def sparse_search(question, k=RETRIEVE_K):

    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = list(
        zip(chunk_ids, chunk_texts, chunk_metadatas, scores)
    )

    scored_chunks.sort(
        key=lambda item: item[3],
        reverse=True
    )

    return [
        (doc_id, text, metadata)
        for doc_id, text, metadata, score
        in scored_chunks[:k]
    ]


#=======================================
# Reciprocal Rank Fusion
#=======================================

def reciprocal_rank_fusion(ranked_lists, k=60):

    rrf_scores = {}
    doc_info = {}

    for ranked_list in ranked_lists:

        for rank, (doc_id, text, metadata) in enumerate(
            ranked_list,
            start=1
        ):

            score = 1 / (k + rank)

            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0) + score
            )

            doc_info[doc_id] = (text, metadata)

    fused = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        (
            doc_id,
            doc_info[doc_id][0],
            doc_info[doc_id][1],
            score
        )
        for doc_id, score in fused
    ]


#=======================================
# Full Retrieval Pipeline
# Dense + BM25 + RRF + Cross-Encoder
#=======================================

def retrieve_documents(question, k=TOP_K):

    dense_results = dense_search(question)
    sparse_results = sparse_search(question)

    fused_results = reciprocal_rank_fusion(
        [dense_results, sparse_results]
    )

    candidates = fused_results[:min(8, len(fused_results))]

    pairs = [
        (question, text)
        for doc_id, text, metadata, rrf_score in candidates
    ]

    scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        (
            doc_id,
            metadata,
            text,
            float(score)
        )
        for (
            (doc_id, text, metadata, rrf_score),
            score
        ) in reranked[:k]
    ]


#=======================================
# Query Rewriting
#=======================================

def rewrite_query(raw_question):

    prompt = f"""Rewrite this customer support question into
a clear, complete question that preserves the customer's intent.

Do not answer the question.
Return only the rewritten question.

Original question: "{raw_question}"
"""

    response = llm_brain.invoke(
        [HumanMessage(content=prompt)]
    )

    return response.content.strip()


#=======================================
# Multi-Query Retrieval
#=======================================

def generate_query_variants(raw_question, n=3):

    prompt = f"""Generate {n} different search queries for
the following customer support question.

Preserve the customer's intent. Explore useful alternative
wording, symptoms, and relevant terminology.

Do not answer the question.
Return exactly {n} queries, one per line.

Original question: "{raw_question}"
"""

    response = llm_brain.invoke(
        [HumanMessage(content=prompt)]
    )

    variants = [
        line.strip().lstrip("-0123456789. ")
        for line in response.content.splitlines()
        if line.strip()
    ]

    return variants[:n]


def multi_query_retrieval(raw_question, n_variants=3):

    variants = generate_query_variants(
        raw_question,
        n=n_variants
    )

    all_queries = [raw_question] + variants
    all_ranked_lists = []

    for query in all_queries:

        dense_results = dense_search(query)
        sparse_results = sparse_search(query)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results]
        )

        all_ranked_lists.append([
            (doc_id, text, metadata)
            for doc_id, text, metadata, score in fused
        ])

    combined_results = reciprocal_rank_fusion(
        all_ranked_lists
    )

    candidates = combined_results[:min(8, len(combined_results))]

    pairs = [
        (raw_question, text)
        for doc_id, text, metadata, rrf_score in candidates
    ]

    scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    final_results = [
        (
            doc_id,
            metadata,
            text,
            float(score)
        )
        for (
            (doc_id, text, metadata, rrf_score),
            score
        ) in reranked[:TOP_K]
    ]

    return variants, final_results


#=======================================
# Display Results
#=======================================

def display_results(title, results):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    for rank, (doc_id, metadata, text, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"Rank {rank}: "
            f"ID={doc_id} | "
            f"Type={metadata.get('doc_type', 'unknown')} | "
            f"Score={score:.4f}"
        )

        print(text[:250])
        print("-" * 75)


#=======================================
# TEST — Query Rewriting
#=======================================

question_a = "My order hasn't come yet"

print("\n\nDEMO A — QUERY REWRITING")
print("Original question:", question_a)

before_a = retrieve_documents(question_a)

rewritten_question = rewrite_query(question_a)

print("Rewritten question:", rewritten_question)

after_a = retrieve_documents(rewritten_question)

display_results(
    "BEFORE — Original Query Top 5",
    before_a
)

display_results(
    "AFTER — Rewritten Query Top 5",
    after_a
)

print(
    "\nExplanation: Query rewriting makes the customer's "
    "question more explicit before retrieval."
)


#=======================================
# TEST — Multi-Query Retrieval
#=======================================

question_b = "My order hasn't come yet"

print("\n\nDEMO B — MULTI-QUERY RETRIEVAL")
print("Original question:", question_b)

before_b = retrieve_documents(question_b)

variants, after_b = multi_query_retrieval(question_b)

print("\nGenerated query variants:")

for number, variant in enumerate(variants, start=1):
    print(f"{number}. {variant}")

display_results(
    "BEFORE — Original Query Top 5",
    before_b
)

display_results(
    "AFTER — Multi-Query Retrieval Top 5",
    after_b
)

print(
    "\nExplanation: Multiple query variations can retrieve "
    "relevant documents that one wording might miss."
)
