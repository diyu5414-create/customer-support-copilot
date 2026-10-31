# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


#=======================================
# V2 Advanced RAG — Complete Single-Index Pipeline
# Metadata Filtering + Hybrid Search + Reranking
# + Query Transformation
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
# Setup
#=======================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
ANSWER_MODEL = os.getenv("ANSWER_MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

embeddings_model = OpenAIEmbeddings(
    api_key=API_KEY,
    model=EMBEDDING_MODEL
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings_model,
    collection_name=COLLECTION_NAME
)

total_documents = vectorstore._collection.count()
print("Connected to vector store. Documents:", total_documents)

if total_documents == 0:
    raise ValueError("Chroma collection is empty. Run the V1 offline pipeline first.")

llm_brain = ChatOpenAI(
    api_key=API_KEY,
    model=ANSWER_MODEL
)


#=======================================
# Build BM25 Index
#=======================================

all_data = vectorstore.get(include=["documents", "metadatas"])

chunk_ids = all_data["ids"]
chunk_texts = all_data["documents"]
chunk_metadatas = all_data["metadatas"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


bm25 = BM25Okapi([tokenize(text) for text in chunk_texts])

print("BM25 index built.")


#=======================================
# Load Cross-Encoder
#=======================================

print("Loading Cross-Encoder model...")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-Encoder model loaded.")


#=======================================
# Configuration
#=======================================

TOP_K = 5
RETRIEVE_K = min(20, total_documents)
RERANK_K = 8

REFUSAL = "I don't know based on the available information."


#=======================================
# Step 1 — Metadata Filtering
#=======================================

def get_search_filter(category=None, product=None, doc_type=None):

    conditions = []

    if category:
        conditions.append({"category": {"$eq": category}})

    if product:
        conditions.append({"product": {"$eq": product}})

    if doc_type:
        conditions.append({"doc_type": {"$eq": doc_type}})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


#=======================================
# Step 2 — Dense Search
#=======================================

def dense_search(question, search_filter=None, k=RETRIEVE_K):

    results = vectorstore.similarity_search(
        question,
        k=min(k, total_documents),
        filter=search_filter
    )

    return [
        (doc.metadata.get("id", doc_id), doc.page_content, doc.metadata)
        for doc, doc_id in zip(
            results,
            [doc.metadata.get("id", "unknown") for doc in results]
        )
    ]


#=======================================
# Step 3 — BM25 Search
#=======================================

def sparse_search(question, search_filter=None, k=RETRIEVE_K):

    scores = bm25.get_scores(tokenize(question))

    scored_chunks = list(
        zip(chunk_ids, chunk_texts, chunk_metadatas, scores)
    )

    if search_filter:
        # Apply the supported assignment filters to BM25 results too.
        for field, condition in search_filter.items():
            if field == "$and":
                for item in condition:
                    key = next(iter(item))
                    value = item[key]["$eq"]
                    scored_chunks = [
                        row for row in scored_chunks
                        if row[2].get(key) == value
                    ]
            else:
                key = field
                value = condition["$eq"]
                scored_chunks = [
                    row for row in scored_chunks
                    if row[2].get(key) == value
                ]

    scored_chunks.sort(key=lambda row: row[3], reverse=True)

    return [
        (doc_id, text, metadata)
        for doc_id, text, metadata, score in scored_chunks[:k]
    ]


#=======================================
# Step 4 — Reciprocal Rank Fusion
#=======================================

def reciprocal_rank_fusion(ranked_lists, k=60):

    rrf_scores = {}
    doc_info = {}

    for ranked_list in ranked_lists:
        for rank, (doc_id, text, metadata) in enumerate(
            ranked_list, start=1
        ):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            )
            doc_info[doc_id] = (text, metadata)

    fused = sorted(
        rrf_scores.items(),
        key=lambda row: row[1],
        reverse=True
    )

    return [
        (doc_id, doc_info[doc_id][0], doc_info[doc_id][1], score)
        for doc_id, score in fused
    ]


#=======================================
# Step 5 — Cross-Encoder Reranking
#=======================================

def rerank_results(question, fused_results):

    candidates = fused_results[:min(RERANK_K, len(fused_results))]

    if not candidates:
        return []

    pairs = [
        (question, text)
        for doc_id, text, metadata, rrf_score in candidates
    ]

    scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(candidates, scores),
        key=lambda row: float(row[1]),
        reverse=True
    )

    return [
        (doc_id, metadata, text, float(score))
        for (
            (doc_id, text, metadata, rrf_score),
            score
        ) in reranked[:TOP_K]
    ]


#=======================================
# Step 6 — Hybrid Search
#=======================================

def hybrid_search(question, category=None, product=None, doc_type=None):

    search_filter = get_search_filter(
        category=category,
        product=product,
        doc_type=doc_type
    )

    dense_results = dense_search(question, search_filter)
    sparse_results = sparse_search(question, search_filter)

    fused_results = reciprocal_rank_fusion(
        [dense_results, sparse_results]
    )

    return rerank_results(question, fused_results)


#=======================================
# Step 7 — Query Rewriting
#=======================================

def rewrite_query(question):

    prompt = f"""Rewrite this customer-support question into a
clear and complete search query without changing its meaning.

Do not answer the question. Return only the rewritten query.

Question: {question}
"""

    response = llm_brain.invoke([HumanMessage(content=prompt)])

    return response.content.strip()


#=======================================
# Step 8 — Multi-Query Retrieval
#=======================================

def generate_query_variants(question, n=3):

    prompt = f"""Generate exactly {n} alternative search queries
that preserve the user's intent and use different relevant wording.

Do not answer the question.
Return one query per line.

Question: {question}
"""

    response = llm_brain.invoke([HumanMessage(content=prompt)])

    variants = [
        line.strip().lstrip("-0123456789. ")
        for line in response.content.splitlines()
        if line.strip()
    ]

    return variants[:n]



def multi_query_search(
    question,
    n=3,
    category=None,
    product=None,
    doc_type=None
):
    search_filter = get_search_filter(
        category=category,
        product=product,
        doc_type=doc_type
    )

    variants = generate_query_variants(question, n)
    all_queries = [question] + variants
    all_ranked_lists = []

    for query in all_queries:
        dense_results = dense_search(query, search_filter)
        sparse_results = sparse_search(query, search_filter)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results]
        )

        all_ranked_lists.append([
            (doc_id, text, metadata)
            for doc_id, text, metadata, score in fused
        ])

    combined = reciprocal_rank_fusion(all_ranked_lists)

    return variants, rerank_results(question, combined)


#=======================================
# Step 9 — Generate a Grounded Answer
#=======================================


def generate_answer(question, results):

    if not results:
        return REFUSAL

    context_parts = []

    for doc_id, metadata, text, score in results:
        context_parts.append(
            f"[Source ID: {doc_id}]\n"
            f"[Document type: {metadata.get('doc_type', 'unknown')}]\n"
            f"[Document date: {metadata.get('date', 'unknown')}]\n"
            f"{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are a customer-support assistant for Dandes Electronics.

Answer the customer using ONLY the provided knowledge-base context.

Rules:
- Do not invent facts, policies, dates, or timelines.
- Cite the source IDs used, for example [RR-001].
- If the context does not contain enough information to answer,
  respond exactly: "{REFUSAL}"
- For current policy questions, prioritize the latest applicable policy document.
- Current policy documents take precedence over older resolved tickets.
- If policies conflict, compare their effective dates and follow the latest effective policy.
- RR-001 states that the current general return policy is effective from 1 July 2026 and replaces all earlier return windows.
- RR-002 provides additional conditions for opened laptops and accessories; use it alongside RR-001 when relevant.
- When answering a policy question, explicitly state the effective date if it is available in the context.
- Do not treat an old resolved ticket as the current policy.
- Include all relevant return conditions supported by the context.
- Do not cite a source unless you use information from it.

Knowledge-base context:
{context}

Customer question:
{question}

Answer:
"""

    response = llm_brain.invoke(
        [HumanMessage(content=prompt)]
    )

    return response.content.strip()

#=======================================
# Step 10 — Complete V2 Pipeline
#=======================================


def v2_pipeline(question, category=None, product=None, doc_type=None):

    print("\n" + "=" * 75)
    print("QUESTION:", question)
    print("=" * 75)

    # Step 1: Query rewriting
    rewritten_query = rewrite_query(question)
    print("\nRewritten query:")
    print(rewritten_query)

    # Step 2: Hybrid retrieval with metadata filtering
    hybrid_results = hybrid_search(
        rewritten_query,
        category=category,
        product=product,
        doc_type=doc_type
    )

    # Step 3: Multi-query retrieval with metadata filtering
    variants, multi_results = multi_query_search(
        question,
        n=3,
        category=category,
        product=product,
        doc_type=doc_type
    )

    print("\nAlternative queries:")
    for variant in variants:
        print("-", variant)

    # Step 4: Combine results from both retrieval paths
    combined = {}
    for doc_id, metadata, text, score in hybrid_results + multi_results:
        if doc_id not in combined:
            combined[doc_id] = (doc_id, metadata, text, score)

    # Step 5: Rerank combined candidates for the original question
    results = rerank_results(
        question,
        [
            (doc_id, text, metadata, score)
            for doc_id, metadata, text, score in combined.values()
        ]
    )

    print("\nRetrieved and reranked sources:")
    for rank, (doc_id, metadata, text, score) in enumerate(results, start=1):
        print(
            f"Rank {rank}: {doc_id} | "
            f"{metadata.get('doc_type', 'unknown')} | "
            f"Cross-Encoder score={score:.4f}"
        )

    answer = generate_answer(question, results)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    print(", ".join(doc_id for doc_id, _, _, _ in results) if results else "None")

    return results, answer



#=======================================
# Demonstrations
#=======================================

if __name__ == "__main__":

    v2_pipeline(
        "Any known issues with the Galaxy Buds3 Pro?",
        product="Galaxy Buds3 Pro",
        doc_type="known_issue"
    )

    v2_pipeline(
        "What does policy RR-004 say?",
        doc_type="return_policy"
    )

    v2_pipeline(
        "Customer got a damaged phone and wants a refund. What's the process?"
    )

    v2_pipeline(
        "My order hasn't come yet"
    )


    v2_pipeline(
        "I bought a laptop, opened it, and want to return it. What is the current return policy?"
    )

    v2_pipeline(
        "Do you sell refrigerators, and what models are available?"
    )

