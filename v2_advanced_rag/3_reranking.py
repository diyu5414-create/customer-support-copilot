# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


#=======================================
# V2 Advanced RAG — Cross-Encoder Reranking
# Retrieve Top 20 → Rerank → Keep Top 5
#=======================================

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder


#=======================================
# Setup — Load Environment Variables
#=======================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL"
)

CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Check your .env file."
    )


#=======================================
# Connect to the Existing Vector Store
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

total_documents = vectorstore._collection.count()

print(
    f"Connected to vector store. "
    f"Items indexed: {total_documents}"
)

if total_documents == 0:
    raise ValueError(
        "The Chroma collection is empty. Run the V1 offline pipeline first."
    )


#=======================================
# Configuration
#=======================================

RETRIEVE_K = 20   # Stage 1: retrieve candidate documents
FINAL_K = 5       # Stage 2: keep the best five documents

QUESTION = (
    "Customer got a damaged phone and wants a refund. "
    "What's the process?"
)


#=======================================
# Helper Function — Display Results
#=======================================

def print_results(title, results):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    if not results:
        print("No documents retrieved.")
        return

    for rank, item in enumerate(results, start=1):

        # Results contain (document, score).
        # The original dense search has no Cross-Encoder score.
        if isinstance(item, tuple):
            doc, score = item
        else:
            doc = item
            score = None

        document_id = doc.metadata.get("id", "Unknown")
        doc_type = doc.metadata.get("doc_type", "Unknown")
        product = doc.metadata.get("product", "Unknown")

        if score is None:
            print(
                f"\nRank {rank}: "
                f"ID={document_id} | "
                f"Type={doc_type} | "
                f"Product={product}"
            )
        else:
            print(
                f"\nRank {rank}: "
                f"ID={document_id} | "
                f"Type={doc_type} | "
                f"Product={product} | "
                f"Cross-Encoder Score={float(score):.4f}"
            )

        print(f"Content: {doc.page_content[:350]}")


#=======================================
# Stage 1 — Original Dense Retrieval
# Retrieve Top 5 for the Baseline
#=======================================

before_results = vectorstore.similarity_search(
    QUESTION,
    k=min(FINAL_K, total_documents)
)

print_results(
    "BEFORE RERANKING — Original Dense Search Top 5",
    before_results
)


#=======================================
# Stage 2 — Retrieve Top 20 Candidates
#=======================================

candidates = vectorstore.similarity_search(
    QUESTION,
    k=min(RETRIEVE_K, total_documents)
)

print("\nNumber of candidate documents:", len(candidates))


#=======================================
# Stage 3 — Load the Cross-Encoder
#=======================================

print("\nLoading Cross-Encoder model...")

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-Encoder model loaded.")


#=======================================
# Stage 4 — Score Question-Document Pairs
#=======================================

pairs = [
    (QUESTION, doc.page_content)
    for doc in candidates
]

scores = cross_encoder.predict(pairs)


#=======================================
# Stage 5 — Sort by Cross-Encoder Score
# Highest Score First
#=======================================

reranked_results = sorted(
    zip(candidates, scores),
    key=lambda item: float(item[1]),
    reverse=True
)


#=======================================
# Stage 6 — Keep the Final Top 5
#=======================================

final_results = reranked_results[:FINAL_K]

print_results(
    "AFTER RERANKING — Cross-Encoder Top 5",
    final_results
)


#=======================================
# Stage 7 — Summary
#=======================================

print("\n" + "=" * 75)
print("RERANKING SUMMARY")
print("=" * 75)

print("Question:", QUESTION)
print("Original dense results:", len(before_results))
print("Candidates reranked:", len(candidates))
print("Final results:", len(final_results))

print(
    "\nTechnique: Dense retrieval → Cross-Encoder scoring → "
    "Sort by relevance → Top 5"
)

print(
    "\nExplanation: Dense retrieval finds candidate documents using "
    "embeddings. The Cross-Encoder then evaluates each question-document "
    "pair together and reorders the candidates by relevance."
)
