# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


#=======================================
# V2 Advanced RAG — Metadata Filtering
#=======================================

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load API key and configuration from .env
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Stop the program if the API key is missing
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

# Create the embedding model used by the existing Chroma collection
embeddings_model = OpenAIEmbeddings(
    api_key=API_KEY,
    model=EMBEDDING_MODEL
)

# Connect to the existing V1 collection
# Do not create a new collection for this experiment
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings_model,
    collection_name=COLLECTION_NAME
)

# Print the number of documents currently indexed
print("Connected to vector store.")
print("Items indexed:", vectorstore._collection.count())

# We must retrieve the top 5 documents for this assignment
TOP_K = 5


#=======================================
# Demo Question
#=======================================

# This is the demo question specified in Section 4.1
question = "Any known issues with the Galaxy Buds3 Pro?"


#=======================================
# PART 1: Before — Basic V1 Search
#=======================================

print("\n" + "=" * 70)
print("BEFORE: BASIC V1 SEARCH")
print("=" * 70)

# Search the entire collection without applying metadata filters
# This is our baseline for comparison
before_results = vectorstore.similarity_search(
    question,
    k=TOP_K
)

# Display the top 5 documents returned by basic V1 search
for doc in before_results:
    print(
        f"- {doc.metadata['id']} | "
        f"{doc.metadata['doc_type']} | "
        f"product={doc.metadata.get('product')} | "
        f"category={doc.metadata.get('category')}"
    )


#=======================================
# PART 2: After — Metadata Filtering
#=======================================

print("\n" + "=" * 70)
print("AFTER: METADATA FILTERING")
print("=" * 70)

# Filter the search to known-issue documents for the requested product
# The filter values must match the actual stored metadata exactly

after_results = vectorstore.similarity_search(
    question,
    k=TOP_K,
    filter={
        "$and": [
            {"doc_type": {"$eq": "known_issue"}},
            {"product": {"$eq": "Galaxy Buds3 Pro"}}
        ]
    }
)


# Display the top 5 matching documents after metadata filtering
for doc in after_results:
    print(
        f"- {doc.metadata['id']} | "
        f"{doc.metadata['doc_type']} | "
        f"product={doc.metadata.get('product')} | "
        f"category={doc.metadata.get('category')}"
    )
print("Filtered result count:", len(after_results))

#=======================================
# PART 3: Explain the Difference
#=======================================

print("\nExplanation:")

# Explain how the filtered search differs from basic V1 search
print(
    "Basic V1 search considers all documents. "
    "Metadata filtering restricts the search to known-issue "
    "documents for the specified product."
)
