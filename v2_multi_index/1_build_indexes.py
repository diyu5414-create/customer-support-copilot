# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


#=======================================
# V2 Multi-Index RAG — Build Indexes
#=======================================

import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


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

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "dandes_electronics_kb.json"
)

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

embeddings_model = OpenAIEmbeddings(
    api_key=API_KEY,
    model=EMBEDDING_MODEL
)


#=======================================
# Load the Original Dataset
#=======================================

with open(DATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

records = data["documents"]

print("Total records in dataset:", len(records))


#=======================================
# Define the Four Indexes
#=======================================

INDEX_CONFIG = {
    "policy_index": [
        "return_policy",
        "shipping_policy"
    ],
    "product_index": [
        "product_faq",
        "known_issue"
    ],
    "order_index": [
        "order_guide"
    ],
    "ticket_index": [
        "resolved_ticket"
    ]
}


#=======================================
# Build Documents for Each Index
#=======================================

for collection_name, doc_types in INDEX_CONFIG.items():

    documents = []
    ids = []

    for record in records:

        if record.get("doc_type") not in doc_types:
            continue

        # Extract the original metadata dictionary.
        record_metadata = record.get("metadata", {})

        # Build Chroma metadata using the required fields.
        metadata = {
            "id": record["id"],
            "doc_type": record["doc_type"]
        }

        for key in [
            "product",
            "brand",
            "category",
            "policy_code",
            "date",
            "related_policy"
        ]:
            value = record_metadata.get(key)

            if value is not None:
                metadata[key] = value

        # Keep one complete record as one document.
        document_text = (
            f"ID: {record['id']}\n"
            f"Type: {record['doc_type']}\n"
            f"Title: {record.get('title', '')}\n"
            f"Content: {record.get('content', '')}"
        )

        documents.append(
            Document(
                page_content=document_text,
                metadata=metadata
            )
        )

        ids.append(record["id"])

    #=======================================
    # Create or Update the Chroma Collection
    #=======================================

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model
    )

    # Remove old documents before rebuilding the collection.
    existing_data = vectorstore.get()

    if existing_data["ids"]:
        vectorstore.delete(ids=existing_data["ids"])

    if documents:
        vectorstore.add_documents(
            documents=documents,
            ids=ids
        )

    print(
        f"{collection_name}: "
        f"{vectorstore._collection.count()} documents"
    )


#=======================================
# Verify the Expected Counts
#=======================================

print("\nExpected document counts:")

EXPECTED_COUNTS = {
    "policy_index": 10,
    "product_index": 20,
    "order_index": 6,
    "ticket_index": 14
}

for collection_name, expected_count in EXPECTED_COUNTS.items():

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model
    )

    actual_count = vectorstore._collection.count()

    print(
        f"{collection_name}: "
        f"{actual_count}/{expected_count}"
    )

    if actual_count != expected_count:
        print(
            f"WARNING: Expected {expected_count}, "
            f"but found {actual_count}."
        )

print("\nMulti-index build completed.")
