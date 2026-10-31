# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.

#=======================================
#--------Online RAG Pipeline------------
#=======================================
import json
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

#=======================================
# PART 1: LOAD ENVIRONMENT VARIABLES
#=======================================
load_dotenv()

# Read configuration values and API key from environment variables
DATA_PATH = "data/dandes_electronics_kb.json"
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

API_KEY = os.getenv("OPENAI_API_KEY")

# Check whether the API key was loaded successfully
if API_KEY:
    print("OPENAI KEY SUCCESSFULLY LOADED")
else:
    raise ValueError("OPENAI_API_KEY is missing from .env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


# load raw data to list of dictionaries
def load_documents(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


# convert raw tickets (dicts) to list of langchain document objects
def load_as_documents(path=DATA_PATH):
    tickets = load_documents(path)
    documents = []
    for ticket in tickets:
        metadata = {
            "id": ticket["id"],
            "doc_type": ticket["doc_type"],
            "product": ticket["metadata"]["product"],
            "brand": ticket["metadata"]["brand"],
            "category": ticket["metadata"]["category"],
            "policy_code": ticket["metadata"]["policy_code"],
            "date": ticket["metadata"]["date"],
            "related_policy": ticket["metadata"]["related_policy"],
        }
        # chroma does not accept None, so remove the null fields
        metadata = {key: value for key, value in metadata.items() if value is not None}

        doc = Document(
            page_content="[" + ticket["id"] + "] " + ticket["title"] + "\n" + ticket["content"],
            metadata=metadata,
        )
        documents.append(doc)
    return documents      


documents = load_as_documents()
print("Total documents:", len(documents))

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=API_KEY,
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
    ids=[document.metadata["id"] for document in documents],
)

print("Documents stored in", COLLECTION_NAME, ":", vectorstore._collection.count())