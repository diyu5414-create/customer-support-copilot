# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


#=======================================
#----- Conversational RAG Pipeline -----
#=======================================

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


#=======================================
# Setup — Connect to Existing Vector Store
#=======================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
ANSWER_MODEL = os.getenv("ANSWER_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

llm_brain = ChatOpenAI(
    api_key=API_KEY,
    model=ANSWER_MODEL,
    max_tokens=300
)

embeddings_model = OpenAIEmbeddings(
    api_key=API_KEY,
    model=EMBEDDING_MODEL
)

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings_model,
    collection_name=COLLECTION_NAME
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

print(
    f"Connected to vector store. "
    f"Items indexed: {vectorstore._collection.count()}"
)


#=======================================
# Step 1: Query Reformulation Chain
#=======================================

contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given a chat history and the latest user question, "
        "rewrite the question into a standalone question that "
        "can be understood without the chat history. "
        "Do not answer the question. "
        "If the question is already standalone, return it as is."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

reformulate_chain = (
    contextualize_prompt
    | llm_brain
    | StrOutputParser()
)


#=======================================
# Step 2: Format Retrieved Documents
#=======================================

def format_docs(docs):

    parts = []

    for doc in docs:
        source_id = doc.metadata["id"]
        parts.append(
            f"[Source: {source_id}]\n{doc.page_content}"
        )

    return "\n\n".join(parts)


#=======================================
# Step 3: QA Prompt + Generation Chain
#=======================================

SYSTEM_INSTRUCTIONS = """You are a customer support assistant
for Dandes Electronics.

Answer using ONLY the retrieved context below.
Do not invent information or use outside knowledge.

If the context does not contain enough information,
your entire response must be exactly this sentence:
I don't know based on the available information.

Do not add any explanation to the refusal.

If documents disagree, use the latest applicable policy
and mention its effective date when available.

Include relevant source IDs in your answer.

Context:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTIONS),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_chain = qa_prompt | llm_brain | StrOutputParser()


#=======================================
# Step 4: Manage Chat History + Full Turn
#=======================================

chat_history = []


def chat(user_message):

    # 4a: Rewrite follow-up questions using chat history
    if len(chat_history) == 0:
        standalone_question = user_message
    else:
        standalone_question = reformulate_chain.invoke({
            "input": user_message,
            "chat_history": chat_history
        })

    print("\nStandalone Question:", standalone_question)

    # 4b: Retrieve documents using the standalone question
    retrieved_docs = retriever.invoke(standalone_question)

    print("\nRetrieved Documents:")

    for doc in retrieved_docs:
        print(
            f"  - {doc.metadata['id']} | "
            f"{doc.metadata['doc_type']}"
        )

    context = format_docs(retrieved_docs)

    # 4c: Generate an answer using context and conversation history
    answer = qa_chain.invoke({
        "input": user_message,
        "chat_history": chat_history,
        "context": context
    })

    # 4d: Collect source IDs
    sources = list(dict.fromkeys(
        doc.metadata["id"] for doc in retrieved_docs
    ))

    print("\nAssistant:", answer)
    print("Sources:", ", ".join(sources))

    # 4e: Save the real user question and answer
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=answer))

    return {
        "query": user_message,
        "standalone_question": standalone_question,
        "answer": answer,
        "sources": sources
    }


#=======================================
# Step 5: Test a Multi-Turn Conversation
#=======================================

if __name__ == "__main__":

    print("\nAsk questions about Dandes Electronics.")
    print("Type 'exit' to stop.\n")

    while True:

        user_message = input("You: ").strip()

        if user_message.lower() == "exit":
            print("Conversation ended.")
            break

        if not user_message:
            continue

        result = chat(user_message)
