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
# PART 1: LOAD ENVIRONMENT VARIABLES
#=======================================

# Load environment variables from the .env file
load_dotenv()

# Read the API key and configuration values
API_KEY = os.getenv("OPENAI_API_KEY")
ANSWER_MODEL = os.getenv("ANSWER_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Check whether the API key is available
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

# Check whether the required configuration values are available
if not all([ANSWER_MODEL, EMBEDDING_MODEL, CHROMA_PATH, COLLECTION_NAME]):
    raise ValueError("Missing model or Chroma configuration in .env")

print("OPENAI KEY SUCCESSFULLY LOADED")


#=======================================
# PART 2: CONNECT TO THE LANGUAGE MODELS
#=======================================

# Initialize the LLM used for question rewriting and answer generation
llm_brain = ChatOpenAI(
    api_key=API_KEY,
    model=ANSWER_MODEL,
    max_tokens=300
)

# Initialize the embedding model to convert questions into vectors
embeddings_model = OpenAIEmbeddings(
    api_key=API_KEY,
    model=EMBEDDING_MODEL
)


#=======================================
# PART 3: CONNECT TO THE EXISTING VECTOR STORE
#=======================================

# Connect to the Chroma collection created by Program 1
vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings_model,
    collection_name=COLLECTION_NAME
)

# Create a retriever that returns the five most relevant documents
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# Display the number of documents stored in the collection
print(
    f"Connected to vector store. "
    f"Items indexed: {vectorstore._collection.count()}"
)


#=======================================
# PART 4: QUERY REFORMULATION CHAIN
#=======================================

# Rewrite follow-up questions into standalone questions
contextualize_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the conversation history and the latest user question, "
        "rewrite the question into a standalone question that can be "
        "understood without the conversation history. "
        "Do not answer the question. "
        "Preserve the original meaning and important details. "
        "If the question is already standalone, return it unchanged."
    ),

    # Insert previous user questions and assistant answers
    MessagesPlaceholder("chat_history"),

    # Insert the latest user question
    ("human", "{input}")
])

# Build the query reformulation chain using LCEL
# Prompt -> LLM -> Plain text
reformulate_chain = (
    contextualize_prompt | llm_brain | StrOutputParser()
)


#=======================================
# PART 5: FORMAT RETRIEVED DOCUMENTS
#=======================================

# Convert retrieved documents into one context string
def format_docs(docs):

    # Store the formatted documents
    parts = []

    # Process each retrieved document
    for doc in docs:

        # Read the source ID from document metadata
        source_id = doc.metadata["id"]

        # Include the source ID and document content
        parts.append(
            f"[Source: {source_id}]\n{doc.page_content}"
        )

    # Combine all documents with blank lines
    return "\n\n".join(parts)


#=======================================
# PART 6: QA PROMPT + GENERATION CHAIN
#=======================================

# Define the rules for answering customer questions
SYSTEM_INSTRUCTIONS = """You are a customer support assistant
for Dandes Electronics.

Answer using ONLY the retrieved context provided below.
Do not invent information or use outside knowledge.

If the context does not contain enough information to answer
the question, your entire response must be exactly:
I don't know based on the available information.

Do not add any explanation or other text to the refusal.

If documents disagree, use the latest applicable policy
and mention its effective date when available.

Include relevant source IDs in your answer.

Use the conversation history to understand follow-up questions.
However, use the retrieved context as the source of factual answers.

Context:
{context}
"""

# Build the question-answering prompt
qa_prompt = ChatPromptTemplate.from_messages([
    # Add system instructions and retrieved context
    ("system", SYSTEM_INSTRUCTIONS),

    # Add the conversation history
    MessagesPlaceholder("chat_history"),

    # Add the current user question
    ("human", "{input}")
])

# Build the answer-generation chain using LCEL
# Prompt -> LLM -> Plain text
qa_chain = qa_prompt | llm_brain | StrOutputParser()


#=======================================
# PART 7: MANAGE CHAT HISTORY + FULL TURN
#=======================================

# Store conversation messages during this program's execution
chat_history = []

# Define the exact refusal required by the assignment
REFUSAL_MESSAGE = "I don't know based on the available information."


# Define a function to process one conversational turn
def chat(user_message):

    #-----------------------------------
    # Step 7a: Reformulate the question
    #-----------------------------------

    # Use the original question when there is no conversation history
    if len(chat_history) == 0:
        standalone_question = user_message

    else:
        # Rewrite the follow-up question using previous messages
        standalone_question = reformulate_chain.invoke({
            "input": user_message,
            "chat_history": chat_history
        }).strip()

    # Display the standalone question
    print("\nStandalone Question:", standalone_question)


    #-----------------------------------
    # Step 7b: Retrieve relevant documents
    #-----------------------------------

    # Search Chroma using the standalone question
    retrieved_docs = retriever.invoke(standalone_question)

    # Display the retrieved document IDs and types
    print("\nRetrieved Documents:")

    for doc in retrieved_docs:
        print(
            f"  - {doc.metadata['id']} | "
            f"{doc.metadata['doc_type']}"
        )

    # Format the retrieved documents as context
    context = format_docs(retrieved_docs)


    #-----------------------------------
    # Step 7c: Generate the answer
    #-----------------------------------

    # Send the current question, history, and context to the LLM
    answer = qa_chain.invoke({
        "input": user_message,
        "chat_history": chat_history,
        "context": context
    }).strip()

    # Normalize the refusal if the LLM includes the required sentence
    # Note: this does not independently verify whether an answer is supported
    if REFUSAL_MESSAGE in answer:
        answer = REFUSAL_MESSAGE


    #-----------------------------------
    # Step 7d: Collect source IDs
    #-----------------------------------

    # Collect retrieved source IDs and remove duplicates
    sources = list(dict.fromkeys(
        doc.metadata["id"] for doc in retrieved_docs
    ))

    # Display the generated answer and retrieved source IDs
    print("\nAssistant:", answer)
    print("Sources:", ", ".join(sources) if sources else "None")


    #-----------------------------------
    # Step 7e: Update conversation history
    #-----------------------------------

    # Save the original user question
    chat_history.append(
        HumanMessage(content=user_message)
    )

    # Save the assistant's final answer
    chat_history.append(
        AIMessage(content=answer)
    )

    # Return the result as a dictionary
    return {
        "query": user_message,
        "standalone_question": standalone_question,
        "answer": answer,
        "sources": sources
    }


#=======================================
# PART 8: TEST THE CONVERSATION
#=======================================

# Run the interactive conversation only when this file is executed
if __name__ == "__main__":

    # Display instructions
    print("\nAsk questions about Dandes Electronics.")
    print("Type 'exit' to stop.\n")

    # Continue until the user exits
    while True:

        # Read the user's question
        user_message = input("You: ").strip()

        # Stop when the user types exit
        if user_message.lower() == "exit":
            print("Conversation ended.")
            break

        # Skip empty input
        if not user_message:
            continue

        # Process the current conversational turn
        result = chat(user_message)
