# INSTALLATION / RUN NOTES
# Run commands from the project root. Install core dependencies with:
#   python -m pip install -r requirements.txt
# Create a root .env file using .env.example and provide your own API key.
# Do not commit .env or expose your API key.
# Optional RAGAS evaluation dependencies are listed in requirements-ragas.txt.


#=======================================
# V2 Multi-Index RAG — Intent Router
# Classify Question and Select Indexes
#=======================================

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


#=======================================
# Setup
#=======================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-6-luna")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

# Do not set temperature=0; this model does not support it.
llm = ChatOpenAI(
    api_key=API_KEY,
    model=ANSWER_MODEL
)

ALL_INDEXES = [
    "policy_index",
    "product_index",
    "order_index",
    "ticket_index"
]

VALID_INTENTS = {
    "policy",
    "product",
    "order",
    "past_issue",
    "out_of_scope"
}


#=======================================
# Router Instructions
#=======================================

ROUTER_PROMPT = """
You are an intent router for Dandes Electronics customer support.

Available intents:
- policy: returns, refunds, delivery rules, shipping charges, COD
- product: product information and known product issues
- order: order procedures, address changes, tracking and damage reporting
- past_issue: previous resolved customer cases
- out_of_scope: questions the knowledge base cannot answer

Available indexes:
- policy_index: return and shipping policies
- product_index: product FAQs and known issues
- order_index: order procedure guides
- ticket_index: resolved customer support tickets

Rules:
1. Return eligibility and refund policy questions use policy_index.
2. General remote-pincode delivery times and free shipping use
   policy_index only.
3. Product questions and known issues use product_index.
4. A product issue plus a request for a previous case uses
   product_index and ticket_index.
5. Specific order procedures use order_index.
6. An address change after shipping plus a previous case uses
   order_index and ticket_index.
7. A damaged phone refund question asking for the official process
   and a previous refund case uses policy_index, order_index,
   and ticket_index.
8. A specific policy code such as RR-004 uses policy_index.
9. If the question is outside the available knowledge base,
   such as "Do you sell refrigerators?", return out_of_scope
   with an empty indexes list.
10. Select only the indexes needed to answer the question.
11. A question may have multiple intents and indexes.

Return only a valid JSON object in this format:
{
  "intents": ["product", "past_issue"],
  "indexes": ["product_index", "ticket_index"],
  "reason": "The question asks about a product problem and a previous case."
}
"""


#=======================================
# Parse JSON Response
#=======================================

def parse_router_response(response_text):

    response_text = response_text.strip()

    if response_text.startswith("```"):
        lines = response_text.splitlines()
        response_text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Router response does not contain JSON")

    result = json.loads(response_text[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Router response must be a JSON object")

    return result


#=======================================
# Validate Router Output
#=======================================

def validate_router_result(result):

    if not all(key in result for key in [
        "intents", "indexes", "reason"
    ]):
        raise ValueError("Router JSON is missing required keys")

    intents = result["intents"]
    indexes = result["indexes"]
    reason = result["reason"]

    if not isinstance(intents, list):
        raise ValueError("intents must be a list")

    if not isinstance(indexes, list):
        raise ValueError("indexes must be a list")

    if not isinstance(reason, str):
        raise ValueError("reason must be a string")

    if not all(
        isinstance(intent, str) and intent in VALID_INTENTS
        for intent in intents
    ):
        raise ValueError("Invalid intent returned")

    if not all(
        isinstance(index, str) and index in ALL_INDEXES
        for index in indexes
    ):
        raise ValueError("Invalid index returned")

    intents = list(dict.fromkeys(intents))
    indexes = list(dict.fromkeys(indexes))

    if "out_of_scope" in intents:
        return {
            "intents": ["out_of_scope"],
            "indexes": [],
            "reason": reason
        }

    if not intents or not indexes:
        raise ValueError("Incomplete routing response")

    return {
        "intents": intents,
        "indexes": indexes,
        "reason": reason
    }


#=======================================
# Route the Question
#=======================================

def route_question(question):

    try:
        response = llm.invoke(
            ROUTER_PROMPT
            + "\n\nCustomer question:\n"
            + question
        )

        result = parse_router_response(response.content)

        return validate_router_result(result)

    except Exception as error:

        print("Router fallback:", str(error))

        return {
            "intents": ["fallback"],
            "indexes": ALL_INDEXES.copy(),
            "reason": "Router failed; searching all indexes."
        }


#=======================================
# Seven Assignment Test Questions
#=======================================

TEST_QUESTIONS = [
    {
        "question": (
            "Can I return an opened laptop on day 8 if I have "
            "the original box and all accessories?"
        ),
        "expected": ["policy_index"]
    },
    {
        "question": (
            "My Galaxy Buds3 Pro left earbud won't connect. Is "
            "this a known issue, and has the same problem been "
            "resolved for another customer? What fixed it?"
        ),
        "expected": ["product_index", "ticket_index"]
    },
    {
        "question": (
            "How long does delivery take to a remote pincode, "
            "and is it free?"
        ),
        "expected": ["policy_index"]
    },
    {
        "question": (
            "A customer wants to change the address after the "
            "order shipped. What is the official process, and "
            "do we have a resolved case like this?"
        ),
        "expected": ["order_index", "ticket_index"]
    },
    {
        "question": "What does policy RR-004 say?",
        "expected": ["policy_index"]
    },
    {
        "question": (
            "A customer received a damaged phone and wants a "
            "refund instead of a replacement. What is the "
            "official process, and do we have a previous "
            "resolved case where a refund was issued?"
        ),
        "expected": [
            "policy_index",
            "order_index",
            "ticket_index"
        ]
    },
    {
        "question": "Do you sell refrigerators?",
        "expected": []
    }
]


#=======================================
# Run Router Tests
#=======================================

if __name__ == "__main__":

    print("\nDANDES ELECTRONICS — INTENT ROUTER TEST")
    print("=" * 75)

    passed = 0

    for number, test in enumerate(TEST_QUESTIONS, start=1):

        result = route_question(test["question"])

        predicted = result["indexes"]
        expected = test["expected"]

        passed_test = set(predicted) == set(expected)

        print(f"\nTest {number}: {test['question']}")
        print("Predicted indexes:", predicted)
        print("Expected indexes: ", expected)
        print("Intents:", result["intents"])
        print("Reason:", result["reason"])
        print("Result:", "PASS" if passed_test else "CHECK")

        if passed_test:
            passed += 1

    print("\nRouter tests passed:", passed, "/", len(TEST_QUESTIONS))
