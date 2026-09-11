import os

import faiss
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from models.escalation_baseline import should_escalate
# --------------------------------------------------
# Configuration
# --------------------------------------------------

INDEX_FILE = "data/processed/applesupport.faiss"
METADATA_FILE = "data/processed/applesupport_metadata.csv"
TRAIN_FILE = "data/golden/intent_train.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


# --------------------------------------------------
# Load environment
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")


# --------------------------------------------------
# Load models and data
# --------------------------------------------------

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

metadata = pd.read_csv(METADATA_FILE)

print("Loading intent training data...")
train_df = pd.read_csv(TRAIN_FILE)

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Loading Groq client...")
groq_client = Groq(api_key=api_key)

print("All components loaded.")


# --------------------------------------------------
# Intent classifier
# --------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(
    train_df["customer_text"]
)

y_train = train_df["intent"]

intent_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

intent_model.fit(X_train, y_train)


def predict_intent(customer_message):

    message_vector = vectorizer.transform(
        [customer_message]
    )

    prediction = intent_model.predict(
        message_vector
    )

    return prediction[0]


# --------------------------------------------------
# FAISS retrieval
# --------------------------------------------------

def retrieve_cases(customer_message, top_k=5):

    query_embedding = embedding_model.encode(
        [customer_message],
        normalize_embeddings=True
    )

    similarities, indices = index.search(
        query_embedding,
        top_k
    )

    results = metadata.iloc[indices[0]].copy()

    results["similarity"] = similarities[0]

    return results


# --------------------------------------------------
# Groq response generation
# --------------------------------------------------

def generate_response(customer_message, retrieved_cases):

    historical_context = ""

    for i, (_, row) in enumerate(
        retrieved_cases.iterrows(),
        start=1
    ):

        historical_context += f"""
Historical case {i}

Customer:
{row["customer_text"]}

AppleSupport response:
{row["brand_response"]}

Similarity:
{row["similarity"]:.4f}

"""


    prompt = f"""
You are a careful customer support assistant for Apple Support.

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL APPLESUPPORT CASES:
{historical_context}

IMPORTANT:
The historical responses above are your ONLY source of
support guidance.

Rules:
1. Use only information supported by the historical responses.
2. Do not add troubleshooting steps from general knowledge.
3. Do not invent policies, refunds, guarantees, links, or procedures.
4. Do not invent specific settings or technical thresholds.
5. If the historical cases do not provide enough information,
   acknowledge that and ask for the relevant information.
6. Do not claim to have performed an action.
7. Keep the response concise and professional.
8. Do not mention the dataset or retrieval system.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate grounded customer support "
                    "responses."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# --------------------------------------------------
# Complete pipeline
# --------------------------------------------------

def run_pipeline(customer_message):

    # 1. Predict intent
    predicted_intent = predict_intent(
        customer_message
    )

    # 2. Retrieve historical cases
    retrieved_cases = retrieve_cases(
        customer_message,
        top_k=5
    )

    # 3. Get strongest retrieval score
    top_similarity = float(
        retrieved_cases.iloc[0]["similarity"]
    )

    # 4. Decide escalation
    decision, reason = should_escalate(
        customer_message,
        predicted_intent,
        top_similarity
    )

    # 5. Generate response
    response = generate_response(
        customer_message,
        retrieved_cases
    )

    return {
        "customer_message": customer_message,
        "intent": predicted_intent,
        "retrieved_cases": retrieved_cases,
        "response": response,
        "decision": decision,
        "reason": reason,
        "top_similarity": top_similarity,
    }


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    customer_message = input(
        "\nEnter a customer message: "
    ).strip()

    result = run_pipeline(customer_message)

    print("\n" + "=" * 80)
    print("PIPELINE RESULT")
    print("=" * 80)

    print("\nCustomer:")
    print(result["customer_message"])

    print("\nPredicted Intent:")
    print(result["intent"])

    print("\nTop Similarity:")
    print(f"{result['top_similarity']:.4f}")

    print("\nDecision:")
    print(result["decision"])

    print("\nReason:")
    print(result["reason"])

    print("\nRetrieved Cases:")

    for i, (_, row) in enumerate(
        result["retrieved_cases"].iterrows(),
        start=1
    ):
        print(
            f"\n{i}. "
            f"[{row['similarity']:.4f}] "
            f"[{row['customer_text']}]"
        )

    print("\n" + "=" * 80)
    print("GENERATED RESPONSE")
    print("=" * 80)

    print(result["response"])