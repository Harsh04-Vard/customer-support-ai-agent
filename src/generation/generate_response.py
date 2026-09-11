import os

import faiss
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")


# --------------------------------------------------
# 2. Load FAISS index and historical data
# --------------------------------------------------

INDEX_FILE = "data/processed/applesupport.faiss"
METADATA_FILE = "data/processed/applesupport_metadata.csv"

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)
metadata = pd.read_csv(METADATA_FILE)

print(f"Historical cases: {len(metadata)}")


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 4. Create Groq client
# --------------------------------------------------

client = Groq(api_key=api_key)


# --------------------------------------------------
# 5. Retrieve similar historical cases
# --------------------------------------------------

def retrieve_cases(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
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
# 6. Generate grounded response
# --------------------------------------------------

def generate_response(customer_message):

    retrieved_cases = retrieve_cases(
        customer_message,
        top_k=5
    )

    historical_context = ""

    for i, (_, row) in enumerate(
        retrieved_cases.iterrows(),
        start=1
    ):
        historical_context += f"""
Historical case {i}:

Customer:
{row["customer_text"]}

AppleSupport response:
{row["brand_response"]}

Similarity:
{row["similarity"]:.4f}

"""


    prompt = f"""
You are a customer support assistant for Apple Support.

A customer has sent the following message:

CUSTOMER MESSAGE:
{customer_message}

Below are similar historical AppleSupport conversations retrieved
from a real customer-support dataset.

HISTORICAL CASES:
{historical_context}
IMPORTANT: The retrieved responses are your ONLY source of factual
support guidance. Do not rely on your general knowledge.


Your task is to write a helpful customer-support response.

Rules:
1. Use ONLY information and troubleshooting actions supported by
   the historical AppleSupport responses provided below.
2. Do NOT add troubleshooting steps from your general knowledge.
3. Do NOT invent company policies, refunds, guarantees, links,
   procedures, or service requirements.
4. Do NOT introduce specific settings, thresholds, or instructions
   unless they are supported by the historical responses.
5. If the retrieved cases do not contain enough information to give
   a specific solution, say that you need more information or direct
   the customer to contact Apple Support.
6. Do NOT claim that you performed an action that you cannot perform.
7. Keep the response concise, professional, and directly relevant.
8. Do not mention the historical dataset or retrieval process.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful customer support assistant "
                    "that generates grounded responses."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content, retrieved_cases


# --------------------------------------------------
# 7. Test with one customer message
# --------------------------------------------------

customer_message = input(
    "\nEnter a customer message: "
).strip()

print("\n" + "=" * 80)
print("CUSTOMER MESSAGE")
print("=" * 80)
print(customer_message)


response, retrieved_cases = generate_response(
    customer_message
)


print("\n" + "=" * 80)
print("RETRIEVED HISTORICAL CASES")
print("=" * 80)

for i, (_, row) in enumerate(
    retrieved_cases.iterrows(),
    start=1
):
    print(f"\n{i}. Similarity: {row['similarity']:.4f}")
    print(f"Customer: {row['customer_text']}")
    print(f"Response: {row['brand_response']}")


print("\n" + "=" * 80)
print("GROUNDED GROQ RESPONSE")
print("=" * 80)
print(response)