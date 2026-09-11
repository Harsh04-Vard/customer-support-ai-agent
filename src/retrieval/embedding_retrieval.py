import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


HISTORY_FILE = "data/processed/applesupport_clean.csv"
TEST_FILE = "data/golden/intent_test.csv"


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

history = pd.read_csv(HISTORY_FILE)
test = pd.read_csv(TEST_FILE)

history = history.dropna(
    subset=["customer_tweet_id", "customer_text", "brand_response"]
)

test = test.dropna(
    subset=["customer_tweet_id", "customer_text"]
)

print(f"Historical conversations: {len(history)}")
print(f"Test queries:             {len(test)}")


# ---------------------------------------------------------
# 2. Load embedding model
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded.")


# ---------------------------------------------------------
# 3. Create embeddings
# ---------------------------------------------------------

print("\nCreating historical embeddings...")

history_embeddings = model.encode(
    history["customer_text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embeddings created.")


# ---------------------------------------------------------
# 4. Retrieval function
# ---------------------------------------------------------

def retrieve(query, query_id, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        history_embeddings
    ).flatten()

    # Prevent exact self-retrieval
    same_query = (
        history["customer_tweet_id"].astype(str)
        == str(query_id)
    )

    similarities[same_query.values] = -1

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = history.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results


# ---------------------------------------------------------
# 5. Test on first 10 queries
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("SEMANTIC RETRIEVAL")
print("=" * 80)


for i, (_, query_row) in enumerate(
    test.head(10).iterrows(),
    start=1
):

    query = query_row["customer_text"]
    query_id = query_row["customer_tweet_id"]

    results = retrieve(
        query,
        query_id,
        top_k=5
    )

    print("\n" + "-" * 80)
    print(f"QUERY {i}")
    print("-" * 80)

    print(f"Customer: {query}")

    print("\nTop 5 retrieved cases:")

    for rank, (_, result) in enumerate(
        results.iterrows(),
        start=1
    ):

        print(
            f"\n{rank}. "
            f"[{result['similarity']:.4f}] "
            f"{result['customer_text']}"
        )