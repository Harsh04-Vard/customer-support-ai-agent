import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
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
# 2. Build TF-IDF index
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

history_matrix = vectorizer.fit_transform(
    history["customer_text"]
)


# ---------------------------------------------------------
# 3. Retrieval function
# ---------------------------------------------------------

def retrieve(query, query_id, top_k=5):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        history_matrix
    ).flatten()

    # Prevent exact self-retrieval
    same_query = history["customer_tweet_id"].astype(str) == str(query_id)
    similarities[same_query.values] = -1

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = history.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results


# ---------------------------------------------------------
# 4. Test retrieval
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("RETRIEVAL EVALUATION")
print("=" * 80)


for i, (_, query_row) in enumerate(test.head(10).iterrows(), start=1):

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
        