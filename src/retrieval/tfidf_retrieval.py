import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = "data/processed/applesupport_clean.csv"


# Load historical AppleSupport conversations
df = pd.read_csv(INPUT_FILE)

# Keep only rows with usable text
df = df.dropna(subset=["customer_text", "brand_response"])

print(f"Loaded {len(df)} historical conversations.")


# Create TF-IDF representation of customer messages
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(df["customer_text"])


def retrieve(query, top_k=5):
    """
    Retrieve the top-k historical conversations
    most similar to the customer query.
    """

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results


# Test query
query = "My iPhone battery is draining very quickly"

results = retrieve(query, top_k=5)

print("\n" + "=" * 80)
print("QUERY")
print("=" * 80)
print(query)

print("\n" + "=" * 80)
print("TOP 5 SIMILAR HISTORICAL CASES")
print("=" * 80)

for i, (_, row) in enumerate(results.iterrows(), start=1):
    print(f"\n--- Result {i} ---")
    print(f"Similarity: {row['similarity']:.4f}")
    print(f"Customer:   {row['customer_text']}")
    print(f"Response:   {row['brand_response']}")