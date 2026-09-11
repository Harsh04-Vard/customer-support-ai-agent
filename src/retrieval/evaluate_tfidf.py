import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TRAIN_FILE = "data/golden/intent_train.csv"
TEST_FILE = "data/golden/intent_test.csv"


# Load data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training cases: {len(train_df)}")
print(f"Test queries:   {len(test_df)}")


# Create TF-IDF embeddings
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

train_vectors = vectorizer.fit_transform(train_df["customer_text"])
test_vectors = vectorizer.transform(test_df["customer_text"])


# Calculate similarity
similarities = cosine_similarity(test_vectors, train_vectors)

k = 5

recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0


# Evaluate
for i in range(len(test_df)):

    # Get indices of top 5 most similar training examples
    top_indices = similarities[i].argsort()[::-1][:k]

    true_intent = test_df.iloc[i]["intent"]

    retrieved_intents = train_df.iloc[top_indices]["intent"].tolist()

    if true_intent in retrieved_intents[:1]:
        recall_at_1 += 1

    if true_intent in retrieved_intents[:3]:
        recall_at_3 += 1

    if true_intent in retrieved_intents[:5]:
        recall_at_5 += 1


# Convert to percentages
recall_at_1 /= len(test_df)
recall_at_3 /= len(test_df)
recall_at_5 /= len(test_df)


print("\n" + "=" * 60)
print("TF-IDF RETRIEVAL EVALUATION")
print("=" * 60)

print(f"\nRecall@1: {recall_at_1:.4f}")
print(f"Recall@3: {recall_at_3:.4f}")
print(f"Recall@5: {recall_at_5:.4f}")


# Show a few examples
print("\n" + "=" * 80)
print("EXAMPLE RETRIEVALS")
print("=" * 80)

for i in range(min(5, len(test_df))):

    top_indices = similarities[i].argsort()[::-1][:5]

    print("\n" + "-" * 80)
    print(f"Query: {test_df.iloc[i]['customer_text']}")
    print(f"True intent: {test_df.iloc[i]['intent']}")

    print("\nRetrieved:")

    for rank, idx in enumerate(top_indices, start=1):

        score = similarities[i][idx]
        intent = train_df.iloc[idx]["intent"]
        text = train_df.iloc[idx]["customer_text"]

        print(f"{rank}. [{score:.4f}] [{intent}] {text}")