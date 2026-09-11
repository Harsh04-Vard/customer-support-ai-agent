import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer


TRAIN_FILE = "data/golden/intent_train.csv"
TEST_FILE = "data/golden/intent_test.csv"


# ---------------------------------------------------------
# 1. Load train/test data
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training cases: {len(train_df)}")
print(f"Test queries:   {len(test_df)}")


# ---------------------------------------------------------
# 2. Load embedding model
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded.")


# ---------------------------------------------------------
# 3. Create embeddings ONLY for training cases
# ---------------------------------------------------------

print("\nCreating training embeddings...")

train_embeddings = model.encode(
    train_df["customer_text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"Embedding shape: {train_embeddings.shape}")


# ---------------------------------------------------------
# 4. Build FAISS index
# ---------------------------------------------------------

dimension = train_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(train_embeddings)

print(f"FAISS vectors: {index.ntotal}")


# ---------------------------------------------------------
# 5. Encode test queries
# ---------------------------------------------------------

print("\nCreating test embeddings...")

test_embeddings = model.encode(
    test_df["customer_text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# 6. Search Top 1, 3 and 5
# ---------------------------------------------------------

k = 5

similarities, indices = index.search(
    test_embeddings,
    k
)


# ---------------------------------------------------------
# 7. Calculate Recall@K
# ---------------------------------------------------------

recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0

for i in range(len(test_df)):

    true_intent = test_df.iloc[i]["intent"]

    retrieved_intents = train_df.iloc[
        indices[i]
    ]["intent"].tolist()

    if true_intent in retrieved_intents[:1]:
        recall_at_1 += 1

    if true_intent in retrieved_intents[:3]:
        recall_at_3 += 1

    if true_intent in retrieved_intents[:5]:
        recall_at_5 += 1


recall_at_1 /= len(test_df)
recall_at_3 /= len(test_df)
recall_at_5 /= len(test_df)


# ---------------------------------------------------------
# 8. Print results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FAISS RETRIEVAL EVALUATION")
print("=" * 60)

print(f"\nRecall@1: {recall_at_1:.4f}")
print(f"Recall@3: {recall_at_3:.4f}")
print(f"Recall@5: {recall_at_5:.4f}")


# ---------------------------------------------------------
# 9. Show examples
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("EXAMPLE RETRIEVALS")
print("=" * 80)

for i in range(min(5, len(test_df))):

    print("\n" + "-" * 80)

    print(f"Query: {test_df.iloc[i]['customer_text']}")
    print(f"True intent: {test_df.iloc[i]['intent']}")

    print("\nRetrieved:")

    for rank in range(5):

        row = train_df.iloc[indices[i][rank]]

        print(
            f"{rank + 1}. "
            f"[{similarities[i][rank]:.4f}] "
            f"[{row['intent']}] "
            f"{row['customer_text']}"
        )