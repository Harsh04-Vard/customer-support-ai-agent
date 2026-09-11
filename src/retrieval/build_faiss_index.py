import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/processed/applesupport_clean.csv"
INDEX_FILE = "data/processed/applesupport.faiss"
METADATA_FILE = "data/processed/applesupport_metadata.csv"


# ---------------------------------------------------------
# 1. Load historical conversations
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df = df.dropna(
    subset=["customer_tweet_id", "customer_text", "brand_response"]
).reset_index(drop=True)

print(f"Historical conversations: {len(df)}")


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

print("\nCreating embeddings...")

embeddings = model.encode(
    df["customer_text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print(f"Embedding shape: {embeddings.shape}")


# ---------------------------------------------------------
# 4. Build FAISS index
# ---------------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors.")


# ---------------------------------------------------------
# 5. Save index + metadata
# ---------------------------------------------------------

faiss.write_index(index, INDEX_FILE)

df.to_csv(METADATA_FILE, index=False)

print("\nSaved:")
print(INDEX_FILE)
print(METADATA_FILE)

print("\nDone!")