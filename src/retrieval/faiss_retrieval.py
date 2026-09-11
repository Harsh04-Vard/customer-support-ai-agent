import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


INDEX_FILE = "data/processed/applesupport.faiss"
METADATA_FILE = "data/processed/applesupport_metadata.csv"


# ---------------------------------------------------------
# 1. Load FAISS index and metadata
# ---------------------------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)

metadata = pd.read_csv(METADATA_FILE)

print(f"FAISS vectors: {index.ntotal}")
print(f"Metadata rows: {len(metadata)}")


# ---------------------------------------------------------
# 2. Load embedding model
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model loaded.")


# ---------------------------------------------------------
# 3. Retrieval function
# ---------------------------------------------------------

def retrieve(query, top_k=5):

    # Convert query into an embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Search FAISS
    similarities, indices = index.search(
        query_embedding,
        top_k
    )

    results = metadata.iloc[indices[0]].copy()

    results["similarity"] = similarities[0]

    return results


# ---------------------------------------------------------
# 4. Test retrieval
# ---------------------------------------------------------

query = "My iPhone battery is draining very quickly"

results = retrieve(query, top_k=5)


print("\n" + "=" * 80)
print("FAISS SEMANTIC RETRIEVAL")
print("=" * 80)

print(f"\nQuery: {query}")

print("\nTop 5 historical cases:")

for rank, (_, row) in enumerate(
    results.iterrows(),
    start=1
):

    print(f"\n--- Result {rank} ---")
    print(f"Similarity: {row['similarity']:.4f}")
    print(f"Customer:   {row['customer_text']}")
    print(f"Response:   {row['brand_response']}")