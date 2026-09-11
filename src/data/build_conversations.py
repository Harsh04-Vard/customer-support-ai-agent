import pandas as pd

FILE_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/applesupport_conversations.csv"

BRAND = "AppleSupport"

print("Loading tweet data...")

# We only need these columns
columns = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

df = pd.read_csv(
    FILE_PATH,
    usecols=columns
)

print(f"Total tweets loaded: {len(df):,}")

# AppleSupport tweets
brand_df = df[df["author_id"] == BRAND].copy()

print(f"AppleSupport tweets: {len(brand_df):,}")

# Customer tweets
customer_df = df[df["inbound"] == True].copy()

print(f"Customer tweets: {len(customer_df):,}")

# Rename columns so the merge is easier to understand
customer_df = customer_df[
    ["tweet_id", "author_id", "created_at", "text"]
].rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "author_id": "customer_id",
        "created_at": "customer_created_at",
        "text": "customer_text",
    }
)

brand_df = brand_df[
    ["tweet_id", "created_at", "text", "in_response_to_tweet_id"]
].rename(
    columns={
        "tweet_id": "brand_tweet_id",
        "created_at": "brand_created_at",
        "text": "brand_response",
    }
)

# Connect Apple's response to the customer tweet it replies to
conversations = brand_df.merge(
    customer_df,
    left_on="in_response_to_tweet_id",
    right_on="customer_tweet_id",
    how="inner",
)

# Keep only useful columns
conversations = conversations[
    [
        "customer_tweet_id",
        "customer_id",
        "customer_created_at",
        "customer_text",
        "brand_tweet_id",
        "brand_created_at",
        "brand_response",
    ]
]

# Remove empty messages
conversations = conversations.dropna(
    subset=["customer_text", "brand_response"]
)

# Save
conversations.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nCustomer → AppleSupport pairs: {len(conversations):,}")
print(f"Saved to: {OUTPUT_PATH}")

print("\n--- SAMPLE CONVERSATIONS ---")

for _, row in conversations.head(10).iterrows():
    print("\nCUSTOMER:")
    print(row["customer_text"])

    print("APPLE SUPPORT:")
    print(row["brand_response"])

    print("-" * 60)