import pandas as pd

FILE_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/applesupport_tweets.csv"

BRAND = "AppleSupport"

print(f"Extracting data for {BRAND}...")

chunks = []

for chunk in pd.read_csv(
    FILE_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
    chunksize=100_000,
):
    # Keep tweets written by AppleSupport
    brand_tweets = chunk[chunk["author_id"] == BRAND]

    if not brand_tweets.empty:
        chunks.append(brand_tweets)

if chunks:
    df = pd.concat(chunks, ignore_index=True)
else:
    df = pd.DataFrame()

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved {len(df):,} AppleSupport tweets.")
print(f"Output: {OUTPUT_PATH}")

print("\nInbound / outbound distribution:")
print(df["inbound"].value_counts())