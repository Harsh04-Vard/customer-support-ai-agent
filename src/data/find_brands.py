import pandas as pd
from collections import Counter

FILE_PATH = "data/raw/twcs.csv"

# Count how many tweets each author wrote as a brand.
# inbound=False means the tweet was sent by the brand/support account.
brand_counts = Counter()

print("Scanning dataset for brand accounts...")

for chunk in pd.read_csv(
    FILE_PATH,
    usecols=["author_id", "inbound"],
    chunksize=100_000
):
    brand_tweets = chunk[chunk["inbound"] == False]

    brand_counts.update(brand_tweets["author_id"].dropna())

print("\nTop 30 author IDs producing brand replies:\n")

for author_id, count in brand_counts.most_common(30):
    print(f"{author_id}: {count:,} replies")