import pandas as pd

INPUT_PATH = "data/processed/applesupport_clean.csv"
OUTPUT_PATH = "data/golden/intent_sample_200.csv"

print("Loading cleaned conversations...")

df = pd.read_csv(INPUT_PATH)

# Reproducible random sample
sample = df.sample(
    n=200,
    random_state=42
).copy()

# Keep only the fields we need for manual intent discovery
sample = sample[
    [
        "customer_tweet_id",
        "customer_text",
        "brand_response",
    ]
]

# Add an empty label column
sample["intent"] = ""

sample.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Created sample with {len(sample)} messages.")
print(f"Saved to: {OUTPUT_PATH}")

print("\nFirst 10 messages:\n")

for i, text in enumerate(sample["customer_text"].head(10), start=1):
    print(f"{i}. {text}")