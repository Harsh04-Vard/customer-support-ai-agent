import pandas as pd
import re

INPUT_PATH = "data/processed/applesupport_conversations.csv"
OUTPUT_PATH = "data/processed/applesupport_clean.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", "", text)

    # Remove common HTML entities
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&amp;", "&")

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("Loading conversations...")

df = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(df):,} conversations.")

df["customer_text"] = df["customer_text"].apply(clean_text)
df["brand_response"] = df["brand_response"].apply(clean_text)

# Remove rows where either side became empty
df = df[
    (df["customer_text"].str.len() > 0)
    & (df["brand_response"].str.len() > 0)
]

df.to_csv(OUTPUT_PATH, index=False)

print(f"Clean conversations: {len(df):,}")
print(f"Saved to: {OUTPUT_PATH}")

print("\n--- CLEAN SAMPLE ---")

for _, row in df.head(10).iterrows():
    print("\nCUSTOMER:")
    print(row["customer_text"])

    print("APPLE SUPPORT:")
    print(row["brand_response"])

    print("-" * 60)