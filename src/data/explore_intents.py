import pandas as pd
import re
from collections import Counter

INPUT_PATH = "data/processed/applesupport_clean.csv"

print("Loading cleaned conversations...")

df = pd.read_csv(INPUT_PATH)

print(f"Total conversations: {len(df):,}")


def tokenize(text):
    text = str(text).lower()

    # Keep words only
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9']+\b", text)

    return words


counter = Counter()

for text in df["customer_text"]:
    counter.update(tokenize(text))


print("\n--- TOP 50 CUSTOMER WORDS ---")

for word, count in counter.most_common(50):
    print(f"{word:20} {count:,}")


# Search for potentially useful issue-related terms
keywords = [
    "iphone",
    "ipad",
    "mac",
    "icloud",
    "itunes",
    "app",
    "update",
    "ios",
    "password",
    "account",
    "login",
    "payment",
    "charge",
    "refund",
    "subscription",
    "music",
    "wifi",
    "bluetooth",
    "battery",
    "screen",
    "keyboard",
    "email",
    "mail",
    "download",
    "purchase",
    "order",
    "crash",
    "error",
    "not working",
]

print("\n--- ISSUE-RELATED TERMS ---")

customer_text = df["customer_text"].str.lower()

for keyword in keywords:
    count = customer_text.str.contains(
        keyword,
        regex=False,
        na=False
    ).sum()

    print(f"{keyword:20} {count:,}")