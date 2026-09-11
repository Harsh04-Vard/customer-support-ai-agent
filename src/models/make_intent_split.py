import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/golden/intent_sample_200.csv"

TRAIN_FILE = "data/golden/intent_train.csv"
TEST_FILE = "data/golden/intent_test.csv"

# Load labelled data
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["intent"])

# Stratified split
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["intent"]
)

# Save the exact split
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("=" * 60)
print("INTENT DATA SPLIT")
print("=" * 60)

print(f"\nTotal: {len(df)}")
print(f"Train: {len(train_df)}")
print(f"Test:  {len(test_df)}")

print("\nTrain distribution:")
print(train_df["intent"].value_counts())

print("\nTest distribution:")
print(test_df["intent"].value_counts())

print("\nSaved:")
print(TRAIN_FILE)
print(TEST_FILE)