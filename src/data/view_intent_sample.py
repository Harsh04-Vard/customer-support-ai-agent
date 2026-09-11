import pandas as pd

INPUT_PATH = "data/golden/intent_sample_200.csv"

df = pd.read_csv(INPUT_PATH)

print("\n" + "=" * 80)
print("INTENT DISCOVERY SAMPLE")
print("=" * 80)

for i, row in df.iterrows():
    print(f"\n{i + 1}. {row['customer_text']}")

print("\n" + "=" * 80)
