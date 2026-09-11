import pandas as pd

FILE_PATH = "data/raw/twcs.csv"

print("Loading a sample of the dataset...")

df = pd.read_csv(
    FILE_PATH,
    nrows=10000
)

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DATA TYPES ---")
print(df.dtypes)