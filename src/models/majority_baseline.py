import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

INPUT_FILE = "data/golden/intent_sample_200.csv"

# Load labelled data
df = pd.read_csv(INPUT_FILE)

# Remove any unlabeled rows
df = df.dropna(subset=["intent"])

X = df["customer_text"]
y = df["intent"]

# Same split will be used later for other intent models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Find the most common intent in training data
majority_class = y_train.value_counts().idxmax()

# Predict the same class for every test example
y_pred = [majority_class] * len(y_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("\n" + "=" * 60)
print("MAJORITY-CLASS BASELINE")
print("=" * 60)

print(f"\nTraining examples: {len(y_train)}")
print(f"Test examples:     {len(y_test)}")
print(f"Majority class:    {majority_class}")

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))