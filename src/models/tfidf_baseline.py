import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


TRAIN_FILE = "data/golden/intent_train.csv"
TEST_FILE = "data/golden/intent_test.csv"


# Load the fixed train/test split
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df["customer_text"]
y_train = train_df["intent"]

X_test = test_df["customer_text"]
y_test = test_df["intent"]


# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# Predict
y_pred = model.predict(X_test_tfidf)


# Evaluate
accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

print(f"\nTraining examples: {len(train_df)}")
print(f"Test examples:     {len(test_df)}")

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)