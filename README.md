# Apple Support AI Agent

An end-to-end customer support AI agent built using historical AppleSupport conversations from the Customer Support on Twitter dataset.

The system classifies a customer issue, retrieves similar historical support cases, generates a grounded response using an LLM, and decides whether the issue can be automatically handled or should be escalated to a human agent.

---

## 1. Project Overview

The goal is to build a practical customer-support agent that uses historical support conversations rather than relying only on general-purpose LLM knowledge.

The pipeline is:

Customer Message
        ↓
Text Preprocessing
        ↓
Intent Classification
        ↓
Similar Case Retrieval
        ↓
Grounded LLM Response
        ↓
Auto-Handle / Escalate

The selected brand is AppleSupport.

---

## 2. Dataset

Dataset:

Customer Support on Twitter

Source:

Kaggle - Customer Support on Twitter

Dataset fields include:

- tweet_id
- author_id
- inbound
- created_at
- text
- response_tweet_id
- in_response_to_tweet_id

The dataset does not contain an explicit brand column.

Brand accounts were identified using repeated outbound account IDs.

AppleSupport was selected because it has a large number of historical support responses, providing enough examples for retrieval and evaluation.

---

## 3. Data Processing

The raw Twitter support dataset was processed through the following steps:

1. Identify brand support accounts.
2. Select AppleSupport conversations.
3. Match customer messages with AppleSupport responses.
4. Remove URLs and Twitter mentions.
5. Decode HTML entities.
6. Normalize whitespace.
7. Remove empty customer/response pairs.
8. Create a manually labelled intent dataset.

After cleaning, the AppleSupport corpus contains approximately 105K usable customer-response pairs.

---

## 4. Intent Classification

A manually labelled golden set of 200 customer messages was created.

The following 13 intent categories were used:

1. Battery
2. Software / OS Update
3. Keyboard / Input
4. App Problem
5. Device Performance
6. Connectivity
7. Calls / Audio
8. Messaging / Photos
9. Apple Services / Media
10. Account / Security
11. Purchase / Billing / Store
12. Hardware / Accessories
13. Other / Unclear

The dataset was split into:

- 160 training examples
- 40 test examples

The split used stratification with a fixed random seed.

### Important limitation

The manually labelled dataset is small and imbalanced.

Some classes contain only a small number of examples, so macro-F1 is reported alongside accuracy rather than relying on accuracy alone.

---

## 5. Intent Baselines

### Majority Baseline

The majority-class baseline predicts the most common intent for every message.

Results:

| Metric | Score |
|---|---:|
| Accuracy | 0.1500 |
| Macro-F1 | 0.0201 |

The majority baseline performs poorly across the minority intent classes.

### TF-IDF + Logistic Regression

The first ML baseline uses:

- TF-IDF features
- unigram and bigram features
- English stop-word removal
- Logistic Regression

Results:

| Metric | Score |
|---|---:|
| Accuracy | 0.3250 |
| Macro-F1 | 0.1843 |

This improves substantially over the majority baseline, although performance remains limited by the small labelled dataset.

---

## 6. Retrieval

Two retrieval approaches were implemented.

### TF-IDF Retrieval

A TF-IDF vectorizer with cosine similarity was used to retrieve historical customer-support examples.

### Semantic Retrieval

Semantic embeddings are generated using:

`sentence-transformers/all-MiniLM-L6-v2`

The embeddings are stored in a FAISS inner-product index.

The vectors are normalized, so inner product corresponds to cosine similarity.

---

## 7. Retrieval Results

Evaluation uses the 160 training examples as the retrieval corpus and the 40 test examples as queries.

The metric measures whether at least one retrieved example has the same manually labelled intent as the query.

| Retrieval Method | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| TF-IDF | 27.5% | 55.0% | 67.5% |
| FAISS + Sentence Transformer | 55.0% | 77.5% | 85.0% |

Semantic retrieval performs substantially better than TF-IDF retrieval on this benchmark.

---

## 8. Important Retrieval Limitation

The retrieval metric above is an **intent-consistency metric**.

It does not directly measure whether the retrieved support response is the best answer to the customer.

For example, a retrieved example may have the same intent but still contain a poor or incomplete support response.

Therefore:

> 85% Recall@5 should NOT be interpreted as "85% of customer questions are answered correctly."

This is one of the most important limitations of the headline retrieval number.

---

## 9. Grounded Response Generation

The retrieved historical cases are passed to a Groq-hosted LLM.

The model is instructed to use the historical AppleSupport responses as the source of factual troubleshooting guidance.

The generation prompt explicitly prevents the model from:

- inventing troubleshooting steps
- inventing company policies
- inventing refunds or guarantees
- introducing unsupported settings or thresholds
- claiming actions it cannot perform

If the retrieved cases do not contain enough information, the model is instructed to request more information or direct the customer to Apple Support.

This reduces hallucinated troubleshooting compared with a general-purpose generation prompt.

---

## 10. Escalation

The system includes a simple escalation policy.

Automatic escalation occurs for:

### Safety / Serious Incidents

Examples include:

- exploded
- swollen
- smoke
- fire
- burning
- overheating
- dangerous
- injury
- hacked
- fraud
- unauthorized charge

### High-Risk Intents

The following intents are automatically escalated:

- Account / Security
- Purchase / Billing / Store
- Hardware / Accessories

### Low Retrieval Confidence

If the top retrieval similarity is below the configured threshold, the system escalates instead of confidently generating a response.

Otherwise, low-risk issues with sufficient retrieval confidence are marked:

`AUTO-HANDLE`

---

## 11. Example

Example customer message:

> My iPhone keeps freezing after the latest iOS update.

The pipeline:

1. Predicts an intent.
2. Retrieves similar historical AppleSupport cases.
3. Uses those cases as grounding context.
4. Generates a concise customer-facing response.
5. Determines whether the issue can be auto-handled.

For safety-sensitive cases such as a battery that has exploded or is smoking, the system instead produces:

`ESCALATE`

with an explanation.

---

## 12. Streamlit Application

A Streamlit interface is provided in:

`app/app.py`

The UI displays:

- Customer message input
- Predicted intent
- Auto-handle / escalation decision
- Retrieval similarity
- Generated response
- Retrieved historical cases
- System information

Run the application with:

```bash
uv run streamlit run app/app.py