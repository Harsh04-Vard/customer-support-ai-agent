#Customer Support AI Agent

An end-to-end customer support AI agent built using historical AppleSupport conversations from the Customer Support on Twitter dataset.

The system takes a customer message, predicts the support intent, retrieves similar historical AppleSupport cases, generates a grounded response using an LLM, and decides whether the issue should be automatically handled or escalated to a human agent.

---

## 1. Problem Framing

### What does "good" mean for AppleSupport?

For a customer support agent, a good response should:

- Correctly understand the customer's issue.
- Retrieve historical cases that are relevant to the problem.
- Use previous AppleSupport responses as evidence.
- Avoid inventing unsupported troubleshooting instructions.
- Be concise and directly useful to the customer.
- Avoid confidently handling risky or sensitive cases.
- Escalate when the system does not have enough evidence.

The project therefore focuses on four core capabilities:

1. Intent classification
2. Historical case retrieval
3. Grounded response generation
4. Auto-handle vs human escalation

### What I chose not to build

To keep the project focused on the core support-agent problem, I did not build:

- A production customer-support ticketing integration.
- Automatic actions on customer accounts.
- A fully trained safety/risk classifier.
- A large-scale human annotation platform.
- A production deployment infrastructure.
- A complex multi-agent architecture.
- A custom foundation model.

The emphasis is on demonstrating a complete, reproducible support-agent pipeline rather than production infrastructure.

---

## 2. System Architecture

The complete pipeline is:

```text
Customer Message
       |
       v
Text Preprocessing
       |
       v
Intent Classification
(TF-IDF + Logistic Regression)
       |
       v
Historical Case Retrieval
(Sentence Transformer + FAISS)
       |
       v
Top Similar Historical Cases
       |
       v
Grounded LLM Generation
(Groq)
       |
       v
Escalation Policy
       |
       +--------------------+
       |                    |
       v                    v
 AUTO-HANDLE            ESCALATE
```

The Streamlit application provides an interface over the same pipeline.

---

## 3. Dataset

### Dataset

The project uses the:

**Customer Support on Twitter (TWCS)** dataset.

The dataset contains customer-support interactions from Twitter and includes fields such as:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

The dataset does not contain an explicit brand column.

Instead, support-brand accounts were identified from repeated outbound account IDs.

### Brand Selection

I selected **AppleSupport** as the target support brand.

AppleSupport was selected because it provides a large number of historical support responses, giving the system enough examples for both retrieval and experimentation while keeping the project focused on a single brand.

---

## 4. Data Processing

The raw dataset is processed to create customer-response pairs.

The main processing steps are:

1. Inspect the raw dataset.
2. Identify repeated outbound support accounts.
3. Select AppleSupport interactions.
4. Match customer messages with AppleSupport responses.
5. Clean URLs and Twitter mentions.
6. Decode HTML entities.
7. Normalize whitespace.
8. Remove empty customer/response pairs.
9. Explore recurring issue terms.
10. Create a manually labelled evaluation sample.

After cleaning, the AppleSupport corpus contains approximately **105K usable customer-response pairs**.

---

## 5. Golden Evaluation Set

A manually labelled golden set of **200 examples** was created, satisfying the requested 150–250 example range.

The examples were sampled from the cleaned AppleSupport customer-support data using a fixed random seed.

Each example was manually assigned one of the following 13 intents:

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

The 200 examples were then split into:

- **160 training examples**
- **40 test examples**

using stratified sampling with a fixed random seed.

### Golden-set limitation

The golden set is intentionally small because the assignment requires manual labelling.

The classes are also imbalanced. The largest classes contain substantially more examples than some of the smaller classes.

Because of this, Macro-F1 is reported in addition to accuracy.

---

## 6. Intent Classification

Two classification approaches were evaluated.

### Baseline 1: Majority Class

The trivial baseline predicts the most frequent intent for every customer message.

Results:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority Baseline | 15.0% | 0.0201 |

The majority classifier provides a simple lower bound and performs poorly on minority intents.

### Baseline 2: TF-IDF + Logistic Regression

The simple ML baseline uses:

- TF-IDF features
- Unigrams and bigrams
- English stop-word removal
- Logistic Regression

Results:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority Baseline | 15.0% | 0.0201 |
| TF-IDF + Logistic Regression | 32.5% | 0.1843 |

The TF-IDF classifier improves over the trivial baseline, but the Macro-F1 shows that performance remains uneven across the intent classes.

---

## 7. Retrieval

The project compares lexical retrieval with semantic retrieval.

### TF-IDF Retrieval

TF-IDF vectors are created for historical customer messages.

Cosine similarity is used to identify similar historical cases.

This provides a lightweight lexical retrieval baseline.

### Semantic Retrieval

The semantic retrieval system uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Customer messages are converted into 384-dimensional embeddings.

The normalized embeddings are stored in a FAISS `IndexFlatIP` index.

Because the embeddings are normalized, inner product corresponds to cosine similarity.

The resulting index contains approximately 105K historical AppleSupport customer-response examples.

---

## 8. Retrieval Evaluation

The retrieval benchmark uses:

- 160 training examples as the retrieval corpus
- 40 test examples as queries

The test query itself is excluded from retrieval.

### Metric

Recall@K is defined here as:

> Whether at least one of the top K retrieved examples has the same manually labelled intent as the query.

This is an **intent-consistency retrieval metric**.

It is not a direct measurement of whether the retrieved response is the best answer to the customer.

### Results

| Retrieval Method | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| TF-IDF | 27.5% | 55.0% | 67.5% |
| FAISS + Sentence Transformer | 55.0% | 77.5% | 85.0% |

Semantic retrieval substantially improves over the lexical TF-IDF retrieval baseline.

---

## 9. What Is Misleading About My Headline Number?

The strongest retrieval result is:

> **85.0% Recall@5**

This number is useful, but it is also potentially misleading if presented as an end-to-end support accuracy number.

The metric only checks whether one of the top five retrieved examples has the **same intent** as the test query.

It does not prove that:

- the retrieved response is correct,
- the retrieved response is complete,
- the generated answer is correct,
- the customer issue is resolved.

For example, two cases can both concern battery problems while one historical response is much more useful than the other.

Therefore, the headline should be stated precisely as:

> **85.0% intent-consistency Recall@5 on a 40-example test benchmark.**

It should not be described as:

> "85% of customer questions are answered correctly."

This distinction is important because retrieval quality and end-to-end answer quality are different problems.

---

## 10. Grounded Response Generation

The retrieved historical cases are passed to a Groq-hosted LLM.

The prompt provides the model with:

- The customer message
- The retrieved historical customer messages
- The corresponding AppleSupport responses

The model is explicitly instructed to use the retrieved AppleSupport responses as its factual support evidence.

### Grounding rules

The generation prompt instructs the model to:

- Use only information supported by the retrieved responses.
- Avoid adding troubleshooting steps from general knowledge.
- Avoid inventing company policies.
- Avoid inventing refunds or guarantees.
- Avoid unsupported settings or numerical thresholds.
- Avoid claiming that an action was performed when it was not.
- Ask for more information or direct the customer to support when evidence is insufficient.
- Keep the response concise and relevant.

This is intended to reduce unsupported LLM-generated troubleshooting.

### Example

For a message such as:

> My iPhone keeps freezing after the latest iOS update.

the system:

1. Predicts an intent.
2. Retrieves similar historical cases.
3. Uses the historical responses as grounding evidence.
4. Generates a customer-facing response.
5. Applies the escalation policy.

---

## 11. Escalation

The system includes a lightweight escalation policy.

### Safety / Serious Incidents

Messages containing serious safety-related terms can be escalated.

Examples include:

- exploded
- swollen
- smoke
- fire
- burning
- overheating
- dangerous
- injury
- hurt

### Security / Fraud

Examples include:

- hacked
- fraud
- unauthorized charge
- stolen

### High-Risk Intents

The following intents are treated conservatively:

- Account / Security
- Purchase / Billing / Store
- Hardware / Accessories

### Low Retrieval Confidence

If the top retrieval similarity is below the configured confidence threshold, the system escalates instead of confidently auto-handling the issue.

Otherwise, a low-risk issue with sufficient retrieval confidence can be marked:

```text
AUTO-HANDLE
```

The escalation component is intentionally implemented as a simple baseline rather than a fully trained risk model.

---

## 12. LLM-as-Judge Evaluation

The project includes an LLM-based evaluation approach for generated responses.

The judge receives:

1. Customer message
2. Retrieved historical cases
3. Generated response

The judge evaluates the response on:

### Relevance

Does the response directly address the customer's issue?

### Groundedness

Are the factual support instructions supported by the retrieved historical responses?

### Helpfulness

Does the response provide useful next steps without unnecessary content?

### Unsupported Claims

Does the response introduce information that is not supported by the retrieved evidence?

A structured scoring rubric can be used for each dimension.

### Judge limitation

An LLM judge should not be treated as ground truth.

A fluent response can still receive a high automated score despite containing unsupported information.

For this reason, human review is required as a separate validation step.

---

## 13. Human Agreement With the Judge

A subset of generated responses should be manually reviewed and compared against the LLM judge.

The purpose is to measure whether the automated judge agrees with a human evaluator.

The comparison can use:

- Agreement percentage
- Cohen's kappa for categorical judgements

The human review should focus especially on:

- groundedness,
- unsupported claims,
- relevance,
- and whether escalation was appropriate.

### Important limitation

The manually reviewed sample is small and should therefore be treated as evidence about judge reliability rather than a statistically definitive estimate of production performance.

No fabricated agreement number is reported here.

---

## 14. Top 5 Failure Modes

### 1. Surface-Word Matching

TF-IDF retrieval can match examples because they share words even when the underlying problem differs.

**Hypothesis:** semantic embeddings are better at capturing the meaning of short customer-support messages.

---

### 2. Missing Conversation Context

Twitter support messages are often short and depend on previous conversation turns.

Messages such as:

> "it's doing that again"

can be difficult to classify or retrieve correctly without earlier turns.

**Hypothesis:** using thread-level context or a conversation summary would improve classification and retrieval.

---

### 3. Ambiguous Intent Boundaries

Some customer messages naturally overlap multiple intent categories.

For example, an issue occurring after an iOS update could involve both software/update behaviour and device performance.

**Hypothesis:** multi-label or hierarchical intent classification could better represent these cases.

---

### 4. Incomplete Historical Responses

Historical support responses are not guaranteed to contain a complete solution.

Some responses primarily ask the customer to send a direct message or refer them to another support resource.

**Hypothesis:** retrieval should consider response quality in addition to message similarity.

---

### 5. LLM Unsupported Troubleshooting

During development, a general generation prompt caused the LLM to introduce troubleshooting advice that was not supported by the retrieved historical responses.

For example, an early battery response introduced specific battery-health thresholds and settings that were not supported by the retrieved evidence.

The generation prompt was subsequently tightened to make the historical responses the required source of factual support guidance.

**Hypothesis:** evidence verification or citation-aware generation would further reduce unsupported advice.

---

## 15. Evaluation Harness

The repository contains scripts for evaluating the main components separately.

### Intent evaluation

```bash
python src/models/majority_baseline.py
python src/models/tfidf_baseline.py
```

### TF-IDF retrieval

```bash
python src/retrieval/evaluate_tfidf.py
```

### FAISS retrieval

```bash
python src/retrieval/evaluate_faiss.py
```

The evaluation scripts use the fixed golden train/test split.

This makes the main reported baseline and retrieval numbers reproducible without changing the evaluation examples.

---

## 16. Decision Log

### 1. Selected AppleSupport

AppleSupport was selected because it provides a large historical support corpus while keeping the project focused on one brand.

### 2. Created a 200-example golden set

200 manually labelled examples satisfy the required 150–250 range while keeping annotation manageable.

### 3. Used 13 support intents

The taxonomy was designed around recurring issue types observed during exploration of AppleSupport conversations.

### 4. Used stratified train/test splitting

Stratification ensures that the smaller intent classes are represented in both training and testing where possible.

### 5. Reported Macro-F1

The golden set is imbalanced, so accuracy alone could hide poor performance on minority classes.

### 6. Added a majority baseline

The majority classifier provides a trivial baseline against which the ML classifier can be compared.

### 7. Added TF-IDF + Logistic Regression

This provides a simple, interpretable ML baseline before introducing semantic models.

### 8. Added TF-IDF retrieval

TF-IDF retrieval provides a lexical baseline for comparing retrieval quality.

### 9. Used Sentence Transformers

Semantic embeddings were selected because customers can describe the same issue using different words.

### 10. Used FAISS

The cleaned corpus contains approximately 105K examples, making vector indexing useful for efficient retrieval.

### 11. Used retrieval-grounded generation

Historical AppleSupport responses provide domain-specific evidence for the LLM.

### 12. Restricted LLM generation

The prompt was tightened after observing unsupported troubleshooting generated from general model knowledge.

### 13. Added conservative escalation

Safety-sensitive, security-sensitive, billing-related and low-confidence cases should not be blindly auto-handled.

### 14. Excluded the raw dataset from GitHub

The raw dataset is large and is downloaded separately rather than committed to the repository.

### 15. Excluded the generated FAISS index

The generated FAISS index is approximately 155 MB, exceeding GitHub's standard 100 MB file limit.

It is therefore generated locally from the processed data.

---

## 17. Reproducibility

### Requirements

The project uses:

- Python 3.12
- uv
- Pandas
- NumPy
- scikit-learn
- PyTorch
- Sentence Transformers
- FAISS
- Groq
- Streamlit

### Install dependencies

From the project root:

```bash
uv sync
```

### Dataset

Download the Customer Support on Twitter dataset and place the raw file at:

```text
data/raw/twcs.csv
```

The raw dataset is intentionally excluded from GitHub.

### Environment variable

Create a local `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit the `.env` file.

### Process the data

Run the data-processing scripts in the repository to:

1. Inspect the dataset.
2. Identify brand accounts.
3. Extract AppleSupport interactions.
4. Build customer-response pairs.
5. Clean the conversations.
6. Create the golden evaluation set.

### Build the FAISS index

```bash
python src/retrieval/build_faiss_index.py
```

This creates:

```text
data/processed/applesupport.faiss
data/processed/applesupport_metadata.csv
```

The FAISS index is generated locally and is intentionally not stored in GitHub.

### Run the pipeline

```bash
python src/pipeline.py
```

### Run the Streamlit application

```bash
uv run streamlit run app/app.py
```

---

## 18. Repository Structure

```text
customer-support-ai-agent/
│
├── app/
│   └── app.py
│
├── data/
│   ├── golden/
│   │   ├── intent_sample_200.csv
│   │   ├── intent_train.csv
│   │   └── intent_test.csv
│   │
│   └── processed/
│       ├── applesupport_tweets.csv
│       ├── applesupport_conversations.csv
│       ├── applesupport_clean.csv
│       └── applesupport_metadata.csv
│
├── src/
│   ├── data/
│   ├── models/
│   ├── retrieval/
│   ├── generation/
│   ├── pipeline.py
│   └── hiver_support_agent/
│
├── tests/
│
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
└── uv.lock
```

The following files are intentionally not committed:

```text
.env
.venv/
data/raw/twcs.csv
data/processed/applesupport.faiss
```

---

## 19. Streamlit Application

The Streamlit application provides a simple support-agent interface.

It displays:

- Customer message input
- Predicted intent
- Auto-handle / escalation decision
- Top retrieval similarity
- Generated response
- Retrieved historical cases
- System information

Launch it using:

```bash
uv run streamlit run app/app.py
```

Example input:

```text
My iPhone keeps freezing after the latest iOS update.
```

The application then runs the complete support pipeline.

---

## 20. Results Summary

### Intent Classification

| Approach | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority Baseline | 15.0% | 0.0201 |
| TF-IDF + Logistic Regression | 32.5% | 0.1843 |

### Retrieval

| Approach | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| TF-IDF | 27.5% | 55.0% | 67.5% |
| FAISS + Sentence Transformer | 55.0% | 77.5% | 85.0% |

The semantic retrieval approach performs substantially better than the lexical retrieval baseline on the intent-consistency benchmark.

---

## 21. Limitations

The project has several important limitations.

### Small labelled dataset

Only 200 examples were manually labelled.

### Class imbalance

The intent classes are not evenly represented.

### Short Twitter messages

Some messages lack enough context to determine the issue accurately.

### Conversation leakage risk

The golden dataset is sampled from Twitter support conversations, and multiple turns can belong to the same underlying conversation.

The current benchmark uses row-level splitting, so some conversational overlap may remain.

### Retrieval metric limitation

Recall@K measures intent consistency rather than answer correctness.

### Historical response quality

Historical support responses can themselves be incomplete.

### Rule-based escalation

The escalation component is a baseline and is not a trained risk model.

### LLM variability

Generated responses depend on the selected LLM and prompt.

---

## 22. What I Would Do With One More Week

If another week were available, I would prioritize the following:

### 1. Expand the golden set

Increase the manually labelled evaluation set and improve representation of smaller intent classes.

### 2. Use conversation context

Retrieve and classify using the complete conversation thread rather than isolated tweets.

### 3. Improve intent classification

Compare the TF-IDF classifier with embedding-based classifiers and stronger supervised models.

### 4. Add reranking

Retrieve a larger candidate set using FAISS and then use a cross-encoder reranker to select the most useful historical responses.

### 5. Improve answer evaluation

Create a larger human-labelled response-quality benchmark.

### 6. Strengthen grounding

Require the generated answer to identify which retrieved evidence supports each factual recommendation.

### 7. Improve escalation

Train a dedicated risk/escalation classifier instead of relying primarily on rules.

### 8. Add regression tests

Maintain a set of known difficult examples to prevent future changes from reintroducing known failure modes.

---

## 23. Conclusion

This project demonstrates an end-to-end customer-support AI agent built around historical AppleSupport conversations.

The system combines:

- Manual intent annotation
- Classical ML classification
- Semantic retrieval
- FAISS vector search
- Retrieval-grounded LLM generation
- Conservative escalation
- Interactive Streamlit UI

The strongest retrieval result is:

> **85.0% intent-consistency Recall@5**

on the 40-example test benchmark.

This number is intentionally not presented as end-to-end customer-support accuracy. The main lesson from the evaluation is that semantic retrieval substantially improves over lexical retrieval, while response correctness and grounding still require separate evaluation.

The project therefore emphasizes not only the headline metric, but also its limitations, failure modes, grounding strategy, escalation behaviour, and reproducibility.

The system takes a customer message, predicts the support intent, retrieves similar historical AppleSupport cases, generates a grounded response using an LLM, and decides whether the issue should be automatically handled or escalated to a human agent.

---

## 1. Problem Framing

### What does "good" mean for AppleSupport?

For a customer support agent, a good response should:

- Correctly understand the customer's issue.
- Retrieve historical cases that are relevant to the problem.
- Use previous AppleSupport responses as evidence.
- Avoid inventing unsupported troubleshooting instructions.
- Be concise and directly useful to the customer.
- Avoid confidently handling risky or sensitive cases.
- Escalate when the system does not have enough evidence.

The project therefore focuses on four core capabilities:

1. Intent classification
2. Historical case retrieval
3. Grounded response generation
4. Auto-handle vs human escalation

### What I chose not to build

To keep the project focused on the core support-agent problem, I did not build:

- A production customer-support ticketing integration.
- Automatic actions on customer accounts.
- A fully trained safety/risk classifier.
- A large-scale human annotation platform.
- A production deployment infrastructure.
- A complex multi-agent architecture.
- A custom foundation model.

The emphasis is on demonstrating a complete, reproducible support-agent pipeline rather than production infrastructure.

---

## 2. System Architecture

The complete pipeline is:

```text
Customer Message
       |
       v
Text Preprocessing
       |
       v
Intent Classification
(TF-IDF + Logistic Regression)
       |
       v
Historical Case Retrieval
(Sentence Transformer + FAISS)
       |
       v
Top Similar Historical Cases
       |
       v
Grounded LLM Generation
(Groq)
       |
       v
Escalation Policy
       |
       +--------------------+
       |                    |
       v                    v
 AUTO-HANDLE            ESCALATE
```

The Streamlit application provides an interface over the same pipeline.

---

## 3. Dataset

### Dataset

The project uses the:

**Customer Support on Twitter (TWCS)** dataset.

The dataset contains customer-support interactions from Twitter and includes fields such as:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

The dataset does not contain an explicit brand column.

Instead, support-brand accounts were identified from repeated outbound account IDs.

### Brand Selection

I selected **AppleSupport** as the target support brand.

AppleSupport was selected because it provides a large number of historical support responses, giving the system enough examples for both retrieval and experimentation while keeping the project focused on a single brand.

---

## 4. Data Processing

The raw dataset is processed to create customer-response pairs.

The main processing steps are:

1. Inspect the raw dataset.
2. Identify repeated outbound support accounts.
3. Select AppleSupport interactions.
4. Match customer messages with AppleSupport responses.
5. Clean URLs and Twitter mentions.
6. Decode HTML entities.
7. Normalize whitespace.
8. Remove empty customer/response pairs.
9. Explore recurring issue terms.
10. Create a manually labelled evaluation sample.

After cleaning, the AppleSupport corpus contains approximately **105K usable customer-response pairs**.

---

## 5. Golden Evaluation Set

A manually labelled golden set of **200 examples** was created, satisfying the requested 150–250 example range.

The examples were sampled from the cleaned AppleSupport customer-support data using a fixed random seed.

Each example was manually assigned one of the following 13 intents:

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

The 200 examples were then split into:

- **160 training examples**
- **40 test examples**

using stratified sampling with a fixed random seed.

### Golden-set limitation

The golden set is intentionally small because the assignment requires manual labelling.

The classes are also imbalanced. The largest classes contain substantially more examples than some of the smaller classes.

Because of this, Macro-F1 is reported in addition to accuracy.

---

## 6. Intent Classification

Two classification approaches were evaluated.

### Baseline 1: Majority Class

The trivial baseline predicts the most frequent intent for every customer message.

Results:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority Baseline | 15.0% | 0.0201 |

The majority classifier provides a simple lower bound and performs poorly on minority intents.

### Baseline 2: TF-IDF + Logistic Regression

The simple ML baseline uses:

- TF-IDF features
- Unigrams and bigrams
- English stop-word removal
- Logistic Regression

Results:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority Baseline | 15.0% | 0.0201 |
| TF-IDF + Logistic Regression | 32.5% | 0.1843 |

The TF-IDF classifier improves over the trivial baseline, but the Macro-F1 shows that performance remains uneven across the intent classes.

---

## 7. Retrieval

The project compares lexical retrieval with semantic retrieval.

### TF-IDF Retrieval

TF-IDF vectors are created for historical customer messages.

Cosine similarity is used to identify similar historical cases.

This provides a lightweight lexical retrieval baseline.

### Semantic Retrieval

The semantic retrieval system uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Customer messages are converted into 384-dimensional embeddings.

The normalized embeddings are stored in a FAISS `IndexFlatIP` index.

Because the embeddings are normalized, inner product corresponds to cosine similarity.

The resulting index contains approximately 105K historical AppleSupport customer-response examples.

---

## 8. Retrieval Evaluation

The retrieval benchmark uses:

- 160 training examples as the retrieval corpus
- 40 test examples as queries

The test query itself is excluded from retrieval.

### Metric

Recall@K is defined here as:

> Whether at least one of the top K retrieved examples has the same manually labelled intent as the query.

This is an **intent-consistency retrieval metric**.

It is not a direct measurement of whether the retrieved response is the best answer to the customer.

### Results

| Retrieval Method | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| TF-IDF | 27.5% | 55.0% | 67.5% |
| FAISS + Sentence Transformer | 55.0% | 77.5% | 85.0% |

Semantic retrieval substantially improves over the lexical TF-IDF retrieval baseline.

---

## 9. What Is Misleading About My Headline Number?

The strongest retrieval result is:

> **85.0% Recall@5**

This number is useful, but it is also potentially misleading if presented as an end-to-end support accuracy number.

The metric only checks whether one of the top five retrieved examples has the **same intent** as the test query.

It does not prove that:

- the retrieved response is correct,
- the retrieved response is complete,
- the generated answer is correct,
- the customer issue is resolved.

For example, two cases can both concern battery problems while one historical response is much more useful than the other.

Therefore, the headline should be stated precisely as:

> **85.0% intent-consistency Recall@5 on a 40-example test benchmark.**

It should not be described as:

> "85% of customer questions are answered correctly."

This distinction is important because retrieval quality and end-to-end answer quality are different problems.

---

## 10. Grounded Response Generation

The retrieved historical cases are passed to a Groq-hosted LLM.

The prompt provides the model with:

- The customer message
- The retrieved historical customer messages
- The corresponding AppleSupport responses

The model is explicitly instructed to use the retrieved AppleSupport responses as its factual support evidence.

### Grounding rules

The generation prompt instructs the model to:

- Use only information supported by the retrieved responses.
- Avoid adding troubleshooting steps from general knowledge.
- Avoid inventing company policies.
- Avoid inventing refunds or guarantees.
- Avoid unsupported settings or numerical thresholds.
- Avoid claiming that an action was performed when it was not.
- Ask for more information or direct the customer to support when evidence is insufficient.
- Keep the response concise and relevant.

This is intended to reduce unsupported LLM-generated troubleshooting.

### Example

For a message such as:

> My iPhone keeps freezing after the latest iOS update.

the system:

1. Predicts an intent.
2. Retrieves similar historical cases.
3. Uses the historical responses as grounding evidence.
4. Generates a customer-facing response.
5. Applies the escalation policy.

---

## 11. Escalation

The system includes a lightweight escalation policy.

### Safety / Serious Incidents

Messages containing serious safety-related terms can be escalated.

Examples include:

- exploded
- swollen
- smoke
- fire
- burning
- overheating
- dangerous
- injury
- hurt

### Security / Fraud

Examples include:

- hacked
- fraud
- unauthorized charge
- stolen

### High-Risk Intents

The following intents are treated conservatively:

- Account / Security
- Purchase / Billing / Store
- Hardware / Accessories

### Low Retrieval Confidence

If the top retrieval similarity is below the configured confidence threshold, the system escalates instead of confidently auto-handling the issue.

Otherwise, a low-risk issue with sufficient retrieval confidence can be marked:

```text
AUTO-HANDLE
```

The escalation component is intentionally implemented as a simple baseline rather than a fully trained risk model.

---

## 12. LLM-as-Judge Evaluation

The project includes an LLM-based evaluation approach for generated responses.

The judge receives:

1. Customer message
2. Retrieved historical cases
3. Generated response

The judge evaluates the response on:

### Relevance

Does the response directly address the customer's issue?

### Groundedness

Are the factual support instructions supported by the retrieved historical responses?

### Helpfulness

Does the response provide useful next steps without unnecessary content?

### Unsupported Claims

Does the response introduce information that is not supported by the retrieved evidence?

A structured scoring rubric can be used for each dimension.

### Judge limitation

An LLM judge should not be treated as ground truth.

A fluent response can still receive a high automated score despite containing unsupported information.

For this reason, human review is required as a separate validation step.

---

## 13. Human Agreement With the Judge

A subset of generated responses should be manually reviewed and compared against the LLM judge.

The purpose is to measure whether the automated judge agrees with a human evaluator.

The comparison can use:

- Agreement percentage
- Cohen's kappa for categorical judgements

The human review should focus especially on:

- groundedness,
- unsupported claims,
- relevance,
- and whether escalation was appropriate.

### Important limitation

The manually reviewed sample is small and should therefore be treated as evidence about judge reliability rather than a statistically definitive estimate of production performance.

No fabricated agreement number is reported here.

---

## 14. Top 5 Failure Modes

### 1. Surface-Word Matching

TF-IDF retrieval can match examples because they share words even when the underlying problem differs.

**Hypothesis:** semantic embeddings are better at capturing the meaning of short customer-support messages.

---

### 2. Missing Conversation Context

Twitter support messages are often short and depend on previous conversation turns.

Messages such as:

> "it's doing that again"

can be difficult to classify or retrieve correctly without earlier turns.

**Hypothesis:** using thread-level context or a conversation summary would improve classification and retrieval.

---

### 3. Ambiguous Intent Boundaries

Some customer messages naturally overlap multiple intent categories.

For example, an issue occurring after an iOS update could involve both software/update behaviour and device performance.

**Hypothesis:** multi-label or hierarchical intent classification could better represent these cases.

---

### 4. Incomplete Historical Responses

Historical support responses are not guaranteed to contain a complete solution.

Some responses primarily ask the customer to send a direct message or refer them to another support resource.

**Hypothesis:** retrieval should consider response quality in addition to message similarity.

---

### 5. LLM Unsupported Troubleshooting

During development, a general generation prompt caused the LLM to introduce troubleshooting advice that was not supported by the retrieved historical responses.

For example, an early battery response introduced specific battery-health thresholds and settings that were not supported by the retrieved evidence.

The generation prompt was subsequently tightened to make the historical responses the required source of factual support guidance.

**Hypothesis:** evidence verification or citation-aware generation would further reduce unsupported advice.

---

## 15. Evaluation Harness

The repository contains scripts for evaluating the main components separately.

### Intent evaluation

```bash
python src/models/majority_baseline.py
python src/models/tfidf_baseline.py
```

### TF-IDF retrieval

```bash
python src/retrieval/evaluate_tfidf.py
```

### FAISS retrieval

```bash
python src/retrieval/evaluate_faiss.py
```

The evaluation scripts use the fixed golden train/test split.

This makes the main reported baseline and retrieval numbers reproducible without changing the evaluation examples.

---

## 16. Decision Log

### 1. Selected AppleSupport

AppleSupport was selected because it provides a large historical support corpus while keeping the project focused on one brand.

### 2. Created a 200-example golden set

200 manually labelled examples satisfy the required 150–250 range while keeping annotation manageable.

### 3. Used 13 support intents

The taxonomy was designed around recurring issue types observed during exploration of AppleSupport conversations.

### 4. Used stratified train/test splitting

Stratification ensures that the smaller intent classes are represented in both training and testing where possible.

### 5. Reported Macro-F1

The golden set is imbalanced, so accuracy alone could hide poor performance on minority classes.

### 6. Added a majority baseline

The majority classifier provides a trivial baseline against which the ML classifier can be compared.

### 7. Added TF-IDF + Logistic Regression

This provides a simple, interpretable ML baseline before introducing semantic models.

### 8. Added TF-IDF retrieval

TF-IDF retrieval provides a lexical baseline for comparing retrieval quality.

### 9. Used Sentence Transformers

Semantic embeddings were selected because customers can describe the same issue using different words.

### 10. Used FAISS

The cleaned corpus contains approximately 105K examples, making vector indexing useful for efficient retrieval.

### 11. Used retrieval-grounded generation

Historical AppleSupport responses provide domain-specific evidence for the LLM.

### 12. Restricted LLM generation

The prompt was tightened after observing unsupported troubleshooting generated from general model knowledge.

### 13. Added conservative escalation

Safety-sensitive, security-sensitive, billing-related and low-confidence cases should not be blindly auto-handled.

### 14. Excluded the raw dataset from GitHub

The raw dataset is large and is downloaded separately rather than committed to the repository.

### 15. Excluded the generated FAISS index

The generated FAISS index is approximately 155 MB, exceeding GitHub's standard 100 MB file limit.

It is therefore generated locally from the processed data.

---

## 17. Reproducibility

### Requirements

The project uses:

- Python 3.12
- uv
- Pandas
- NumPy
- scikit-learn
- PyTorch
- Sentence Transformers
- FAISS
- Groq
- Streamlit

### Install dependencies

From the project root:

```bash
uv sync
```

### Dataset

Download the Customer Support on Twitter dataset and place the raw file at:

```text
data/raw/twcs.csv
```

The raw dataset is intentionally excluded from GitHub.

### Environment variable

Create a local `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit the `.env` file.

### Process the data

Run the data-processing scripts in the repository to:

1. Inspect the dataset.
2. Identify brand accounts.
3. Extract AppleSupport interactions.
4. Build customer-response pairs.
5. Clean the conversations.
6. Create the golden evaluation set.

### Build the FAISS index

```bash
python src/retrieval/build_faiss_index.py
```

This creates:

```text
data/processed/applesupport.faiss
data/processed/applesupport_metadata.csv
```

The FAISS index is generated locally and is intentionally not stored in GitHub.

### Run the pipeline

```bash
python src/pipeline.py
```

### Run the Streamlit application

```bash
uv run streamlit run app/app.py
```

---

## 18. Repository Structure

```text
customer-support-ai-agent/
│
├── app/
│   └── app.py
│
├── data/
│   ├── golden/
│   │   ├── intent_sample_200.csv
│   │   ├── intent_train.csv
│   │   └── intent_test.csv
│   │
│   └── processed/
│       ├── applesupport_tweets.csv
│       ├── applesupport_conversations.csv
│       ├── applesupport_clean.csv
│       └── applesupport_metadata.csv
│
├── src/
│   ├── data/
│   ├── models/
│   ├── retrieval/
│   ├── generation/
│   ├── pipeline.py
│   └── hiver_support_agent/
│
├── tests/
│
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
└── uv.lock
```

The following files are intentionally not committed:

```text
.env
.venv/
data/raw/twcs.csv
data/processed/applesupport.faiss
```

---

## 19. Streamlit Application

The Streamlit application provides a simple support-agent interface.

It displays:

- Customer message input
- Predicted intent
- Auto-handle / escalation decision
- Top retrieval similarity
- Generated response
- Retrieved historical cases
- System information

Launch it using:

```bash
uv run streamlit run app/app.py
```

Example input:

```text
My iPhone keeps freezing after the latest iOS update.
```

The application then runs the complete support pipeline.

---

## 20. Results Summary

### Intent Classification

| Approach | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority Baseline | 15.0% | 0.0201 |
| TF-IDF + Logistic Regression | 32.5% | 0.1843 |

### Retrieval

| Approach | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|
| TF-IDF | 27.5% | 55.0% | 67.5% |
| FAISS + Sentence Transformer | 55.0% | 77.5% | 85.0% |

The semantic retrieval approach performs substantially better than the lexical retrieval baseline on the intent-consistency benchmark.

---

## 21. Limitations

The project has several important limitations.

### Small labelled dataset

Only 200 examples were manually labelled.

### Class imbalance

The intent classes are not evenly represented.

### Short Twitter messages

Some messages lack enough context to determine the issue accurately.

### Conversation leakage risk

The golden dataset is sampled from Twitter support conversations, and multiple turns can belong to the same underlying conversation.

The current benchmark uses row-level splitting, so some conversational overlap may remain.

### Retrieval metric limitation

Recall@K measures intent consistency rather than answer correctness.

### Historical response quality

Historical support responses can themselves be incomplete.

### Rule-based escalation

The escalation component is a baseline and is not a trained risk model.

### LLM variability

Generated responses depend on the selected LLM and prompt.

---

## 22. What I Would Do With One More Week

If another week were available, I would prioritize the following:

### 1. Expand the golden set

Increase the manually labelled evaluation set and improve representation of smaller intent classes.

### 2. Use conversation context

Retrieve and classify using the complete conversation thread rather than isolated tweets.

### 3. Improve intent classification

Compare the TF-IDF classifier with embedding-based classifiers and stronger supervised models.

### 4. Add reranking

Retrieve a larger candidate set using FAISS and then use a cross-encoder reranker to select the most useful historical responses.

### 5. Improve answer evaluation

Create a larger human-labelled response-quality benchmark.

### 6. Strengthen grounding

Require the generated answer to identify which retrieved evidence supports each factual recommendation.

### 7. Improve escalation

Train a dedicated risk/escalation classifier instead of relying primarily on rules.

### 8. Add regression tests

Maintain a set of known difficult examples to prevent future changes from reintroducing known failure modes.

---

## 23. Conclusion

This project demonstrates an end-to-end customer-support AI agent built around historical AppleSupport conversations.

The system combines:

- Manual intent annotation
- Classical ML classification
- Semantic retrieval
- FAISS vector search
- Retrieval-grounded LLM generation
- Conservative escalation
- Interactive Streamlit UI

The strongest retrieval result is:

> **85.0% intent-consistency Recall@5**

on the 40-example test benchmark.

This number is intentionally not presented as end-to-end customer-support accuracy. The main lesson from the evaluation is that semantic retrieval substantially improves over lexical retrieval, while response correctness and grounding still require separate evaluation.

The project therefore emphasizes not only the headline metric, but also its limitations, failure modes, grounding strategy, escalation behaviour, and reproducibility.
