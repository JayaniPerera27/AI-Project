# AI Resume Screener - Separate Google Colab Notebooks

Create these seven notebooks in Google Drive:

1. `01_data_preparation.ipynb`
2. `02_preprocessing_skill_extraction.ipynb`
3. `03_baseline_models.ipynb`
4. `04_bert_training.ipynb`
5. `05_feedback_fewshot.ipynb`
6. `06_evaluation.ipynb`
7. `07_demo.ipynb`

Each section below contains the cells for one notebook. Run cells from top to bottom.

---

# Notebook 1: `01_data_preparation.ipynb`

## Cell 1 - Install packages

```python
!pip -q install datasets pandas scikit-learn pyarrow
```

## Cell 2 - Mount Drive and create folders

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path

BASE_PATH = Path("/content/drive/MyDrive/AI-Resume-Screener")

for folder in [
    "data/raw",
    "data/processed",
    "models/word2vec",
    "models/bert",
    "outputs",
]:
    (BASE_PATH / folder).mkdir(parents=True, exist_ok=True)

print(BASE_PATH)
```

## Cell 3 - Load the paired resume-job dataset

```python
from datasets import load_dataset

fit_dataset = load_dataset("med2425/resume-job-fit-merged-v1")
fit_dataset
```

The dataset contains `resume`, `jd`, `label`, `source`, `resume_domain`, and
`jd_domain`. Labels are `Good Fit`, `Potential Fit`, and `No Fit`.

## Cell 4 - Convert to pandas and inspect

```python
import pandas as pd

train_df = fit_dataset["train"].to_pandas()
test_df = fit_dataset["test"].to_pandas()

print(train_df.shape, test_df.shape)
display(train_df.head(2))
print(train_df["label"].value_counts())
```

## Cell 5 - Clean invalid rows and convert labels to 0-1 scores

```python
LABEL_TO_SCORE = {
    "No Fit": 0.0,
    "Potential Fit": 0.5,
    "Good Fit": 1.0,
}

required_columns = ["resume", "jd", "label"]

train_df = train_df.dropna(subset=required_columns).copy()
test_df = test_df.dropna(subset=required_columns).copy()

train_df = train_df[train_df["label"].isin(LABEL_TO_SCORE)].copy()
test_df = test_df[test_df["label"].isin(LABEL_TO_SCORE)].copy()

train_df["score"] = train_df["label"].map(LABEL_TO_SCORE).astype(float)
test_df["score"] = test_df["label"].map(LABEL_TO_SCORE).astype(float)

train_df = train_df.drop_duplicates(subset=["resume", "jd"]).reset_index(drop=True)
test_df = test_df.drop_duplicates(subset=["resume", "jd"]).reset_index(drop=True)

print(train_df.shape, test_df.shape)
```

## Cell 6 - Create validation set

```python
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_df,
    test_size=0.10,
    random_state=42,
    stratify=train_df["label"],
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)
```

## Cell 7 - Save processed splits

```python
train_df.to_parquet(BASE_PATH / "data/processed/train.parquet", index=False)
val_df.to_parquet(BASE_PATH / "data/processed/validation.parquet", index=False)
test_df.to_parquet(BASE_PATH / "data/processed/test.parquet", index=False)

print("Saved all splits.")
```

---

# Notebook 2: `02_preprocessing_skill_extraction.ipynb`

## Cell 1 - Install packages and spaCy model

```python
!pip -q install pandas pyarrow spacy datasets
!python -m spacy download en_core_web_sm
```

## Cell 2 - Mount Drive and load data

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import pandas as pd

BASE_PATH = Path("/content/drive/MyDrive/AI-Resume-Screener")

train_df = pd.read_parquet(BASE_PATH / "data/processed/train.parquet")
val_df = pd.read_parquet(BASE_PATH / "data/processed/validation.parquet")
test_df = pd.read_parquet(BASE_PATH / "data/processed/test.parquet")
```

## Cell 3 - Preprocessing function

```python
import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_batch(texts):
    cleaned_texts = [basic_clean(text) for text in texts]
    results = []

    for doc in nlp.pipe(cleaned_texts, batch_size=64):
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
        ]
        results.append(" ".join(tokens))

    return results
```

## Cell 4 - Preprocess a manageable baseline sample

Processing all 90,000+ long pairs with spaCy can take a long time. Use a
representative sample for Word2Vec and baseline experiments.

```python
BASELINE_TRAIN_SIZE = min(15000, len(train_df))

baseline_train_df = train_df.sample(
    n=BASELINE_TRAIN_SIZE,
    random_state=42,
).reset_index(drop=True)

baseline_test_df = test_df.sample(
    n=min(2000, len(test_df)),
    random_state=42,
).reset_index(drop=True)

baseline_train_df["clean_resume"] = preprocess_batch(baseline_train_df["resume"])
baseline_train_df["clean_jd"] = preprocess_batch(baseline_train_df["jd"])

baseline_test_df["clean_resume"] = preprocess_batch(baseline_test_df["resume"])
baseline_test_df["clean_jd"] = preprocess_batch(baseline_test_df["jd"])
```

## Cell 5 - Skill extraction using PhraseMatcher

Extend this list using skills found in your datasets.

```python
from spacy.matcher import PhraseMatcher

SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#",
    "sql", "mysql", "postgresql", "mongodb", "redis",
    "machine learning", "deep learning", "natural language processing",
    "nlp", "computer vision", "data analysis", "data science",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "docker", "kubernetes", "git", "linux", "aws", "azure", "gcp",
    "react", "angular", "node.js", "django", "flask", "fastapi",
    "power bi", "tableau", "excel", "project management",
    "communication", "leadership", "problem solving",
]

skill_nlp = spacy.blank("en")
matcher = PhraseMatcher(skill_nlp.vocab, attr="LOWER")
matcher.add("SKILL", [skill_nlp.make_doc(skill) for skill in SKILLS])

def extract_skills(text):
    doc = skill_nlp(str(text))
    matches = matcher(doc)
    return sorted({doc[start:end].text.lower() for _, start, end in matches})

def compare_skills(resume, jd):
    resume_skills = set(extract_skills(resume))
    jd_skills = set(extract_skills(jd))

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": sorted(resume_skills & jd_skills),
        "missing_skills": sorted(jd_skills - resume_skills),
    }

display(compare_skills(
    baseline_test_df.loc[0, "resume"],
    baseline_test_df.loc[0, "jd"],
))
```

## Cell 6 - Save preprocessed baseline data

```python
baseline_train_df.to_parquet(
    BASE_PATH / "data/processed/baseline_train.parquet",
    index=False,
)
baseline_test_df.to_parquet(
    BASE_PATH / "data/processed/baseline_test.parquet",
    index=False,
)
```

## Optional Cell 7 - Load a skill-labelled dataset for NER evaluation

The main paired dataset does not contain human gold skill annotations. This
secondary dataset includes job and resume skill lists, but its annotations may
be automatically generated and should be described honestly in the report.

```python
from datasets import load_dataset

skill_dataset = load_dataset("batuhanmtl/job_resume_fit", split="train")
skill_df = skill_dataset.to_pandas()

print(skill_df.columns.tolist())
display(skill_df.head(1))
```

---

# Notebook 3: `03_baseline_models.ipynb`

## Cell 1 - Install packages and load data

```python
!pip -q install pandas pyarrow scikit-learn gensim

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import pandas as pd
import numpy as np

BASE_PATH = Path("/content/drive/MyDrive/AI-Resume-Screener")

train_df = pd.read_parquet(BASE_PATH / "data/processed/baseline_train.parquet")
test_df = pd.read_parquet(BASE_PATH / "data/processed/baseline_test.parquet")
```

## Cell 2 - TF-IDF baseline

Fit one shared TF-IDF vocabulary using training resumes and JDs. Do not fit a
new vectorizer separately for every pair.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
import joblib

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

tfidf.fit(
    pd.concat([train_df["clean_resume"], train_df["clean_jd"]])
)

resume_matrix = tfidf.transform(test_df["clean_resume"])
jd_matrix = tfidf.transform(test_df["clean_jd"])

test_df["tfidf_score"] = 1 - paired_cosine_distances(
    resume_matrix,
    jd_matrix,
)

joblib.dump(tfidf, BASE_PATH / "models/tfidf_vectorizer.joblib")
```

## Cell 3 - Train Word2Vec

```python
from gensim.models import Word2Vec

sentences = (
    train_df["clean_resume"].str.split().tolist()
    + train_df["clean_jd"].str.split().tolist()
)

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10,
    seed=42,
)

w2v_model.save(str(BASE_PATH / "models/word2vec/word2vec.model"))
```

## Cell 4 - Calculate Word2Vec pair scores

```python
def document_vector(text, model):
    vectors = [
        model.wv[word]
        for word in str(text).split()
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)

def cosine_score(vector_a, vector_b):
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)

test_df["word2vec_score"] = [
    cosine_score(
        document_vector(resume, w2v_model),
        document_vector(jd, w2v_model),
    )
    for resume, jd in zip(test_df["clean_resume"], test_df["clean_jd"])
]
```

## Cell 5 - Save baseline predictions

```python
test_df.to_parquet(
    BASE_PATH / "outputs/baseline_predictions.parquet",
    index=False,
)

display(test_df[["label", "score", "tfidf_score", "word2vec_score"]].head())
```

---

# Notebook 4: `04_bert_training.ipynb`

## Cell 1 - Enable GPU and install packages

Select `Runtime > Change runtime type > T4 GPU`, then run:

```python
!pip -q install transformers datasets accelerate evaluate scikit-learn
```

## Cell 2 - Mount Drive and check GPU

```python
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import torch

BASE_PATH = Path("/content/drive/MyDrive/AI-Resume-Screener")

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

## Cell 3 - Load data

Start with a smaller training sample to confirm the pipeline works. Increase
`TRAIN_SIZE` after the first successful run.

```python
import pandas as pd

train_df = pd.read_parquet(BASE_PATH / "data/processed/train.parquet")
val_df = pd.read_parquet(BASE_PATH / "data/processed/validation.parquet")

TRAIN_SIZE = min(20000, len(train_df))
VAL_SIZE = min(3000, len(val_df))

train_small = train_df.sample(TRAIN_SIZE, random_state=42).reset_index(drop=True)
val_small = val_df.sample(VAL_SIZE, random_state=42).reset_index(drop=True)

print(train_small.shape, val_small.shape)
```

## Cell 4 - Tokenize raw text

```python
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = Dataset.from_pandas(
    train_small[["resume", "jd", "score"]],
    preserve_index=False,
)
val_ds = Dataset.from_pandas(
    val_small[["resume", "jd", "score"]],
    preserve_index=False,
)

def tokenize_batch(batch):
    return tokenizer(
        batch["resume"],
        batch["jd"],
        truncation=True,
        max_length=512,
    )

train_ds = train_ds.map(tokenize_batch, batched=True)
val_ds = val_ds.map(tokenize_batch, batched=True)

train_ds = train_ds.rename_column("score", "labels")
val_ds = val_ds.rename_column("score", "labels")
```

## Cell 5 - Create regression model and metrics

```python
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.clip(predictions.reshape(-1), 0, 1)
    labels = labels.reshape(-1)

    correlation = (
        pearsonr(labels, predictions).statistic
        if len(np.unique(labels)) > 1
        else 0.0
    )

    return {
        "mae": mean_absolute_error(labels, predictions),
        "rmse": mean_squared_error(labels, predictions) ** 0.5,
        "pearson": correlation,
    }
```

## Cell 6 - Train BERT

```python
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=str(BASE_PATH / "models/bert/checkpoints"),
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="pearson",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

## Cell 7 - Save model and training log

```python
FINAL_MODEL_PATH = BASE_PATH / "models/bert/final"

trainer.save_model(str(FINAL_MODEL_PATH))
tokenizer.save_pretrained(str(FINAL_MODEL_PATH))

pd.DataFrame(trainer.state.log_history).to_csv(
    BASE_PATH / "outputs/bert_training_log.csv",
    index=False,
)
```

---

# Notebook 5: `05_feedback_fewshot.ipynb`

## Cell 1 - Install package and mount Drive

```python
!pip -q install openai

from google.colab import drive
drive.mount("/content/drive")
```

## Cell 2 - Enter API key securely

```python
from getpass import getpass
from openai import OpenAI

client = OpenAI(api_key=getpass("Enter OpenAI API key: "))
```

## Cell 3 - Feedback generation

Use a structured prompt. Do not ask the model to reveal private chain-of-thought.

```python
import json

def generate_feedback(
    resume,
    jd,
    match_score,
    matched_skills,
    missing_skills,
    few_shot_examples="",
):
    prompt = f"""
You are an HR screening assistant. Evaluate the candidate only from the
provided resume and job description. Do not invent experience or skills.

{few_shot_examples}

JOB DESCRIPTION:
{jd}

RESUME:
{resume}

MODEL MATCH SCORE:
{match_score:.2f}

DETECTED MATCHED SKILLS:
{matched_skills}

DETECTED MISSING SKILLS:
{missing_skills}

Return valid JSON with these keys:
- overall_match
- strengths
- matched_skills
- missing_skills
- recommendations
- limitations
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text
```

## Cell 4 - Few-shot example for a niche role

```python
NICHE_EXAMPLES = """
EXAMPLE:
Role: Quantum Computing Researcher
Evidence: Resume includes Qiskit, quantum algorithms, and two publications.
Missing evidence: Error correction and production quantum hardware experience.
Expected assessment: Strong research fit with clearly stated technical gaps.
"""
```

## Cell 5 - Test feedback

```python
sample_resume = """
Machine learning engineer experienced with Python, PyTorch, Docker,
and deploying NLP models.
"""

sample_jd = """
Seeking an NLP engineer with Python, PyTorch, Docker, Kubernetes,
transformers, and AWS experience.
"""

print(generate_feedback(
    resume=sample_resume,
    jd=sample_jd,
    match_score=0.72,
    matched_skills=["python", "pytorch", "docker"],
    missing_skills=["kubernetes", "aws"],
    few_shot_examples=NICHE_EXAMPLES,
))
```

---

# Notebook 6: `06_evaluation.ipynb`

## Cell 1 - Install packages and load model/data

```python
!pip -q install transformers datasets accelerate pandas pyarrow scipy scikit-learn

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import pandas as pd
import numpy as np
import torch

BASE_PATH = Path("/content/drive/MyDrive/AI-Resume-Screener")

baseline_df = pd.read_parquet(BASE_PATH / "outputs/baseline_predictions.parquet")
```

## Cell 2 - Load trained BERT

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = str(BASE_PATH / "models/bert/final")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```

## Cell 3 - Generate BERT predictions

```python
def predict_bert_scores(resumes, jds, batch_size=16):
    scores = []

    for start in range(0, len(resumes), batch_size):
        batch_resumes = resumes[start:start + batch_size]
        batch_jds = jds[start:start + batch_size]

        encoded = tokenizer(
            batch_resumes,
            batch_jds,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits.reshape(-1)

        scores.extend(torch.clamp(logits, 0, 1).cpu().numpy().tolist())

    return scores

baseline_df["bert_score"] = predict_bert_scores(
    baseline_df["resume"].tolist(),
    baseline_df["jd"].tolist(),
)
```

## Cell 4 - Regression evaluation

```python
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_regression(true_scores, predicted_scores):
    return {
        "MAE": mean_absolute_error(true_scores, predicted_scores),
        "RMSE": mean_squared_error(true_scores, predicted_scores) ** 0.5,
        "Pearson": pearsonr(true_scores, predicted_scores).statistic,
    }

model_results = pd.DataFrame({
    "TF-IDF": evaluate_regression(
        baseline_df["score"],
        baseline_df["tfidf_score"],
    ),
    "Word2Vec": evaluate_regression(
        baseline_df["score"],
        baseline_df["word2vec_score"],
    ),
    "BERT": evaluate_regression(
        baseline_df["score"],
        baseline_df["bert_score"],
    ),
}).T

display(model_results)
```

## Cell 5 - Classification-style evaluation

```python
from sklearn.metrics import classification_report

def score_to_label(score):
    if score < 0.25:
        return "No Fit"
    if score < 0.75:
        return "Potential Fit"
    return "Good Fit"

baseline_df["bert_predicted_label"] = baseline_df["bert_score"].map(score_to_label)

print(classification_report(
    baseline_df["label"],
    baseline_df["bert_predicted_label"],
    digits=3,
))
```

## Cell 6 - Save results

```python
model_results.to_csv(BASE_PATH / "outputs/model_comparison.csv")
baseline_df.to_parquet(BASE_PATH / "outputs/final_predictions.parquet", index=False)
```

For feedback quality, ask at least two human reviewers to score a sample of
reports for correctness, usefulness, clarity, and hallucination risk.

---

# Notebook 7: `07_demo.ipynb`

## Cell 1 - Install and load dependencies

```python
!pip -q install transformers torch spacy pdfplumber openai
!python -m spacy download en_core_web_sm

from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from getpass import getpass
import torch
import spacy

BASE_PATH = Path("/content/drive/MyDrive/AI-Resume-Screener")
```

## Cell 2 - Load BERT model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = str(BASE_PATH / "models/bert/final")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```

## Cell 3 - Skill extraction and match scoring

```python
from spacy.matcher import PhraseMatcher

SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#",
    "sql", "mysql", "postgresql", "mongodb", "machine learning",
    "deep learning", "natural language processing", "nlp",
    "tensorflow", "pytorch", "scikit-learn", "docker", "kubernetes",
    "git", "linux", "aws", "azure", "gcp", "react", "node.js",
    "django", "flask", "fastapi", "power bi", "tableau", "excel",
    "project management", "communication", "leadership",
]

skill_nlp = spacy.blank("en")
matcher = PhraseMatcher(skill_nlp.vocab, attr="LOWER")
matcher.add("SKILL", [skill_nlp.make_doc(skill) for skill in SKILLS])

def extract_skills(text):
    doc = skill_nlp(str(text))
    return sorted({
        doc[start:end].text.lower()
        for _, start, end in matcher(doc)
    })

def get_match_score(resume, jd):
    encoded = tokenizer(
        resume,
        jd,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        score = model(**encoded).logits.item()

    return max(0.0, min(1.0, score))

def analyze_resume(resume, jd):
    resume_skills = set(extract_skills(resume))
    jd_skills = set(extract_skills(jd))

    return {
        "match_score": get_match_score(resume, jd),
        "matched_skills": sorted(resume_skills & jd_skills),
        "missing_skills": sorted(jd_skills - resume_skills),
    }
```

## Cell 4 - Optional PDF upload

```python
from google.colab import files
import pdfplumber

def upload_and_extract_pdf():
    uploaded = files.upload()
    filename = next(iter(uploaded))

    text = []
    with pdfplumber.open(filename) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    return "\n".join(text)

# Uncomment to upload a resume PDF:
# resume_text = upload_and_extract_pdf()
```

## Cell 5 - Run the demo

```python
resume_text = """
Paste resume text here, or use the PDF upload cell.
"""

job_description = """
Paste job description here.
"""

result = analyze_resume(resume_text, job_description)

print(f"Match score: {result['match_score'] * 100:.1f}%")
print("Matched skills:", result["matched_skills"])
print("Missing skills:", result["missing_skills"])
```

## Cell 6 - Optional LLM feedback

```python
from openai import OpenAI

client = OpenAI(api_key=getpass("Enter OpenAI API key: "))

def generate_demo_feedback(resume, jd, analysis):
    prompt = f"""
Act as an HR screening assistant. Use only the supplied evidence.

Job description:
{jd}

Resume:
{resume}

Match score: {analysis["match_score"]:.2f}
Matched skills: {analysis["matched_skills"]}
Missing skills: {analysis["missing_skills"]}

Write concise sections titled Overall Match, Strengths, Gaps,
Recommendations, and Limitations.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    return response.output_text

print(generate_demo_feedback(resume_text, job_description, result))
```

---

# Important Notes for the Report

- The primary paired dataset's labels were generated by an LLM, not by human
  recruiters. State this as a limitation.
- Mapping `No Fit`, `Potential Fit`, and `Good Fit` to `0.0`, `0.5`, and `1.0`
  creates an ordinal regression target. Explain this design decision.
- BERT receives raw text. Cleaned and lemmatized text is used only for
  traditional NLP baselines and skill extraction.
- Because BERT accepts at most 512 tokens, long resumes and JDs are truncated.
  Mention this limitation. A later improvement could score text sections or
  use a long-context model.
- Use human reviewers for qualitative feedback evaluation and for checking
  bias, unsupported claims, and hallucinations.
