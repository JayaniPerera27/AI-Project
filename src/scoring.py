from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import BERT_MODEL_DIR, MAX_BERT_LENGTH, TFIDF_MODEL_PATH, WORD2VEC_MODEL_PATH


def _clean_for_baseline(text):
    return " ".join(str(text).lower().split())


@lru_cache(maxsize=1)
def load_tfidf_model():
    if not TFIDF_MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(TFIDF_MODEL_PATH)


@lru_cache(maxsize=1)
def load_word2vec_model():
    if not WORD2VEC_MODEL_PATH.exists():
        return None
    from gensim.models import Word2Vec
    return Word2Vec.load(str(WORD2VEC_MODEL_PATH))


@lru_cache(maxsize=1)
def load_bert_model():
    if not BERT_MODEL_DIR.exists() or not (BERT_MODEL_DIR / "config.json").exists():
        return None, None, None

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def tfidf_score(resume_text, job_description):
    vectorizer = load_tfidf_model()
    resume = _clean_for_baseline(resume_text)
    job = _clean_for_baseline(job_description)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = vectorizer.fit_transform([resume, job])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0]), "Pair TF-IDF fallback"

    resume_vector = vectorizer.transform([resume])
    job_vector = vectorizer.transform([job])
    return float(cosine_similarity(resume_vector, job_vector)[0][0]), "Trained TF-IDF"


def word2vec_score(resume_text, job_description):
    model = load_word2vec_model()
    if model is None:
        return None

    def document_vector(text):
        vectors = [
            model.wv[word]
            for word in _clean_for_baseline(text).split()
            if word in model.wv
        ]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    left = document_vector(resume_text)
    right = document_vector(job_description)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def bert_score(resume_text, job_description):
    tokenizer, model, device = load_bert_model()
    if model is None:
        return None

    import torch

    encoded = tokenizer(
        resume_text,
        job_description,
        truncation=True,
        max_length=MAX_BERT_LENGTH,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        score = model(**encoded).logits.reshape(-1)[0].item()

    return max(0.0, min(1.0, score))


def score_resume(resume_text, job_description, skill_coverage):
    tfidf, tfidf_source = tfidf_score(resume_text, job_description)
    word2vec = word2vec_score(resume_text, job_description)
    bert = bert_score(resume_text, job_description)

    if bert is not None:
        overall = 0.80 * bert + 0.20 * skill_coverage
        source = "Fine-tuned BERT + skill coverage"
    elif word2vec is not None:
        overall = 0.45 * tfidf + 0.35 * word2vec + 0.20 * skill_coverage
        source = "TF-IDF + Word2Vec + skill coverage"
    else:
        overall = 0.75 * tfidf + 0.25 * skill_coverage
        source = "TF-IDF fallback + skill coverage"

    return {
        "overall": max(0.0, min(1.0, overall)),
        "bert": bert,
        "tfidf": tfidf,
        "tfidf_source": tfidf_source,
        "word2vec": word2vec,
        "source": source,
    }


def model_status():
    return {
        "BERT": BERT_MODEL_DIR.exists() and (BERT_MODEL_DIR / "config.json").exists(),
        "Word2Vec": WORD2VEC_MODEL_PATH.exists(),
        "Trained TF-IDF": TFIDF_MODEL_PATH.exists(),
    }

