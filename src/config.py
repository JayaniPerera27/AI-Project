from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
BERT_MODEL_DIR = MODELS_DIR / "bert" / "final"
WORD2VEC_MODEL_PATH = MODELS_DIR / "word2vec" / "word2vec.model"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
SKILLS_PATH = MODELS_DIR / "skills.json"

MAX_BERT_LENGTH = 512

