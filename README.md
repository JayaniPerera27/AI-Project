# ResumeFit AI

A Streamlit web application for the AI-Powered Job Description Analyzer and
Resume Screener project. It accepts a resume and job description, extracts
skills, calculates match signals, and produces an explainable feedback report related to the uploaded resume.

## Run the application

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`

The application works without trained models using a TF-IDF fallback. It
automatically enables trained models when their artifacts are added.

## Add trained Colab models

Download the following folders/files from:

```text
Google Drive/MyDrive/AI-Resume-Screener/models/
```

Place them in this local structure:

```text
AI-Project/
├── models/
│   ├── bert/
│   │   └── final/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       ├── special_tokens_map.json
│   │       └── vocab.txt
│   ├── word2vec/
│   │   └── word2vec.model
│   ├── tfidf_vectorizer.joblib
│   └── skills.json
├── src/
└── app.py
```

`skills.json` is optional and should contain a JSON array:

```json
["python", "sql", "docker", "machine learning"]
```

Restart Streamlit after adding or replacing model artifacts.

If you want to use BERT or Word2Vec locally, also install the optional model
dependencies:

```powershell
pip install -r requirements-models.txt
```

For free cloud deployment, keep the large trained model files out of Git. The
app will still run with the TF-IDF fallback and skill matching.

## Deploy the application

### Recommended: Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to `https://share.streamlit.io` and sign in with GitHub.
3. Click **Create app**.
4. Select your GitHub repository, branch, and set the main file path to
   `app.py`.
5. In **Advanced settings**, choose a supported Python version and add secrets
   only if needed.
6. Click **Deploy**.

Optional secret for AI feedback:

```toml
OPENAI_API_KEY = "your_key_here"
OPENAI_MODEL = "gpt-4.1-mini"
```

### Alternative: Render

Create a Render **Web Service** from the GitHub repository.

Use these settings:

```text
Build Command: pip install -r requirements.txt
Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Add `OPENAI_API_KEY` and `OPENAI_MODEL` as environment variables only if you
want AI feedback.

## Optional OpenAI feedback

Copy `.env.example` to `.env`, then add:

```text
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4.1-mini
```

Without an API key, the app creates a local structured feedback report.

## Scoring behavior

- With BERT: `80% BERT + 20% skill coverage`
- With Word2Vec but no BERT: `45% TF-IDF + 35% Word2Vec + 20% skill coverage`
- Without trained models: `75% pair-level TF-IDF + 25% skill coverage`

The score provides decision support only and must not replace human hiring
judgment.

