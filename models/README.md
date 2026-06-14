# Trained Model Placement

Copy the model artifacts exported from Google Drive/Colab into this folder.

```text
models/
├── bert/
│   └── final/
│       ├── config.json
│       ├── model.safetensors        # or pytorch_model.bin
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       └── vocab.txt
├── word2vec/
│   └── word2vec.model
├── tfidf_vectorizer.joblib
└── skills.json                      # optional custom skill list
```

The application automatically detects available artifacts when it starts.
Without trained artifacts, it remains usable through a per-document TF-IDF
fallback score.

