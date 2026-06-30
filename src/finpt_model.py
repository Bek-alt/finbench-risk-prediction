"""
finpt_model.py
A FinPT-like approach: convert each tabular row into a natural-language
customer profile, embed it with a sentence transformer, and classify on
the resulting embedding.

NOTE ON A BUG FIXED HERE:
In the original course notebook, rows were converted to text *after*
StandardScaler had already transformed the numeric columns, producing
profiles like "income is -0.27" instead of "income is 45000". That
defeats the purpose of a natural-language profile (the FinPT paper
converts the *raw* human-readable values into text) and likely
understated the FinPT-like model's performance. This module always
builds text profiles from raw, untransformed values.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from baseline_models import evaluate_predictions


def row_to_text(row: pd.Series) -> str:
    """Convert one raw tabular row into a simple natural-language profile sentence."""
    parts = [f"{col} is {val}" for col, val in row.items()]
    return ", ".join(parts) + "."


def dataframe_to_profiles(X_raw: pd.DataFrame) -> pd.Series:
    return X_raw.apply(row_to_text, axis=1)


def embed_profiles(texts: pd.Series, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                    encoder: SentenceTransformer = None) -> np.ndarray:
    """Embed text profiles with a sentence-transformer. Pass `encoder` to reuse a loaded model."""
    if encoder is None:
        encoder = SentenceTransformer(model_name)
    return encoder.encode(texts.tolist(), show_progress_bar=True)


def train_and_evaluate_finpt_like(
    X_train_raw: pd.DataFrame,
    y_train,
    X_test_raw: pd.DataFrame,
    y_test,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    verbose: bool = True,
) -> dict:
    """
    Full FinPT-like pipeline: raw rows -> text profiles -> sentence embeddings ->
    logistic regression classifier -> evaluation metrics.
    """
    if verbose:
        print("Converting rows to natural-language profiles...")
    train_texts = dataframe_to_profiles(X_train_raw)
    test_texts = dataframe_to_profiles(X_test_raw)

    if verbose:
        print("Example profile:", train_texts.iloc[0])
        print(f"Loading sentence transformer: {model_name}")

    encoder = SentenceTransformer(model_name)
    X_train_emb = embed_profiles(train_texts, encoder=encoder)
    X_test_emb = embed_profiles(test_texts, encoder=encoder)

    if verbose:
        print("Embedding shape:", X_train_emb.shape)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)
    y_prob = clf.predict_proba(X_test_emb) if hasattr(clf, "predict_proba") else None

    return evaluate_predictions(y_test, y_pred, y_prob)
