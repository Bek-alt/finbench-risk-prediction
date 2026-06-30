"""
baseline_models.py
Train and evaluate traditional ML baselines (Logistic Regression, Random Forest,
Gradient Boosting, Neural Network/MLP) on FinBench tabular data.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from data_utils import build_preprocessor


def get_baseline_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=500, random_state=42),
    }


def evaluate_predictions(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None:
        # y_prob can be (n, 2) from predict_proba, or already a 1D positive-class score
        pos_scores = y_prob[:, 1] if getattr(y_prob, "ndim", 1) == 2 else y_prob
        metrics["AUC-ROC"] = roc_auc_score(y_true, pos_scores)
    else:
        metrics["AUC-ROC"] = None
    return metrics


def train_and_evaluate_baselines(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    numerical_cols: list,
    categorical_cols: list,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train all baseline models inside a preprocessing pipeline and return a results table."""
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    models = get_baseline_models()

    results = {}
    for name, model in models.items():
        if verbose:
            print(f"Training {name}...")

        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        results[name] = metrics

        if verbose:
            print(classification_report(y_test, y_pred, zero_division=0))

    return pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]]
