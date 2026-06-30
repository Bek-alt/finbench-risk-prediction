"""
data_utils.py
Loading and preprocessing utilities for the FinBench dataset
(https://huggingface.co/datasets/yuweiyin/FinBench).
"""

from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class FinBenchData:
    """Container for a loaded FinBench task split."""
    X_raw: pd.DataFrame          # untouched feature values (used for FinPT text profiles)
    y: list
    numerical_cols: list
    categorical_cols: list


def load_finbench_split(dataset_name: str, split: str = "train") -> FinBenchData:
    """
    Load a single split (train/val/test) of a FinBench task (e.g. 'cd2', 'ld2').

    Returns the *raw* (unscaled, unencoded) feature dataframe alongside the
    numerical/categorical column lists, so that downstream code can choose
    to scale/encode for sklearn models, or keep raw values for natural-language
    profile generation (FinPT-like pipeline).
    """
    dataset = load_dataset("yuweiyin/FinBench", dataset_name, trust_remote_code=True)

    X = pd.DataFrame(dataset[split]["X_ml"])
    y = dataset[split]["y"]

    meta = dataset[split][0]  # each row carries col_name / num_idx / cat_idx metadata
    col_names = meta["col_name"]
    num_idx = meta["num_idx"]
    cat_idx = meta["cat_idx"]

    X.columns = col_names
    numerical_cols = [col_names[i] for i in num_idx]
    categorical_cols = [col_names[i] for i in cat_idx]

    return FinBenchData(X_raw=X, y=y, numerical_cols=numerical_cols, categorical_cols=categorical_cols)


def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Standard sklearn preprocessing pipeline: scale numeric, one-hot encode categorical."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
