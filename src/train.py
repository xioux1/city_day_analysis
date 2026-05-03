from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .features import build_preprocessor
from .models import build_model


def train_model(X: pd.DataFrame, y: pd.Series, task: str, split_cfg: dict[str, Any], model_cfg: dict[str, Any]):
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X_train)
    model = build_model(task=task, model_type=model_cfg["type"], params=model_cfg["params"])
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])

    pipeline.fit(X_train, y_train)
    return {
        "model": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
