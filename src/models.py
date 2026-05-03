from __future__ import annotations

from typing import Any

from sklearn.neural_network import MLPClassifier, MLPRegressor


def build_model(task: str, model_type: str, params: dict[str, Any]):
    if model_type != "mlp":
        raise ValueError(f"Unsupported model type: {model_type}")
    if task == "classification":
        return MLPClassifier(**params)
    if task == "regression":
        return MLPRegressor(**params)
    raise ValueError(f"Unsupported task: {task}")
