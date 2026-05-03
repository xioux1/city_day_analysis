from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, task: str) -> tuple[dict[str, Any], pd.DataFrame]:
    y_pred = model.predict(X_test)
    pred_df = pd.DataFrame({"actual": y_test.values, "predicted": y_pred})

    if task == "classification":
        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
            except Exception:
                pass
        return metrics, pred_df

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, y_pred)),
    }
    return metrics, pred_df
