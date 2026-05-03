"""End-to-end dataset analysis and model selection pipeline for city_day.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
DATA_PATH = Path("city_day.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def infer_problem_type(series: pd.Series) -> str:
    unique_values = series.nunique(dropna=True)
    if series.dtype == "object" or unique_values <= 20:
        return "classification"
    return "regression"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def run_classification(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1_weighted", "roc_auc_ovr_weighted"],
        return_train_score=True,
    )

    print("=== Cross-validation (classification) ===")
    for metric in ["test_accuracy", "test_f1_weighted", "test_roc_auc_ovr_weighted"]:
        print(f"{metric}: mean={scores[metric].mean():.4f}, std={scores[metric].std():.4f}")

    search_space = [
        {
            "model": [LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)],
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        {
            "model": [RandomForestClassifier(random_state=RANDOM_STATE)],
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 8, 16],
            "model__class_weight": [None, "balanced"],
        },
    ]

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=search_space,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n=== Final test (classification) ===")
    print(f"Best params: {grid.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    if y.nunique() == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))


def run_regression(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["r2", "neg_mean_absolute_error"],
        return_train_score=True,
    )

    print("=== Cross-validation (regression) ===")
    print(f"test_r2: mean={scores['test_r2'].mean():.4f}, std={scores['test_r2'].std():.4f}")
    print(
        "test_mae: mean="
        f"{-scores['test_neg_mean_absolute_error'].mean():.4f}, "
        f"std={scores['test_neg_mean_absolute_error'].std():.4f}"
    )

    param_distributions = {
        "model": [RandomForestRegressor(random_state=RANDOM_STATE)],
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [None, 8, 16, 32],
        "model__min_samples_split": [2, 5, 10],
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n=== Final test (regression) ===")
    print(f"Best params: {random_search.best_params_}")
    print(f"R2: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")


def main() -> None:
    df = load_data(DATA_PATH)
    target_col = df.columns[-1]

    print(f"Dataset shape: {df.shape}")
    print(f"Target column selected: {target_col}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    problem_type = infer_problem_type(y)
    print(f"Inferred task type: {problem_type}")

    if problem_type == "classification":
        run_classification(X, y)
    else:
        run_regression(X, y)


if __name__ == "__main__":
    main()
