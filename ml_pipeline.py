"""Aplicación del notebook de clase al dataset city_day.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_PATH = Path("city_day.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")
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
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    base_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(50,),
                    activation="relu",
                    alpha=0.0001,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\n=== Clasificación con enfoque notebook ===")
    print("(extraido del notebook de clase) Modelo base MLPClassifier + Pipeline")

    base_pipeline.fit(X_train, y_train)
    y_pred_base = base_pipeline.predict(X_test)
    print(f"Accuracy test (base): {accuracy_score(y_test, y_pred_base):.4f}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scores_cv_base = cross_val_score(base_pipeline, X, y, cv=cv, scoring="accuracy")
    print("(extraido del notebook de clase) Validación cruzada con accuracy")
    print(f"CV mean: {scores_cv_base.mean():.4f} | std: {scores_cv_base.std():.4f}")

    param_grid_clf = {
        "mlp__hidden_layer_sizes": [(20,), (50,), (100,), (50, 50)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
    }

    combinaciones = list(ParameterGrid(param_grid_clf))
    print("(extraido del notebook de clase) Combinaciones ParameterGrid:", len(combinaciones))

    grid_search_clf = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid_clf,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search_clf.fit(X, y)

    best_model_grid = grid_search_clf.best_estimator_
    y_pred_grid = best_model_grid.predict(X_test)

    param_dist_clf = {
        "mlp__hidden_layer_sizes": [(20,), (50,), (100,), (150,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__alpha": np.logspace(-5, -1, 20),
        "mlp__learning_rate_init": np.logspace(-4, -1, 20),
    }

    random_search_clf = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist_clf,
        n_iter=15,
        scoring="accuracy",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )
    random_search_clf.fit(X_train, y_train)

    best_model_random = random_search_clf.best_estimator_
    y_pred_random = best_model_random.predict(X_test)

    print("\n=== Comparación final clasificación ===")
    print(f"Grid best params: {grid_search_clf.best_params_}")
    print(f"Random best params: {random_search_clf.best_params_}")
    print(
        pd.DataFrame(
            {
                "Modelo": ["Base", "GridSearchCV", "RandomizedSearchCV"],
                "Score_validacion": [
                    scores_cv_base.mean(),
                    grid_search_clf.best_score_,
                    random_search_clf.best_score_,
                ],
                "Score_test": [
                    accuracy_score(y_test, y_pred_base),
                    accuracy_score(y_test, y_pred_grid),
                    accuracy_score(y_test, y_pred_random),
                ],
            }
        )
    )
    print("Matriz de confusión (mejor Grid):")
    print(confusion_matrix(y_test, y_pred_grid))
    print(classification_report(y_test, y_pred_grid))


def run_regression(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)
    base_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(50,),
                    activation="relu",
                    alpha=0.0001,
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\n=== Regresión con enfoque notebook ===")
    print("(extraido del notebook de clase) Modelo base MLPRegressor + Pipeline")

    base_pipeline.fit(X_train, y_train)
    y_pred_base = base_pipeline.predict(X_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores_cv_reg = cross_val_score(base_pipeline, X, y, cv=cv, scoring="r2")

    param_grid_reg = {
        "mlp__hidden_layer_sizes": [(50,), (100,)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001],
    }
    grid_search_reg = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid_reg,
        scoring="r2",
        cv=3,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search_reg.fit(X, y)

    best_model_grid = grid_search_reg.best_estimator_
    y_pred_grid = best_model_grid.predict(X_test)

    param_dist_reg = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (150,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh", "logistic"],
        "mlp__alpha": np.logspace(-5, -1, 20),
        "mlp__learning_rate_init": np.logspace(-4, -2, 20),
    }
    random_search_reg = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist_reg,
        n_iter=8,
        scoring="r2",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )
    random_search_reg.fit(X_train, y_train)

    best_model_random = random_search_reg.best_estimator_
    y_pred_random = best_model_random.predict(X_test)

    print("(extraido del notebook de clase) CV R2 base:", round(scores_cv_reg.mean(), 4))
    print("\n=== Comparación final regresión ===")
    print(
        pd.DataFrame(
            {
                "Modelo": ["Base", "GridSearchCV", "RandomizedSearchCV"],
                "Score_validacion_R2": [
                    scores_cv_reg.mean(),
                    grid_search_reg.best_score_,
                    random_search_reg.best_score_,
                ],
                "Score_test_R2": [
                    r2_score(y_test, y_pred_base),
                    r2_score(y_test, y_pred_grid),
                    r2_score(y_test, y_pred_random),
                ],
            }
        )
    )
    print(f"MAE Base: {mean_absolute_error(y_test, y_pred_base):.4f}")
    print(f"RMSE Base: {np.sqrt(mean_squared_error(y_test, y_pred_base)):.4f}")


def main() -> None:
    df = load_data(DATA_PATH)
    target_col = df.columns[-1]

    print(f"Dataset shape: {df.shape}")
    print(f"Target seleccionado: {target_col}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    print("(ya estaba implementado, coincide con notebook) Uso de Pipeline para evitar fugas.")
    problem_type = infer_problem_type(y)
    print(f"Tipo inferido: {problem_type}")

    if problem_type == "classification":
        run_classification(X, y)
    else:
        run_regression(X, y)


if __name__ == "__main__":
    main()
