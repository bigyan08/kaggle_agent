# The ML core. No LLM here: the agent only chooses the target column, plain Python does the rest.
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app import guardrails as g
from app.config import MAX_ROWS, RUNS_DIR


def load_table(path: Path) -> pd.DataFrame:
    """Read a tabular file, sampling down to MAX_ROWS rows if it is larger."""
    extension = path.suffix.lower()
    if extension == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif extension == ".tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False)
    elif extension in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif extension == ".parquet":
        df = pd.read_parquet(path)
    elif extension == ".json":
        df = pd.read_json(path)
    else:
        raise g.GuardrailViolation(f"Unsupported file type '{extension}'.")
    return df.sample(MAX_ROWS, random_state=42) if len(df) > MAX_ROWS else df


def infer_task_type(target: pd.Series) -> str:
    """Text/bool targets or targets with few distinct values are classification."""
    if not pd.api.types.is_numeric_dtype(target) or target.dtype == bool or target.nunique() <= 20:
        return "classification"
    return "regression"


def prepare_data(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows without a target and ID-like text columns, then split into X and y."""
    g.check_target_column(df, target_column)
    df = df.dropna(subset=[target_column])
    g.check_enough_rows(len(df))
    X = df.drop(columns=[target_column])
    id_like = [c for c in X.columns if X[c].dtype == object and X[c].nunique() == len(X)]
    return X.drop(columns=id_like), df[target_column]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numbers: fill gaps + scale. Everything else: fill gaps + one-hot encode."""
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
        ]
    )
    numeric_columns = X.select_dtypes("number").columns
    other_columns = X.columns.difference(numeric_columns)
    return ColumnTransformer([("numeric", numeric, numeric_columns), ("categorical", categorical, other_columns)])


def baseline_models(task_type: str) -> dict:
    """Three models: a Dummy (the 'does it beat guessing?' reference), a linear model, a random forest."""
    if task_type == "classification":
        return {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        }
    return {
        "dummy": DummyRegressor(strategy="mean"),
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }


def evaluate(task_type: str, y_true, y_pred) -> dict:
    """Metrics for one model. 'score' (higher = better) is what we rank models by."""
    if task_type == "classification":
        f1 = f1_score(y_true, y_pred, average="macro")
        return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1, "score": f1}
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {"rmse": rmse, "mae": mean_absolute_error(y_true, y_pred), "r2": r2, "score": r2}


def train_baselines(df: pd.DataFrame, target_column: str, task_type: str = "auto") -> tuple[list[dict], dict]:
    """Train and evaluate the baselines. Returns (results, fitted pipelines by model name)."""
    X, y = prepare_data(df, target_column)
    if task_type == "auto":
        task_type = infer_task_type(y)
    if task_type == "classification":
        g.check_classes(y)

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    results, pipelines = [], {}
    for name, model in baseline_models(task_type).items():
        pipeline = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(task_type, y_test, pipeline.predict(X_test))
        results.append({"model": name, "task_type": task_type, **{k: round(v, 4) for k, v in metrics.items()}})
        pipelines[name] = pipeline
    return results, pipelines


def save_run(results: list[dict], pipelines: dict, dataset_ref: str, target_column: str) -> str:
    """Save results.json and the best model under data/runs/<run_id>/. Returns the run_id."""
    best = max(results, key=lambda r: r["score"])
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{dataset_ref.replace('/', '__')}"
    folder = RUNS_DIR / run_id
    folder.mkdir(parents=True)
    summary = {"dataset": dataset_ref, "target": target_column, "best_model": best["model"], "results": results}
    (folder / "results.json").write_text(json.dumps(summary, indent=2))
    joblib.dump(pipelines[best["model"]], folder / "model.joblib")
    return run_id
