import json

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from app import guardrails as g
from app import ml


@pytest.fixture
def iris() -> pd.DataFrame:
    return load_iris(as_frame=True).frame  # target column: "target" (3 classes)


@pytest.fixture
def diabetes() -> pd.DataFrame:
    return load_diabetes(as_frame=True).frame  # target column: "target" (numbers)


def test_infer_task_type(iris, diabetes):
    assert ml.infer_task_type(iris["target"]) == "classification"
    assert ml.infer_task_type(diabetes["target"]) == "regression"
    assert ml.infer_task_type(pd.Series(["a", "b", "a"])) == "classification"


def test_load_table_reads_csv(iris, tmp_path):
    path = tmp_path / "iris.csv"
    iris.to_csv(path, index=False)
    assert len(ml.load_table(path)) == 150


def test_classification_trains_three_models_and_beats_dummy(iris):
    results, _ = ml.train_baselines(iris, "target")
    scores = {r["model"]: r["score"] for r in results}
    assert set(scores) == {"dummy", "logistic_regression", "random_forest"}
    assert scores["random_forest"] > scores["dummy"]


def test_regression_trains_three_models_and_beats_dummy(diabetes):
    results, _ = ml.train_baselines(diabetes, "target")
    scores = {r["model"]: r["score"] for r in results}
    assert set(scores) == {"dummy", "ridge", "random_forest"}
    assert scores["random_forest"] > scores["dummy"]


def test_handles_text_and_missing_values(iris):
    iris = iris.copy()
    iris["color"] = ["red", "green", "blue"] * 50
    iris.loc[::7, "sepal length (cm)"] = None
    results, _ = ml.train_baselines(iris, "target")
    assert len(results) == 3


def test_unknown_target_is_blocked(iris):
    with pytest.raises(g.GuardrailViolation):
        ml.train_baselines(iris, "nope")


def test_too_few_rows_is_blocked(iris):
    with pytest.raises(g.GuardrailViolation):
        ml.train_baselines(iris.head(10), "target")


def test_single_class_is_blocked(iris):
    iris = iris.copy()
    iris["target"] = 0
    with pytest.raises(g.GuardrailViolation):
        ml.train_baselines(iris, "target", task_type="classification")


def test_save_run_writes_files(iris, tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "RUNS_DIR", tmp_path)
    results, pipelines = ml.train_baselines(iris, "target")
    run_id = ml.save_run(results, pipelines, "owner/iris", "target")
    assert (tmp_path / run_id / "model.joblib").exists()
    assert json.loads((tmp_path / run_id / "results.json").read_text())["target"] == "target"
