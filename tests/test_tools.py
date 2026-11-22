import pytest
from sklearn.datasets import load_iris

from app import ml, tools


@pytest.fixture
def iris_dataset(monkeypatch, tmp_path):
    """A fake downloaded dataset 'o/iris' containing iris.csv."""
    monkeypatch.setattr(tools, "DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(ml, "RUNS_DIR", tmp_path)
    folder = tools.dataset_folder("o/iris")
    folder.mkdir(parents=True)
    load_iris(as_frame=True).frame.to_csv(folder / "iris.csv", index=False)


def test_inspect_file_lists_columns(iris_dataset):
    output = tools.inspect_file.invoke({"dataset_ref": "o/iris", "file_name": "iris.csv"})
    assert "150 rows x 5 columns" in output
    assert "target" in output


def test_train_baselines_returns_metrics_and_run_id(iris_dataset):
    args = {"dataset_ref": "o/iris", "file_name": "iris.csv", "target_column": "target"}
    output = tools.train_baselines.invoke(args)
    assert "random_forest" in output
    assert "Run id:" in output


def test_errors_are_returned_as_text_not_raised(iris_dataset):
    bad_column = {"dataset_ref": "o/iris", "file_name": "iris.csv", "target_column": "nope"}
    assert tools.train_baselines.invoke(bad_column).startswith("Blocked by guardrail")
    escape = {"dataset_ref": "o/iris", "file_name": "../../secret.csv"}
    assert tools.inspect_file.invoke(escape).startswith("Blocked by guardrail")
